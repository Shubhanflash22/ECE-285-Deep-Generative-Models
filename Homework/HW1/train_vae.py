"""
HW1: Variational Autoencoder on the Chest X-ray Pneumonia dataset.

Usage:
    pip install torch torchvision scipy matplotlib numpy
    # download + unzip https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    python train_vae.py --data_dir /path/to/chest_xray

Outputs (in --out_dir, default ./outputs):
    vae.pt, loss_curve.pdf, samples.png, results.json, results.tex
results.tex is \\input by report.tex so the report fills itself in.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import save_image


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class VAE(nn.Module):
    """Convolutional VAE for 1x128x128 images."""

    def __init__(self, latent_dim=128, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        layers, c_in = [], 1
        for c in ch:  # 128 -> 64 -> 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
            c_in = c
        self.encoder = nn.Sequential(*layers)
        feat = ch[-1] * 4 * 4
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feat)

        layers, rch = [], list(ch[::-1])  # 256,256,128,64,32
        for i in range(len(rch) - 1):
            layers += [nn.ConvTranspose2d(rch[i], rch[i + 1], 4, 2, 1),
                       nn.BatchNorm2d(rch[i + 1]), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(rch[-1], 1, 4, 2, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.ch[-1], 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        return self.decode(torch.randn(n, self.latent_dim, device=device))


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Negative ELBO (per-sample average): BCE reconstruction + beta * KL."""
    bs = x.size(0)
    rec = F.binary_cross_entropy(recon, x, reduction="sum") / bs
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bs
    return rec + beta * kl, rec, kl


# ----------------------------------------------------------------------------
# FID / Inception Score (Inception-v3 from torchvision, 299x299 input)
# ----------------------------------------------------------------------------
class InceptionFeatures(nn.Module):
    """Returns (2048-d pool features, 1000-d logits) for images in [0,1], 1 or 3 channels."""

    def __init__(self, device):
        super().__init__()
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
        self._feat = None
        self.net.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_feat", o.flatten(1)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        logits = self.net(x)  # torchvision applies transform_input internally
        return self._feat, logits


@torch.no_grad()
def extract(images_iter, model):
    feats, logits = [], []
    for x in images_iter:
        f, l = model(x)
        feats.append(f.cpu().double())
        logits.append(l.cpu().double())
    return torch.cat(feats).numpy(), torch.cat(logits)


def compute_fid(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps).dot(s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def compute_is(logits, splits=10):
    p = F.softmax(logits, dim=1)
    scores = []
    for chunk in p.chunk(splits):
        py = chunk.mean(0, keepdim=True)
        kl = (chunk * (torch.log(chunk + 1e-12) - torch.log(py + 1e-12))).sum(1).mean()
        scores.append(torch.exp(kl).item())
    return float(np.mean(scores)), float(np.std(scores))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="path to chest_xray/ (contains train/ test/ val/)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--img_size", type=int, default=128)  # model architecture assumes 128
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n_eval", type=int, default=2000, help="#real and #generated images for FID/IS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    assert args.img_size == 128, "architecture is built for 128x128 inputs"

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),  # [0,1]
    ])
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=tf)
    test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tf)
    print(f"train: {len(train_ds)} images, test: {len(test_ds)} images")
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VAE(args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {"train": [], "train_rec": [], "train_kl": [], "test": []}
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = rec_t = kl_t = n = 0
        for x, _ in train_dl:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, args.beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            b = x.size(0)
            tot += loss.item() * b; rec_t += rec.item() * b; kl_t += kl.item() * b; n += b
        sched.step()

        model.eval()
        tl = tn = 0
        with torch.no_grad():
            for x, _ in test_dl:
                x = x.to(device)
                recon, mu, logvar = model(x)
                l, _, _ = vae_loss(recon, x, mu, logvar, args.beta)
                tl += l.item() * x.size(0); tn += x.size(0)
        hist["train"].append(tot / n); hist["train_rec"].append(rec_t / n)
        hist["train_kl"].append(kl_t / n); hist["test"].append(tl / tn)
        print(f"epoch {ep:3d}/{args.epochs}  train {tot/n:.2f} (rec {rec_t/n:.2f}, kl {kl_t/n:.2f})  test {tl/tn:.2f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "vae.pt"))

    # ---- loss curves ----
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    ep_axis = np.arange(1, args.epochs + 1)
    ax[0].plot(ep_axis, hist["train"], label="train")
    ax[0].plot(ep_axis, hist["test"], label="test")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("negative ELBO"); ax[0].legend(); ax[0].set_title("Total loss")
    ax[1].plot(ep_axis, hist["train_rec"], label="reconstruction (BCE)")
    ax[1].plot(ep_axis, hist["train_kl"], label="KL")
    ax[1].set_xlabel("epoch"); ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("Train loss components")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.pdf"))
    plt.close()

    # ---- generated samples ----
    model.eval()
    torch.manual_seed(args.seed + 1)
    save_image(model.sample(64, device).cpu(), os.path.join(args.out_dir, "samples.png"), nrow=8, padding=2)
    # real images for visual comparison
    real_vis = torch.stack([train_ds[i][0] for i in np.random.RandomState(0).choice(len(train_ds), 64, replace=False)])
    save_image(real_vis, os.path.join(args.out_dir, "real.png"), nrow=8, padding=2)

    # ---- FID / IS ----
    print("computing FID / IS ...")
    inc = InceptionFeatures(device)
    idx = np.random.RandomState(args.seed).choice(len(train_ds), min(args.n_eval, len(train_ds)), replace=False)

    def real_batches():
        for i in range(0, len(idx), 50):
            yield torch.stack([train_ds[j][0] for j in idx[i:i + 50]]).to(device)

    def fake_batches():
        for i in range(0, args.n_eval, 50):
            yield model.sample(min(50, args.n_eval - i), device)

    f_real, lg_real = extract(real_batches(), inc)
    f_fake, lg_fake = extract(fake_batches(), inc)
    fid = compute_fid(f_real, f_fake)
    is_fake, is_fake_std = compute_is(lg_fake)
    is_real, is_real_std = compute_is(lg_real)
    print(f"FID = {fid:.2f}, IS(generated) = {is_fake:.3f} +- {is_fake_std:.3f}, IS(real) = {is_real:.3f}")

    res = dict(vars(args), n_params=n_params, fid=fid, is_fake=is_fake, is_fake_std=is_fake_std,
               is_real=is_real, is_real_std=is_real_std,
               final_train_loss=hist["train"][-1], final_test_loss=hist["test"][-1], history=hist)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)

    # ---- macros for the LaTeX report ----
    tex = rf"""\renewcommand{{\hpEpochs}}{{{args.epochs}}}
\renewcommand{{\hpBatch}}{{{args.batch_size}}}
\renewcommand{{\hpLR}}{{{args.lr:g}}}
\renewcommand{{\hpLatent}}{{{args.latent_dim}}}
\renewcommand{{\hpBeta}}{{{args.beta:g}}}
\renewcommand{{\hpImg}}{{{args.img_size}}}
\renewcommand{{\hpParams}}{{{n_params/1e6:.1f}M}}
\renewcommand{{\hpNeval}}{{{args.n_eval}}}
\renewcommand{{\resFID}}{{{fid:.2f}}}
\renewcommand{{\resIS}}{{{is_fake:.2f} $\pm$ {is_fake_std:.2f}}}
\renewcommand{{\resISreal}}{{{is_real:.2f} $\pm$ {is_real_std:.2f}}}
\renewcommand{{\resTrainLoss}}{{{hist['train'][-1]:.1f}}}
\renewcommand{{\resTestLoss}}{{{hist['test'][-1]:.1f}}}
"""
    with open(os.path.join(args.out_dir, "results.tex"), "w") as f:
        f.write(tex)
    print("done. Files written to", args.out_dir)


if __name__ == "__main__":
    main()
