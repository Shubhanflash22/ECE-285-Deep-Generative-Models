"""
HW2: DCGAN-style GAN on WikiArt paintings (64x64 or 128x128, taken from the data).

Data: a uint8 array (N,S,S,3), S = 64 or 128, made by preprocess_wikiart.py; the whole array is kept on the GPU.

Usage:
    pip install pytorch-ignite            # inside the venv; used for FID / Inception Score
    python train_gan.py --data wikiart128.npy --out_dir outputs
    python train_gan.py --dummy 512 --epochs 2 --n_eval 500 --out_dir smoke      # quick smoke test, no data needed

Outputs (in --out_dir): gan.pt, loss_curve.pdf, samples.png, real.png, results.json, results.tex

Options that matter: --loss {bce,hinge}, --sn (spectral norm in D, no BatchNorm in D), --ema (generator EMA decay).
FID/IS use pytorch-ignite as in https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/: ignite's default FID
(softmax features, tiny scale, "blog FID"), ignite FID on 2048-d pooled features ("pool FID", standard scale) and
InceptionScore; images resized to 299x299 and ImageNet-normalised (--norm minus1 = the blog's [-1,1] input). Samples are drawn with BatchNorm in batch-statistics mode (batches of 250).
"""
import argparse
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class Generator(nn.Module):
    """z (z_dim) -> 3 x S x S image in [-1,1], S = 64 or 128. 4x4 start, one stride-2 ConvT per doubling."""

    def __init__(self, z_dim=128, ch=64, size=64):
        super().__init__()
        c = ch
        widths = [c * 8, c * 4, c * 2, c] + ([max(c // 2, 8)] if size == 128 else [])   # 4,8,16,32(,64)
        layers = [nn.ConvTranspose2d(z_dim, widths[0], 4, 1, 0, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(True)]
        for i in range(1, len(widths)):
            layers += [nn.ConvTranspose2d(widths[i - 1], widths[i], 4, 2, 1, bias=False),
                       nn.BatchNorm2d(widths[i]), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(widths[-1], 3, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """3 x S x S image -> one logit. BatchNorm, or spectral norm (and no BatchNorm) if sn=True."""

    def __init__(self, ch=64, sn=False, size=64):
        super().__init__()
        c = ch

        def conv(i, o, k, s, p, bias):
            m = nn.Conv2d(i, o, k, s, p, bias=bias)
            nn.init.normal_(m.weight, 0.0, 0.02)
            if bias:
                nn.init.zeros_(m.bias)
            return spectral_norm(m) if sn else m

        widths = [c, c * 2, c * 4, c * 8] + ([c * 8] if size == 128 else [])           # S/2, S/4, ..., 4
        layers = [conv(3, widths[0], 4, 2, 1, True), nn.LeakyReLU(0.2, True)]
        for i in range(1, len(widths)):
            layers.append(conv(widths[i - 1], widths[i], 4, 2, 1, sn))
            if not sn:
                layers.append(nn.BatchNorm2d(widths[i]))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(conv(widths[-1], 1, 4, 1, 0, True))                              # 4x4 -> 1x1
        self.net = nn.Sequential(*layers)
        if not sn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).view(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------
# Sampling and evaluation (FID / IS with pytorch-ignite, as in the course blog)
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(G, n, z_dim, dev, seed, bs=250):
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n - i), z_dim, device=dev, generator=gen)
        out.append(G(z))
    return torch.cat(out)


def to_299(x):
    return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep(x, norm="imagenet"):
    """x in [-1,1] -> input of the Inception network. The torchvision weights expect ImageNet-normalised input
    (norm='imagenet'); the ignite blog feeds [-1,1] directly (norm='minus1')."""
    if norm == "imagenet":
        x = ((x + 1.0) / 2.0 - _MEAN.to(x.device)) / _STD.to(x.device)
    return to_299(x)


def new_fid_metrics(dev):
    """(blog FID, pool FID). Blog FID = ignite's default FID: Inception softmax probabilities (1000-d), tiny scale.
    Pool FID = standard FID on the 2048-d pooled features, using the same ignite FID class."""
    from ignite.metrics import FID
    from ignite.metrics.gan.utils import InceptionModel
    fid_blog = FID(device=dev)
    fid_pool = FID(num_features=2048, feature_extractor=InceptionModel(return_features=True, device=dev), device=dev)
    return fid_blog, fid_pool


def patch_ignite_fid():
    """ignite's FID takes a matrix square root of a 2048x2048 matrix with a slow general routine, which can take
    very long on a CPU-capped machine. Replace it by an equivalent symmetric-eigenvalue computation (same value)."""
    try:
        import ignite.metrics.gan.fid as m
    except Exception:
        return False
    if not hasattr(m, "fid_score"):
        return False

    def fid_score_fast(mu1, mu2, sigma1, sigma2, eps=1e-6):
        mu1, mu2 = [t.detach().cpu().double().numpy() if torch.is_tensor(t) else np.asarray(t, np.float64)
                    for t in (mu1, mu2)]
        s1, s2 = [t.detach().cpu().double().numpy() if torch.is_tensor(t) else np.asarray(t, np.float64)
                  for t in (sigma1, sigma2)]
        w, v = np.linalg.eigh((s1 + s1.T) / 2)
        r1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
        mm = r1 @ s2 @ r1
        ev = np.linalg.eigvalsh((mm + mm.T) / 2)
        tr_covmean = np.sqrt(np.clip(ev, 0, None)).sum()
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2.0 * tr_covmean)

    m.fid_score = fid_score_fast
    return True


@torch.no_grad()
def compute_metrics(G, real_u8, n, z_dim, dev, seed, norm="imagenet", with_real_is=False, bs=250):
    """FID (blog + pool) of n generated vs n real images, IS of the generated (and optionally real) images."""
    from ignite.metrics import InceptionScore
    patched = patch_ignite_fid()
    fid_blog, fid_pool = new_fid_metrics(dev)
    isc = InceptionScore(device=dev)
    isc_real = InceptionScore(device=dev) if with_real_is else None
    g = torch.Generator().manual_seed(seed)
    ridx = torch.randperm(real_u8.shape[0], generator=g)[:n].to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    for i in range(0, len(ridx), bs):
        idx = ridx[i:i + bs]
        real = real_u8[idx].permute(0, 3, 1, 2).float() / 127.5 - 1.0
        fake = G(torch.randn(len(idx), z_dim, device=dev, generator=gen))
        real, fake = prep(real, norm), prep(fake, norm)
        fid_blog.update((fake, real))
        fid_pool.update((fake, real))
        isc.update(fake)
        if isc_real is not None:
            isc_real.update(real)
    out = dict(fid=float(fid_pool.compute()), fid_blog=float(fid_blog.compute()), is_fake=float(isc.compute()),
               fid_patched=patched)
    if isc_real is not None:
        out["is_real"] = float(isc_real.compute())
    return out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def moving_avg(x, k):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikiart128.npy", help="uint8 array (N,S,S,3), S=64/128, from preprocess_wikiart.py")
    ap.add_argument("--dummy", type=int, default=0, help="use this many random images instead of --data (smoke test)")
    ap.add_argument("--dummy_size", type=int, default=128, choices=[64, 128], help="image size for --dummy")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--g_ch", type=int, default=64, help="generator base width")
    ap.add_argument("--d_ch", type=int, default=64, help="discriminator base width")
    ap.add_argument("--loss", choices=["bce", "hinge"], default="bce")
    ap.add_argument("--smooth", type=float, default=1.0, help="real label for bce (1.0 = no smoothing)")
    ap.add_argument("--sn", action="store_true", help="spectral norm in D (replaces its BatchNorm)")
    ap.add_argument("--ema", type=float, default=0.999, help="generator EMA decay (0 = off); EMA weights are evaluated")
    ap.add_argument("--no_flip", action="store_true", help="disable random horizontal flips")
    ap.add_argument("--n_eval", type=int, default=5000, help="images for the final FID / IS")
    ap.add_argument("--eval_every", type=int, default=0, help="quick FID (2000 imgs) every k epochs (0 = off)")
    ap.add_argument("--norm", choices=["imagenet", "minus1"], default="imagenet",
                    help="Inception input scaling: ImageNet mean/std (default) or [-1,1] as in the ignite blog")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- data (whole dataset on the device as uint8)
    if a.dummy:
        arr = np.random.RandomState(a.seed).randint(0, 256, (a.dummy, a.dummy_size, a.dummy_size, 3), dtype=np.uint8)
    else:
        arr = np.load(a.data)
    assert arr.ndim == 4 and arr.shape[1] == arr.shape[2] and arr.shape[1] in (64, 128) and arr.shape[3] == 3 \
        and arr.dtype == np.uint8, arr.shape
    size = arr.shape[1]
    data = torch.from_numpy(arr).to(dev)
    N = data.shape[0]
    steps = N // a.batch_size
    print(f"device {dev}; {N} images of {size}x{size}; {steps} steps/epoch", flush=True)

    # ---- models
    G = Generator(a.z_dim, a.g_ch, size).to(dev)
    D = Discriminator(a.d_ch, a.sn, size).to(dev)
    ema_G = copy.deepcopy(G) if a.ema > 0 else None
    if ema_G is not None:
        for p in ema_G.parameters():
            p.requires_grad_(False)
    optG = torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(a.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(a.beta1, 0.999))
    print(f"G params {n_params(G) / 1e6:.2f}M, D params {n_params(D) / 1e6:.2f}M", flush=True)

    def d_loss_fn(d_real, d_fake):
        if a.loss == "hinge":
            return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        return (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, a.smooth))
                + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))

    def g_loss_fn(d_fake):
        if a.loss == "hinge":
            return -d_fake.mean()
        return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))   # non-saturating

    hist = dict(lossD_it=[], lossG_it=[], epoch_lossD=[], epoch_lossG=[], epoch_Dreal=[], epoch_Dfake=[],
                fid_epochs=[], fid_quick=[])
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(N, device=dev)
        sums = torch.zeros(4, device=dev)
        for i in range(steps):
            x = data[perm[i * a.batch_size:(i + 1) * a.batch_size]].permute(0, 3, 1, 2).float() / 127.5 - 1.0
            if not a.no_flip:
                flip = torch.rand(x.size(0), device=dev) < 0.5
                x = torch.where(flip[:, None, None, None], x.flip(3), x)
            z = torch.randn(x.size(0), a.z_dim, device=dev)
            fake = G(z)

            # discriminator step
            d_real, d_fake = D(x), D(fake.detach())
            lossD = d_loss_fn(d_real, d_fake)
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

            # generator step
            lossG = g_loss_fn(D(fake))
            optG.zero_grad(set_to_none=True)
            lossG.backward()
            optG.step()

            if ema_G is not None:
                with torch.no_grad():
                    for pe, p in zip(ema_G.parameters(), G.parameters()):
                        pe.mul_(a.ema).add_(p.detach(), alpha=1.0 - a.ema)
                    for be, b in zip(ema_G.buffers(), G.buffers()):
                        be.copy_(b)

            sums += torch.stack([lossD.detach(), lossG.detach(), d_real.detach().mean(), d_fake.detach().mean()])
            if i % 10 == 0:
                hist["lossD_it"].append(float(lossD))
                hist["lossG_it"].append(float(lossG))
        m = (sums / steps).tolist()
        hist["epoch_lossD"].append(m[0]); hist["epoch_lossG"].append(m[1])
        hist["epoch_Dreal"].append(m[2]); hist["epoch_Dfake"].append(m[3])
        msg = (f"epoch {ep:3d}/{a.epochs}  lossD {m[0]:.3f}  lossG {m[1]:.3f}  "
               f"D(real) {m[2]:+.2f}  D(fake) {m[3]:+.2f}  [{(time.time() - t0) / 60:.1f} min]")
        if a.eval_every and (ep % a.eval_every == 0 or ep == a.epochs):
            q = compute_metrics(ema_G if ema_G is not None else G, data, min(2000, N), a.z_dim, dev, a.seed + 100, a.norm)
            hist["fid_epochs"].append(ep); hist["fid_quick"].append(q["fid"])
            msg += f"  quick(2000) FIDpool {q['fid']:.1f}  FIDblog {q['fid_blog']:.4f}  IS {q['is_fake']:.2f}"
        print(msg, flush=True)
    train_min = (time.time() - t0) / 60

    # ---- final evaluation of the (EMA) generator
    Gf = ema_G if ema_G is not None else G
    res = compute_metrics(Gf, data, min(a.n_eval, N), a.z_dim, dev, a.seed + 7, a.norm, with_real_is=True)
    print(f"FINAL  FID(pool,2048) {res['fid']:.2f}  FID(blog,softmax) {res['fid_blog']:.5f}  IS(fake) {res['is_fake']:.3f}  IS(real) {res['is_real']:.3f}  "
          f"(n={min(a.n_eval, N)}, fid_patched={res['fid_patched']})", flush=True)

    # ---- outputs
    torch.save(dict(G=G.state_dict(), ema_G=(ema_G.state_dict() if ema_G is not None else None),
                    D=D.state_dict(), args=vars(a)), os.path.join(a.out_dir, "gan.pt"))
    with torch.no_grad():
        s = generate(Gf, 64, a.z_dim, dev, a.seed + 3, bs=64)
        save_image(s * 0.5 + 0.5, os.path.join(a.out_dir, "samples.png"), nrow=8, padding=2)
        gi = torch.Generator().manual_seed(a.seed + 5)
        ridx = torch.randperm(N, generator=gi)[:64].to(dev)
        save_image(data[ridx].permute(0, 3, 1, 2).float() / 255.0, os.path.join(a.out_dir, "real.png"),
                   nrow=8, padding=2)

    ncols = 3 if hist["fid_quick"] else 2
    fig, ax = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.0))
    k = 20
    for name, key in [("Discriminator", "lossD_it"), ("Generator", "lossG_it")]:
        y = moving_avg(hist[key], k)
        ax[0].plot(np.arange(len(y)) * 10 + k * 5, y, label=name, lw=1)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss"); ax[0].set_title("GAN losses (moving avg.)")
    ax[0].legend()
    ep_axis = np.arange(1, len(hist["epoch_Dreal"]) + 1)
    ax[1].plot(ep_axis, hist["epoch_Dreal"], label="D(real)", lw=1)
    ax[1].plot(ep_axis, hist["epoch_Dfake"], label="D(fake)", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean logit"); ax[1].set_title("Discriminator outputs"); ax[1].legend()
    if ncols == 3:
        ax[2].plot(hist["fid_epochs"], hist["fid_quick"], marker="o", ms=3, lw=1)
        ax[2].set_xlabel("epoch"); ax[2].set_ylabel("FID, pool features (2000 imgs)"); ax[2].set_title("FID during training")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "loss_curve.pdf"))
    plt.close(fig)

    out = dict(vars(a), img_size=int(size), n_images=int(N), steps_per_epoch=int(steps), params_G=n_params(G), params_D=n_params(D),
               train_minutes=train_min, n_eval_used=int(min(a.n_eval, N)),
               fid=res["fid"], fid_blog=res["fid_blog"], is_fake=res["is_fake"], is_real=res["is_real"], fid_patched=res["fid_patched"],
               final_lossD=hist["epoch_lossD"][-1], final_lossG=hist["epoch_lossG"][-1], history=hist)
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(out, f)
    macros = dict(hpImg=size, hpEpochs=a.epochs, hpBatch=a.batch_size, hpLRG=f"{a.lr_g:g}", hpLRD=f"{a.lr_d:g}",
                  hpBetaOne=a.beta1, hpZ=a.z_dim, hpLoss=a.loss, hpSN="yes" if a.sn else "no", hpEMA=a.ema,
                  hpParamsG=f"{n_params(G) / 1e6:.1f}M", hpParamsD=f"{n_params(D) / 1e6:.1f}M",
                  hpNeval=out["n_eval_used"], resFID=f"{res['fid']:.2f}", resFIDblog=f"{res['fid_blog']:.4f}", resIS=f"{res['is_fake']:.2f}",
                  resISreal=f"{res['is_real']:.2f}", resLossD=f"{out['final_lossD']:.3f}",
                  resLossG=f"{out['final_lossG']:.3f}")
    with open(os.path.join(a.out_dir, "results.tex"), "w") as f:
        for k_, v in macros.items():
            f.write(f"\\renewcommand{{\\{k_}}}{{{v}}}\n")
    print(f"done in {train_min:.1f} min; wrote outputs to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
