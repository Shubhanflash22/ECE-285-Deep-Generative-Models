"""
Hyperparameter sweep for train_gan.py. Finished runs are skipped, so you can stop and re-run any time.

    nohup python sweep_gan.py --round 1 --data wikiart128.npy --par 2 > sweep_log1.txt 2>&1 &
    tail -f sweep_log1.txt
    python sweep_gan.py --summarize_only        # rank whatever has finished

Runs save a checkpoint every 10 epochs and resume from it, so after a server restart you just re-run the same
command. Each run's gan.pt is deleted after it finishes (DataHub disk quota); samples, curves and results.json are kept.
Ranking is by FID on 2048-d pooled features (lower is better); the blog FID and IS are shown too.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# name -> (extra arguments for train_gan.py, epochs)
ROUND1 = {   # 60 epochs each: compare the training recipe
    "bce_bn":          ([], 60),                                                       # plain DCGAN
    "hinge_sn":        (["--loss", "hinge", "--sn"], 60),
    "hinge_sn_ttur":   (["--loss", "hinge", "--sn", "--lr_g", "1e-4", "--lr_d", "4e-4", "--beta1", "0.0"], 60),
    "bce_sn":          (["--loss", "bce", "--sn"], 60),
    "hinge_sn_wide":   (["--loss", "hinge", "--sn", "--g_ch", "96", "--d_ch", "96"], 60),
    "hinge_sn_bs64":   (["--loss", "hinge", "--sn", "--batch_size", "64"], 60),
}
ROUND2 = {   # longer training with linear learning-rate decay (second half of training)
    "bce_bn_e150_decay":    (["--decay", "linear"], 150),
    "hinge_sn_e150_decay":  (["--loss", "hinge", "--sn", "--decay", "linear"], 150),
}
ROUND3 = {   # 60 epochs, DiffAugment on real and fake images before D (the discriminator was overfitting)
    "bce_bn_da":    (["--diffaug"], 60),
    "hinge_sn_da":  (["--loss", "hinge", "--sn", "--diffaug"], 60),
    "bce_sn_da":    (["--loss", "bce", "--sn", "--diffaug"], 60),
}
ROUND4 = {   # 60 epochs: DiffAugment + batch 64 together, and a second seed of the best run (repeatability)
    "hinge_sn_da_bs64": (["--loss", "hinge", "--sn", "--diffaug", "--batch_size", "64"], 60),
    "hinge_sn_da_s1":   (["--loss", "hinge", "--sn", "--diffaug", "--seed", "1"], 60),
}
_BEST = ["--loss", "hinge", "--sn", "--diffaug", "--batch_size", "64", "--decay", "linear"]
ROUND5 = {   # the chosen recipe (hinge + spectral norm + DiffAugment, batch 64) trained longer, with linear LR decay
    "hinge_sn_da_bs64_e150": (_BEST, 150),
    "hinge_sn_da_bs64_e600": (_BEST, 600),
}
ROUNDS = {1: ROUND1, 2: ROUND2, 3: ROUND3, 4: ROUND4, 5: ROUND5}


def run_one(job, a):
    name, extra, epochs = job
    out = os.path.join(a.root, name)
    if os.path.exists(os.path.join(out, "results.json")):
        print(f"[skip]  {name}", flush=True)
        return
    cmd = [sys.executable, "train_gan.py", "--data", a.data, "--epochs", str(epochs), "--n_eval", str(a.n_eval),
           "--eval_every", str(a.eval_every), "--out_dir", out, "--ckpt_every", "10", "--resume"] + extra
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    t0 = time.time()
    print(f"[start] {name}  ({' '.join(extra)})", flush=True)
    rc = 1
    for attempt in (1, 2):
        with open(os.path.join(a.root, name + ".log"), "w" if attempt == 1 else "a") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        if rc == 0 and os.path.exists(os.path.join(out, "results.json")):
            break
        print(f"[retry] {name} failed (exit={rc}), attempt {attempt}", flush=True)
    ck = os.path.join(out, "gan.pt")
    if os.path.exists(ck) and not a.keep_ckpt:
        os.remove(ck)
    print(f"[done]  {name} exit={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)


def summarize():
    rows = []
    for p in glob.glob("sweep_gan*/*/results.json"):
        r = json.load(open(p))
        rows.append(dict(name=os.path.basename(os.path.dirname(p)), root=os.path.dirname(os.path.dirname(p)),
                         dir=os.path.dirname(p), epochs=r["epochs"], loss=r["loss"], sn=r["sn"], lr_g=r["lr_g"],
                         lr_d=r["lr_d"], beta1=r["beta1"], batch=r["batch_size"], g_ch=r["g_ch"], d_ch=r["d_ch"],
                         ema=r["ema"], decay=r.get("decay", "none"), diffaug=r.get("diffaug", False), fid=r["fid"], fid_blog=r["fid_blog"], is_fake=r["is_fake"],
                         is_real=r["is_real"], minutes=r["train_minutes"]))
    if not rows:
        print("No finished runs yet.")
        return
    rows.sort(key=lambda d: d["fid"])
    print(f"\n{'run':<22}{'ep':>4}{'FID(pool)':>11}{'FID(blog)':>11}{'IS':>7}{'min':>6}")
    for i, d in enumerate(rows):
        print(f"{d['name']:<22}{d['epochs']:>4}{d['fid']:>11.2f}{d['fid_blog']:>11.4f}{d['is_fake']:>7.2f}"
              f"{d['minutes']:>6.0f}{'  <-- best' if i == 0 else ''}")
    with open("sweep_gan_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = rows[0]
    os.makedirs("outputs_best", exist_ok=True)
    for fn in ["loss_curve.pdf", "samples.png", "real.png", "results.tex", "results.json"]:
        src = os.path.join(best["dir"], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join("outputs_best", fn))
    print(f"\nBest: {best['name']} (FID pool {best['fid']:.2f}). Copied to outputs_best/; wrote sweep_gan_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikiart128.npy")
    ap.add_argument("--round", type=int, default=1, choices=sorted(ROUNDS))
    ap.add_argument("--par", type=int, default=2, help="runs in parallel (training is GPU-bound)")
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    a = ap.parse_args()
    if not a.summarize_only:
        a.root = f"sweep_gan{a.round}"
        os.makedirs(a.root, exist_ok=True)
        cfgs = ROUNDS[a.round]
        jobs = [(n, e, ep) for n, (e, ep) in cfgs.items()]
        print(f"round {a.round}: {len(jobs)} runs, {a.par} in parallel", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.par) as ex:
            list(ex.map(lambda j: run_one(j, a), jobs))
        print(f"Sweep finished in {(time.time() - t0) / 60:.1f} min", flush=True)
        miss = [n for n, _, _ in jobs if not os.path.exists(os.path.join(a.root, n, "results.json"))]
        if miss:
            print("NO RESULT for:", miss, flush=True)
    summarize()


if __name__ == "__main__":
    main()