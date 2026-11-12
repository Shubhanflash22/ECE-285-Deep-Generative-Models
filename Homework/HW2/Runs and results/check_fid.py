"""
Sanity check of the FID code path used in train_gan.py (needs a GPU or just runs slower on CPU).
Compares FID for two clearly different image sets (must be large) and for two sets of the same kind (must be small),
with the input scaling used in the ignite blog ([-1,1]) and with ImageNet normalization; both the pool FID
(2048-d, standard) and the blog FID (ignite default, softmax) are printed.
    python check_fid.py
"""
import time

import torch
import torch.nn.functional as F

import train_gan as T

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
n = 600

def smooth(k):   # smooth colour blobs: clearly different statistics from white noise
    return F.interpolate(torch.rand(k, 3, 4, 4) * 2 - 1, size=128, mode="bicubic", align_corners=False).clamp(-1, 1)


A = torch.rand(n, 3, 128, 128) * 2 - 1        # white noise
A2 = torch.rand(n, 3, 128, 128) * 2 - 1       # white noise, other draw
B = smooth(n)                                 # smooth blobs
B2 = smooth(n)


def fids_of(x, y, norm):
    fb, fp = T.new_fid_metrics(dev)
    for i in range(0, n, 200):
        xa, ya = T.prep(x[i:i + 200].to(dev), norm), T.prep(y[i:i + 200].to(dev), norm)
        fb.update((xa, ya))
        fp.update((xa, ya))
    return float(fp.compute()), float(fb.compute())


T.patch_ignite_fid()
for norm in ["minus1", "imagenet"]:
    t = time.time()
    print(f"--- input scaling = {norm}", flush=True)
    for name, (x, y) in [("noise vs blobs (different)", (A, B)), ("blobs vs blobs (same)", (B, B2)),
                         ("noise vs noise (same)", (A, A2))]:
        fp, fb = fids_of(x, y, norm)
        print(f"  {name:28s} FID pool = {fp:9.2f}   FID blog = {fb:9.5f}", flush=True)
    print(f"  [{time.time() - t:.0f} s]", flush=True)