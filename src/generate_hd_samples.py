"""
generate_hd_samples.py — Paradox v3 Sovereign Sample Generator
===============================================================
Generates its own HD training data using THREE autonomous strategies:

  MODE 1  [download]  — Pulls real HD photos from Picsum Photos (free, no key)
                        Supports thousands of unique seeds → unlimited variety.

  MODE 2  [synth]     — Uses Team A's decoder to synthesise images by sampling
                        random latent vectors from the QVS probability manifold.
                        Fully offline, no internet required.

  MODE 3  [both]      — Runs download + synth together for maximum diversity.

Usage:
  python generate_hd_samples.py --mode both --n 200 --out ../hd_images
  python generate_hd_samples.py --mode download --n 500 --res 512
  python generate_hd_samples.py --mode synth --n 300 \\
      --sender_path ../checkpoints/universal_genesis_core.pth
"""

import os
import sys
import math
import random
import argparse
import urllib.request
import urllib.error
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Pathing ──────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import LatentGenesisCore

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PICSUM_SOURCES = [
    "https://picsum.photos/seed/{seed}/{w}/{h}",
    "https://picsum.photos/{w}/{h}?random={seed}",
]

AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1),
    T.RandomResizedCrop(256, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
    T.RandomGrayscale(p=0.03),
])

_lock = threading.Lock()
_counter = {"ok": 0, "fail": 0}


def _bar(current: int, total: int, prefix: str = "", width: int = 40):
    filled = int(width * current / max(1, total))
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / max(1, total)
    print(f"\r{prefix} |{bar}| {pct:5.1f}%  ({current}/{total})", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Download real HD photos
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(idx: int, seed: int, out_dir: str, res: int,
                  augments_per_image: int) -> bool:
    """Download one image (with retry) and save augmented variants."""
    url = PICSUM_SOURCES[idx % len(PICSUM_SOURCES)].format(
        seed=seed, w=res, h=res
    )
    headers = {"User-Agent": "Mozilla/5.0 (ParadoxV3-Sampler/1.0)"}
    
    for attempt in range(3):
        try:
            req  = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
            
            # Save original
            base_path = os.path.join(out_dir, f"real_{seed:06d}.jpg")
            img.save(base_path, quality=95)
            
            # Save augmented variants
            for k in range(augments_per_image):
                aug = AUGMENT(img)
                aug.save(os.path.join(out_dir, f"real_{seed:06d}_aug{k}.jpg"),
                         quality=92)
            
            with _lock:
                _counter["ok"] += 1
            return True

        except Exception as e:
            if attempt == 2:
                with _lock:
                    _counter["fail"] += 1
                return False
            time.sleep(0.5 * (attempt + 1))
    return False


def download_hd_photos(n: int, out_dir: str, res: int = 512,
                       augments: int = 2, workers: int = 8):
    """
    Download `n` unique HD photos at `res × res` using parallel threads.
    Each photo is saved + `augments` augmented variants → total files = n*(1+augments).
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = random.sample(range(1, 999999), n)
    total = n

    print(f"\n{'─'*60}")
    print(f"  📥  DOWNLOAD MODE — {n} real HD photos @ {res}×{res}")
    print(f"  Variants per image : {1 + augments}   (1 original + {augments} augmented)")
    print(f"  Total files target : {n * (1 + augments)}")
    print(f"  Output             : {out_dir}")
    print(f"{'─'*60}")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, i, seed, out_dir, res, augments): seed
            for i, seed in enumerate(seeds)
        }
        for fut in as_completed(futures):
            done += 1
            _bar(done, total, prefix="  Downloading")

    print()  # newline after bar
    print(f"\n  [✓] Downloaded : {_counter['ok']}   Failed : {_counter['fail']}")
    _counter["ok"] = _counter["fail"] = 0


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Synthesise samples from Team A latent space
# ─────────────────────────────────────────────────────────────────────────────

def _sovereign_latent_sample(qvs_model, latent_ch: int, device) -> torch.Tensor:
    """
    Samples a latent vector by running a QVS trajectory.
    The QVS phase-bias modulates the sign of each latent channel,
    creating diverse, non-Gaussian manifold samples.
    """
    import numpy as np
    # Spatial size of latent map for 256-px input = 256/16 = 16
    H = W = 16
    mu     = torch.randn(1, latent_ch, H, W, device=device) * 0.6
    logvar = torch.full((1, latent_ch, H, W), -1.0, device=device)
    std    = torch.exp(0.5 * logvar)

    # Per-channel QVS phase bias
    biases = []
    for c in range(latent_ch):
        asc_id   = qvs_model.create_asc(size=2)
        qvs_model.SUPERPOSE(asc_id, [(0,0),(0,1),(1,0),(1,1)])
        intensity = mu[0, c].mean().item()
        qvs_model.WEAVE(asc_id, phase_angle=intensity * math.pi)
        outcome  = qvs_model.COLLAPSE(asc_id)
        biases.append(1.0 if sum(outcome) % 2 == 0 else -1.0)
        qvs_model.delete_asc(asc_id)

    bias_t = torch.tensor(biases, device=device).view(1, latent_ch, 1, 1)
    eps    = torch.randn_like(std) * bias_t
    return mu + eps * std


def synthesise_samples(n: int, out_dir: str,
                       sender_path: str, latent_channels: int = 16,
                       augments: int = 1):
    """
    Uses Team A's decoder to synthesise `n` HD images from random latent vectors.
    The QVS engine seeds the latent distribution for maximal diversity.
    No internet required.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'─'*60}")
    print(f"  🧠  SYNTH MODE — {n} images from Team A latent space")
    print(f"  Checkpoint : {sender_path}")
    print(f"  Device     : {device}")
    print(f"{'─'*60}")

    # Load Team A
    model = LatentGenesisCore(latent_channels=latent_channels).to(device)
    ckpt  = torch.load(sender_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    qvs   = model.qvs

    to_pil = T.ToPILImage()

    with torch.no_grad():
        for i in range(n):
            z = _sovereign_latent_sample(qvs, latent_channels, device)
            # Clamp + quantise (mimics the SovereignQuantizer)
            z_q = torch.clamp(z, -1.0, 1.0)
            z_q = z_q + (torch.round(z_q * 127.5) / 127.5 - z_q).detach()

            # Decode
            img_t = model.decoder(z_q)           # (1, 3, H, W) ∈ [-1,1]
            img_t = (img_t.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
            img_pil = to_pil(img_t.cpu())

            # Resize to 256 for consistency
            img_pil = img_pil.resize((256, 256), Image.LANCZOS)
            img_pil.save(os.path.join(out_dir, f"synth_{i:06d}.jpg"), quality=95)

            # Augmented variants
            for k in range(augments):
                aug = AUGMENT(img_pil)
                aug.save(os.path.join(out_dir, f"synth_{i:06d}_aug{k}.jpg"),
                         quality=92)

            _bar(i + 1, n, prefix="  Synthesising")

    print()
    print(f"\n  [✓] Synthesised {n} images  ({n*(1+augments)} files total)")


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def _report(out_dir: str):
    files = [f for f in os.listdir(out_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    total_mb = sum(
        os.path.getsize(os.path.join(out_dir, f)) for f in files
    ) / 1_048_576

    print(f"\n{'═'*60}")
    print(f"  🏆  SOVEREIGN SAMPLE GENERATION COMPLETE")
    print(f"{'═'*60}")
    print(f"  Output folder : {out_dir}")
    print(f"  Total images  : {len(files)}")
    print(f"  Disk usage    : {total_mb:.1f} MB")
    print(f"{'═'*60}")
    print(f"\n  Now run Team B training with:")
    print(f"    python hybrid_enhancer.py --mode train --data_dir {out_dir}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Paradox v3 — Sovereign HD Sample Generator"
    )

    p.add_argument("--mode", type=str, default="both",
                   choices=["download", "synth", "both"],
                   help=(
                       "download = real HD photos from internet | "
                       "synth    = AI-generated from Team A latent space | "
                       "both     = maximum diversity (recommended)"
                   ))

    # Output
    p.add_argument("--out",  type=str, default="../hd_images",
                   help="Output directory for generated samples")

    # Download settings
    p.add_argument("--n",        type=int, default=200,
                   help="Number of BASE images to generate per mode")
    p.add_argument("--res",      type=int, default=512,
                   help="Resolution for downloaded real photos (px)")
    p.add_argument("--augments", type=int, default=2,
                   help="Augmented variants saved per base image")
    p.add_argument("--workers",  type=int, default=8,
                   help="Parallel download threads")

    # Synth settings
    p.add_argument("--sender_path",    type=str,
                   default="../checkpoints/universal_genesis_core.pth",
                   help="Path to Team A (universal_genesis_core.pth)")
    p.add_argument("--latent_channels", type=int, default=16,
                   help="Must match Team A training (default 16)")

    args = p.parse_args()

    print(f"\n{'═'*60}")
    print("  🔱 Paradox v3 — Sovereign HD Sample Generator")
    print(f"  Mode: {args.mode.upper()}")
    print(f"{'═'*60}")

    if args.mode in ("download", "both"):
        download_hd_photos(
            n        = args.n,
            out_dir  = args.out,
            res      = args.res,
            augments = args.augments,
            workers  = args.workers,
        )

    if args.mode in ("synth", "both"):
        if not os.path.exists(args.sender_path):
            print(f"\n  [!] Team A checkpoint not found: {args.sender_path}")
            print("      Skipping synth mode. Run download mode only, or")
            print("      provide --sender_path to your checkpoint.")
        else:
            synthesise_samples(
                n               = args.n,
                out_dir         = args.out,
                sender_path     = args.sender_path,
                latent_channels = args.latent_channels,
                augments        = max(0, args.augments - 1),
            )

    _report(args.out)
