"""
test_stages.py — Paradox Stage Evaluation Engine
=================================================
Tests each training stage independently and generates a visual report card.
Each stage has clear PASS/FAIL criteria so you know training is working
before investing time in the next stage.

Usage:
  # Test individual stages
  python src/test_stages.py --stage 1 --model_path checkpoints/universal_genesis_core.pth
  python src/test_stages.py --stage 2 --model_path checkpoints/stage2_genesis_core.pth
  python src/test_stages.py --stage 3 --model_path checkpoints/stage3_rl_genesis_core.pth

  # Full side-by-side comparison of all 3 stages on the same images
  python src/test_stages.py --compare \\
    --s1 checkpoints/universal_genesis_core.pth \\
    --s2 checkpoints/stage2_genesis_core.pth \\
    --s3 checkpoints/stage3_rl_genesis_core.pth
"""

import os
import sys
import argparse
import random
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms as T

# Resolve src directory
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model import LatentGenesisCore
from train import ssim_loss, QuantumWaveletLoss


# ── PASS/FAIL Thresholds per Stage ──────────────────────────────────────────

CRITERIA = {
    1: {"psnr": 14.0,  "ssim": 0.35, "label": "Stage 1 — Monster VAE-GAN"},
    2: {"psnr": 22.0,  "ssim": 0.65, "label": "Stage 2 — Fine-Tune"},
    3: {"psnr": 26.0,  "ssim": 0.78, "label": "Stage 3 — RL from Mistakes"},
}

# Hard test images for Stage 3 (high-frequency patterns)
HARD_SEEDS = [42, 137, 256, 512, 999]   # deterministic "hard" picsum seeds


# ── Utilities ────────────────────────────────────────────────────────────────

def load_model(model_path: str, latent_channels: int, device: torch.device) -> LatentGenesisCore:
    model = LatentGenesisCore(latent_channels=latent_channels, device=str(device)).to(device)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        stage = ckpt.get('stage', '?')
        epoch = ckpt.get('epoch', '?')
        print(f"   ✔ Loaded: {model_path}  (stage={stage}, epoch={epoch})")
    else:
        print(f"   ✘ Checkpoint NOT found: {model_path}  — using random weights")
    model.eval()
    return model


def fetch_image(seed: int, size: int = 256) -> Image.Image:
    url  = f"https://picsum.photos/seed/{seed}/{size}/{size}"
    path = f"_test_img_{seed}_{size}.jpg"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return Image.open(path).convert('RGB')


def preprocess(img: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return tf(img).unsqueeze(0).to(device)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) normalised tensor → uint8 HWC numpy."""
    img = t[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    return (img.clamp(0, 1).numpy() * 255).astype(np.uint8)


def compute_metrics(model: LatentGenesisCore,
                    images_t: torch.Tensor,
                    qwm: QuantumWaveletLoss,
                    device: torch.device) -> Dict:
    """Run inference and return metrics dict."""
    with torch.no_grad():
        recon, mu, logvar = model(images_t)

    mse      = F.mse_loss(recon, images_t).item()
    psnr     = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
    ssim_val = 1.0 - ssim_loss(recon, images_t).item()
    qwm_val  = qwm(recon, images_t).item()

    return {"psnr": psnr, "ssim": ssim_val, "mse": mse, "qwm": qwm_val,
            "recon": recon, "mu": mu, "logvar": logvar}


def print_report_card(stage: int, metrics: Dict, passed: bool):
    c = CRITERIA[stage]
    bar = "█" * 40
    print(f"\n{'='*60}")
    print(f"  📋 {c['label']} — REPORT CARD")
    print(f"{'='*60}")
    print(f"  PSNR  : {metrics['psnr']:6.2f} dB   (need ≥ {c['psnr']:.1f} dB)   {'✅' if metrics['psnr'] >= c['psnr'] else '❌'}")
    print(f"  SSIM  : {metrics['ssim']:6.4f}      (need ≥ {c['ssim']:.2f})        {'✅' if metrics['ssim'] >= c['ssim'] else '❌'}")
    print(f"  MSE   : {metrics['mse']:6.6f}")
    print(f"  QWM↓  : {metrics['qwm']:6.4f}")
    verdict = "✅  PASSED — proceed to next stage" if passed else "❌  FAILED — train more epochs before proceeding"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════════════════
# SINGLE-STAGE TEST
# ════════════════════════════════════════════════════════════════════════════

def test_stage(stage: int, model_path: str, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label  = CRITERIA[stage]['label']
    print(f"\n🔍 Testing {label}")
    print(f"   Device : {device}")

    model = load_model(model_path, args.latent_channels, device)
    qwm   = QuantumWaveletLoss(channels=3).to(device)

    # ── Choose test images ───────────────────────────────────────────────────
    # Stage 3 uses intentionally hard high-freq seeds; stages 1-2 use random
    seeds  = HARD_SEEDS if stage == 3 else [random.randint(0, 9999) for _ in range(5)]
    size   = 256

    originals, recons = [], []
    all_psnr, all_ssim, all_qwm = [], [], []

    for seed in seeds:
        img_pil = fetch_image(seed, size)
        inp     = preprocess(img_pil, size, device)
        m       = compute_metrics(model, inp, qwm, device)

        all_psnr.append(m['psnr'])
        all_ssim.append(m['ssim'])
        all_qwm.append(m['qwm'])
        originals.append(tensor_to_np(inp))
        recons.append(tensor_to_np(m['recon']))

    avg = {"psnr": np.mean(all_psnr), "ssim": np.mean(all_ssim),
           "mse":  0.0,               "qwm":  np.mean(all_qwm)}

    passed = (avg['psnr'] >= CRITERIA[stage]['psnr'] and
              avg['ssim'] >= CRITERIA[stage]['ssim'])
    print_report_card(stage, avg, passed)

    # ── Visual Report Card ───────────────────────────────────────────────────
    n    = len(seeds)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.patch.set_facecolor('#0d0d0d')

    status_color = '#00ff88' if passed else '#ff4444'
    verdict_txt  = "✅ PASSED" if passed else "❌ FAILED"

    fig.suptitle(
        f"{label}\n"
        f"PSNR: {avg['psnr']:.2f} dB  |  SSIM: {avg['ssim']:.4f}  |  QWM↓: {avg['qwm']:.4f}  |  {verdict_txt}",
        fontsize=14, color=status_color, fontweight='bold', y=0.98
    )

    for i in range(n):
        # Row 0: Originals
        axes[0, i].imshow(originals[i])
        axes[0, i].set_title(f"Original #{i+1}", color='#aaaaaa', fontsize=9)
        axes[0, i].axis('off')

        # Row 1: Reconstructions
        axes[1, i].imshow(recons[i])
        axes[1, i].set_title(
            f"Recon\nPSNR: {all_psnr[i]:.1f}dB | SSIM: {all_ssim[i]:.3f}",
            color='#00ccff', fontsize=8
        )
        axes[1, i].axis('off')

        # Row 2: Difference map
        diff = np.abs(originals[i].astype(float) - recons[i].astype(float))
        diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
        axes[2, i].imshow(diff_norm, cmap='hot')
        axes[2, i].set_title("Error Map", color='#ff8844', fontsize=9)
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"stage{stage}_report_card.png"
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"   💾 Report saved → {out}")

    if not passed:
        needed = CRITERIA[stage]
        print(f"\n   💡 Tips to fix:")
        if avg['psnr'] < needed['psnr']:
            print(f"      → PSNR too low ({avg['psnr']:.2f} < {needed['psnr']}): train more epochs or increase batch size")
        if avg['ssim'] < needed['ssim']:
            print(f"      → SSIM too low ({avg['ssim']:.4f} < {needed['ssim']}): increase SSIM loss weight")

    return passed


# ════════════════════════════════════════════════════════════════════════════
# CROSS-STAGE COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def test_compare(args):
    """Run all 3 models on the same 4 images and produce a single comparison dashboard."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🏆 Full Stage Comparison | Device: {device}")

    paths = {1: args.s1, 2: args.s2, 3: args.s3}
    models, loaded = {}, {}
    for s, p in paths.items():
        if p and os.path.exists(p):
            models[s] = load_model(p, args.latent_channels, device)
            loaded[s] = True
        else:
            print(f"   ⚠️  Stage {s} checkpoint not found ({p}) — skipping")
            loaded[s] = False

    if not any(loaded.values()):
        print("   ✘ No checkpoints found. Nothing to compare.")
        return

    qwm   = QuantumWaveletLoss(channels=3).to(device)
    seeds = [7, 42, 137, 512]    # fixed seeds for a fair comparison
    size  = 256

    stage_labels  = {1: "S1: Monster", 2: "S2: Fine-Tune", 3: "S3: RL"}
    stage_colors  = {1: '#ff6644',     2: '#44aaff',        3: '#44ff88'}
    active_stages = [s for s in [1, 2, 3] if loaded[s]]
    n_cols        = 1 + len(active_stages)    # Original + one col per stage
    n_rows        = len(seeds)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor('#0d0d0d')
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    summary_psnr = {s: [] for s in active_stages}
    summary_ssim = {s: [] for s in active_stages}

    for row, seed in enumerate(seeds):
        img_pil = fetch_image(seed, size)
        inp     = preprocess(img_pil, size, device)

        # Original
        axes[row, 0].imshow(tensor_to_np(inp))
        axes[row, 0].set_title("Original", color='#ffffff', fontsize=9)
        axes[row, 0].axis('off')

        # Each stage
        for col, s in enumerate(active_stages, start=1):
            m = compute_metrics(models[s], inp, qwm, device)
            summary_psnr[s].append(m['psnr'])
            summary_ssim[s].append(m['ssim'])

            axes[row, col].imshow(tensor_to_np(m['recon']))
            axes[row, col].set_title(
                f"{stage_labels[s]}\n{m['psnr']:.1f}dB | SSIM:{m['ssim']:.3f}",
                color=stage_colors[s], fontsize=8
            )
            axes[row, col].axis('off')

    # ── PSNR Summary bar chart ───────────────────────────────────────────────
    fig.subplots_adjust(bottom=0.18)
    ax_bar = fig.add_axes([0.1, 0.02, 0.8, 0.13])
    ax_bar.set_facecolor('#1a1a1a')

    bar_w = 0.25
    xs    = np.arange(len(seeds))
    for i, s in enumerate(active_stages):
        offset = (i - len(active_stages) / 2 + 0.5) * bar_w
        ax_bar.bar(xs + offset, summary_psnr[s], bar_w,
                   label=f"{stage_labels[s]} (avg {np.mean(summary_psnr[s]):.1f}dB)",
                   color=stage_colors[s], alpha=0.85)

    for thresh_s, thresh_v in [(1, 14), (2, 22), (3, 26)]:
        if thresh_s in active_stages:
            ax_bar.axhline(thresh_v, color=stage_colors[thresh_s],
                           linestyle='--', linewidth=0.7, alpha=0.6)

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([f"Img {i+1}" for i in range(len(seeds))], color='#aaaaaa', fontsize=8)
    ax_bar.set_ylabel("PSNR (dB)", color='#aaaaaa', fontsize=8)
    ax_bar.tick_params(colors='#aaaaaa')
    ax_bar.spines[:].set_color('#333333')
    ax_bar.legend(fontsize=7, loc='upper right', facecolor='#1a1a1a', labelcolor='white')

    fig.suptitle("🏆 Paradox — Stage Progression Comparison", fontsize=14,
                 color='#ffffff', fontweight='bold', y=1.0)

    out = "stage_comparison_dashboard.png"
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n   💾 Comparison dashboard saved → {out}")

    # Print summary table
    print(f"\n{'='*55}")
    print(f"  {'Stage':<20} {'Avg PSNR':>10}   {'Avg SSIM':>10}")
    print(f"  {'-'*50}")
    for s in active_stages:
        verdict = "✅" if np.mean(summary_psnr[s]) >= CRITERIA[s]['psnr'] else "❌"
        print(f"  {stage_labels[s]:<20} {np.mean(summary_psnr[s]):>8.2f} dB   {np.mean(summary_ssim[s]):>10.4f}  {verdict}")
    print(f"{'='*55}\n")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox Stage Evaluation")

    # Single-stage mode
    parser.add_argument("--stage",           type=int,  default=None,
                        help="Stage to test: 1, 2, or 3")
    parser.add_argument("--model_path",      type=str,  default=None,
                        help="Checkpoint path for single-stage test")

    # Compare mode
    parser.add_argument("--compare",         action="store_true",
                        help="Compare all stages side-by-side")
    parser.add_argument("--s1",              type=str,  default="checkpoints/universal_genesis_core.pth")
    parser.add_argument("--s2",              type=str,  default="checkpoints/stage2_genesis_core.pth")
    parser.add_argument("--s3",              type=str,  default="checkpoints/stage3_rl_genesis_core.pth")

    parser.add_argument("--latent_channels", type=int,  default=16)
    args = parser.parse_args()

    if args.compare:
        test_compare(args)
    elif args.stage is not None:
        if args.model_path is None:
            defaults = {
                1: "checkpoints/universal_genesis_core.pth",
                2: "checkpoints/stage2_genesis_core.pth",
                3: "checkpoints/stage3_rl_genesis_core.pth",
            }
            args.model_path = defaults.get(args.stage, "")
        test_stage(args.stage, args.model_path, args)
    else:
        parser.print_help()
