"""
test_stages.py — Paradox Stage Evaluation Engine (Honest Evaluation)
=====================================================================
Tests each training stage independently and generates a visual report card.
Now uses the STL-10 Test Split for domain-consistent evaluation.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Resolve src directory
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model import LatentGenesisCore
from train import ssim_loss, QuantumWaveletLoss
from data import get_dataloaders

# ── PASS/FAIL Thresholds ────────────────────────────────────────────────────
CRITERIA = {
    1: {"psnr": 14.0, "ssim": 0.35, "label": "Stage 1 — Monster VAE-GAN"},
    2: {"psnr": 22.0, "ssim": 0.65, "label": "Stage 2 — Fine-Tune"},
    3: {"psnr": 26.0, "ssim": 0.78, "label": "Stage 3 — RL from Mistakes"},
}

def load_model(model_path: str, latent_channels: int, device: torch.device) -> LatentGenesisCore:
    model = LatentGenesisCore(latent_channels=latent_channels, device=str(device)).to(device)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        s = ckpt.get('stage', '?')
        e = ckpt.get('epoch', '?')
        print(f"   ✔ Loaded: {model_path} (stage={s}, epoch={e})")
    else:
        print(f"   ✘ Checkpoint NOT found: {model_path}")
    model.eval()
    return model

def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    img = t[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    return (img.clamp(0, 1).numpy() * 255).astype(np.uint8)

def compute_metrics(model, img_t, qwm):
    with torch.no_grad():
        recon, _, _ = model(img_t)
    mse  = F.mse_loss(recon, img_t).item()
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
    ssim = 1.0 - ssim_loss(recon, img_t).item()
    qwm_val = qwm(recon, img_t).item()
    return {"psnr": psnr, "ssim": ssim, "qwm": qwm_val, "recon": recon}

def run_evaluation(stage, model_path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔍 Evaluating {CRITERIA[stage]['label']}")
    
    model = load_model(model_path, args.latent_channels, device)
    qwm   = QuantumWaveletLoss(channels=3).to(device)
    
    # Use STL-10 Test Split for domain consistency
    _, testloader = get_dataloaders(batch_size=1, stage=stage)
    
    # Deterministic selection of 5 images from test set
    indices = [10, 42, 100, 256, 512]
    originals, recons = [], []
    all_psnr, all_ssim, all_qwm = [], [], []

    test_subset = Subset(testloader.dataset, indices)
    
    for img_t, _ in test_subset:
        img_t = img_t.to(device).unsqueeze(0) if img_t.ndim == 3 else img_t.to(device)
        m = compute_metrics(model, img_t, qwm)
        
        all_psnr.append(m['psnr'])
        all_ssim.append(m['ssim'])
        all_qwm.append(m['qwm'])
        originals.append(tensor_to_np(img_t))
        recons.append(tensor_to_np(m['recon']))

    avg_psnr, avg_ssim = np.mean(all_psnr), np.mean(all_ssim)
    passed = avg_psnr >= CRITERIA[stage]['psnr'] and avg_ssim >= CRITERIA[stage]['ssim']
    
    # Print Report
    c = CRITERIA[stage]
    print(f"\n{'='*60}\n  REPORT CARD: {c['label']}\n{'='*60}")
    print(f"  PSNR: {avg_psnr:.2f} dB (Gate: {c['psnr']}) {'✅' if avg_psnr >= c['psnr'] else '❌'}")
    print(f"  SSIM: {avg_ssim:.4f}    (Gate: {c['ssim']}) {'✅' if avg_ssim >= c['ssim'] else '❌'}")
    print(f"\n  Verdict: {'✅ PASSED' if passed else '❌ FAILED'}\n{'='*60}\n")

    # Generate Visualization
    n = len(indices)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.patch.set_facecolor('#0d0d0d')
    color = '#00ff88' if passed else '#ff4444'
    fig.suptitle(f"{c['label']} | PSNR: {avg_psnr:.2f}dB | {'PASSED' if passed else 'FAILED'}", color=color, fontsize=16, fontweight='bold')

    for i in range(n):
        axes[0,i].imshow(originals[i]); axes[0,i].axis('off'); axes[0,i].set_title("Original", color='white')
        axes[1,i].imshow(recons[i]); axes[1,i].axis('off'); axes[1,i].set_title(f"Recon\n{all_psnr[i]:.1f}dB", color='#00ccff')
        diff = np.abs(originals[i].astype(float) - recons[i].astype(float))
        axes[2,i].imshow((diff / (diff.max()+1e-8) * 255).astype(np.uint8), cmap='hot'); axes[2,i].axis('off'); axes[2,i].set_title("Error Map", color='orange')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = f"stage{stage}_report_card.png"
    plt.savefig(out, dpi=120, facecolor='#0d0d0d')
    print(f"   💾 Saved: {out}")

def run_comparison(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n🏆 Full Stage Comparison (Domain Consistent)")
    
    stages = [1, 2, 3]
    paths  = {1: args.s1, 2: args.s2, 3: args.s3}
    models = {}
    for s in stages:
        if os.path.exists(paths[s]):
            models[s] = load_model(paths[s], args.latent_channels, device)
    
    if not models: return
    
    _, testloader = get_dataloaders(batch_size=1, stage=1)
    indices = [10, 42, 137, 256]
    test_subset = Subset(testloader.dataset, indices)
    
    n_rows, n_cols = len(indices), 1 + len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.patch.set_facecolor('#0d0d0d')
    
    for r, (img_t, _) in enumerate(test_subset):
        img_t = img_t.to(device).unsqueeze(0)
        axes[r,0].imshow(tensor_to_np(img_t)); axes[r,0].axis('off'); axes[r,0].set_title("Original", color='white')
        
        for c, s in enumerate(models.keys(), 1):
            with torch.no_grad():
                recon, _, _ = models[s](img_t)
            mse = F.mse_loss(recon, img_t).item()
            psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
            axes[r,c].imshow(tensor_to_np(recon)); axes[r,c].axis('off')
            axes[r,c].set_title(f"Stage {s}\n{psnr:.1f}dB", color='cyan')

    plt.tight_layout()
    plt.savefig("stage_comparison_dashboard.png", dpi=130, facecolor='#0d0d0d')
    print("   💾 Saved: stage_comparison_dashboard.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--s1", default="checkpoints/universal_genesis_core.pth")
    parser.add_argument("--s2", default="checkpoints/stage2_genesis_core.pth")
    parser.add_argument("--s3", default="checkpoints/stage3_rl_genesis_core.pth")
    parser.add_argument("--latent_channels", type=int, default=16)
    args = parser.parse_args()
    
    if args.compare: run_comparison(args)
    elif args.stage: run_evaluation(args.stage, args.model_path, args)
