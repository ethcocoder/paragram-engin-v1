"""
train.py — Paradox Genesis: Universal Master Training Engine
=============================================================
The definitive training pipeline for building a Universal Neural Core.
Learns "visual grammar" from 100,000+ images to compress ANY random 
HD image without prior knowledge of its content.

Logic:
    - Perceptual VGG-16 Loss (Texture Matching)
    - MS-SSIM Structural Loss (Shape Matching)
    - 4-Stage 16-channel Manifold (~96x Reduction)
    - STL-10 Pattern Learning (100k Unlabeled Images)
"""

import os
import argparse
import logging
from typing import Tuple, Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

_SSIM_WINDOW_CACHE = {}
from tqdm import tqdm

from data import get_dataloaders
from model import LatentGenesisCore

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


# ── Perceptual Loss (The 'Universal Texture' Engine) ─────────────────────────

class PerceptualLoss(nn.Module):
    """
    Uses pre-trained VGG16 to compare deep 'visual concepts' rather than 
    raw pixels. This is what allows the model to learn 'patterns' 
    rather than just memorizing images.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4])   # Edge concepts
        self.slice2 = nn.Sequential(*vgg[4:9])  # Texture concepts
        self.slice3 = nn.Sequential(*vgg[9:16]) # Structural concepts
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        y = (y * 0.5 + 0.5 - self.mean) / self.std
        
        x_f1, y_f1 = self.slice1(x), self.slice1(y)
        x_f2, y_f2 = self.slice2(x_f1), self.slice2(y_f1)
        x_f3, y_f3 = self.slice3(x_f2), self.slice3(y_f2)
        
        return F.mse_loss(x_f1, y_f1) + F.mse_loss(x_f2, y_f2) + F.mse_loss(x_f3, y_f3)


# ── SSIM Loss ────────────────────────────────────────────────────────────────

def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)


def ssim_loss(x, y, window_size: int = 11):
    C_ch = x.shape[1]
    cache_key = (x.device.type, x.device.index, str(x.dtype), C_ch, window_size)
    window = _SSIM_WINDOW_CACHE.get(cache_key)
    if window is None:
        window = _gaussian_window(window_size).to(device=x.device, dtype=x.dtype)
        window = window.expand(C_ch, 1, window_size, window_size).contiguous()
        _SSIM_WINDOW_CACHE[cache_key] = window
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=C_ch)
    mu_y = F.conv2d(y, window, padding=pad, groups=C_ch)
    sig_xx = F.conv2d(x * x, window, padding=pad, groups=C_ch) - mu_x**2
    sig_yy = F.conv2d(y * y, window, padding=pad, groups=C_ch) - mu_y**2
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=C_ch) - mu_x*mu_y

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sig_xx + sig_yy + C2))
    return 1.0 - ssim_map.mean()


# ── Master Compression Loss ──────────────────────────────────────────────────

def compression_loss(recon_x, x, mu, logvar, kld_weight=0.001, perc_model=None):
    l1_l   = F.l1_loss(recon_x, x)
    ssim_l = ssim_loss(recon_x, x)
    
    # --- Paradox Safety Protocol 1.1 ---
    # Clamp logvar to prevent KLD explosion (e^10 is ~22k, enough for variance)
    logvar_c = torch.clamp(logvar, -10, 10)
    kld_l  = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
    
    perc_l = torch.tensor(0.0, device=x.device)
    if perc_model is not None:
        perc_l = perc_model(recon_x, x)

    # Balanced weight for universal Generalization
    total = l1_l + (0.5 * ssim_l) + (0.1 * perc_l) + (kld_weight * kld_l)
    return total, l1_l, ssim_l, perc_l, kld_l


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[*] Paradox Universal Master: Initiating 'World Pattern' Learning on %s", device)

    torch.backends.cudnn.benchmark = True

    # 1. Universal Data Loader (Sampled for Speed)
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size, root="./data", num_workers=2,
        pin_memory=(device.type == "cuda"), use_hd=args.use_hd,
        sample_limit=args.sample_limit
    )

    # 2. Elite 16-channel Core
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    perc_model = PerceptualLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Universal KLD ramp
        kld_weight = min(1.0, epoch / max(1, args.epochs // 2)) * 0.001
        
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            outputs, mu, logvar = model(images)
            loss, l1_l, ssim_l, perc_l, kld_l = compression_loss(
                outputs, images, mu, logvar, kld_weight, perc_model
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{ssim_l.item():.4f}", perc=f"{perc_l.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        log.info(f"[*] Epoch {epoch+1} Completed. Master Loss: {epoch_loss:.4f}")

        # Save Universal Core
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_channels': args.latent_channels,
            }, os.path.join(args.checkpoint_dir, 'universal_genesis_core.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox Master Training")
    parser.add_argument("--batch_size", type=int, default=16) # Optimized for T4 VRAM
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4) # Lowered for ELITE stability
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--sample_limit", type=int, default=5000) # Speed/GENERALIZATION balance
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--use_hd", type=bool, default=True) # Always HD-Ready patterns
    args = parser.parse_args()
    train(args)
