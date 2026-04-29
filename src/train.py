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
    window = _gaussian_window(window_size).to(x.device).expand(C_ch, 1, window_size, window_size).contiguous()
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


from model import LatentGenesisCore, EliteDiscriminator

# ── Master Compression Loss ──────────────────────────────────────────────────

def compression_loss(recon_x, x, mu, logvar, kld_weight=0.001, perc_model=None):
    l1_l   = F.l1_loss(recon_x, x)
    ssim_l = ssim_loss(recon_x, x)
    
    logvar_c = torch.clamp(logvar, -10, 10)
    kld_l  = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
    
    perc_l = torch.tensor(0.0, device=x.device)
    if perc_model is not None:
        perc_l = perc_model(recon_x, x)

    # Elite Balance: L1 (1.0) + SSIM (0.5) + PERC (0.1) + KLD
    total = l1_l + (0.5 * ssim_l) + (0.1 * perc_l) + (kld_weight * kld_l)
    return total, l1_l, ssim_l, perc_l, kld_l


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[*] Paradox MONSTER Engine: Initiating Adversarial Phase on %s", device)

    torch.backends.cudnn.benchmark = True

    # 1. Universal Data Loader
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size, root="./data", num_workers=2,
        pin_memory=(device.type == "cuda"), use_hd=args.use_hd,
        sample_limit=args.sample_limit
    )

    # 2. VAE-GAN Core
    model = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    discriminator = EliteDiscriminator().to(device)
    perc_model = PerceptualLoss().to(device)

    # 3. Dual Optimizers (Generator & Discriminator)
    opt_g = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        kld_weight = min(1.0, epoch / max(1, args.epochs // 2)) * 0.001
        
        model.train()
        discriminator.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            
            # --- 🚀 Phase 1: Train Discriminator ---
            opt_d.zero_grad(set_to_none=True)
            
            # Real images
            d_real = discriminator(images)
            loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            
            # Fake images
            recon, mu, logvar = model(images)
            d_fake = discriminator(recon.detach())
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # --- 🎨 Phase 2: Train Generator (VAE-GAN) ---
            opt_g.zero_grad(set_to_none=True)
            
            # Reconstruction Loss
            loss_rec, l1_l, ssim_l, perc_l, kld_l = compression_loss(
                recon, images, mu, logvar, kld_weight, perc_model
            )
            
            # Adversarial Loss (Target: Fool the Discriminator)
            d_fake_g = discriminator(recon)
            loss_adv = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))
            
            # Total MONSTER Loss (G)
            # GAN Weight is 0.05 to preserve signal integrity while adding texture
            loss_g = loss_rec + (0.05 * loss_adv)
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_g.step()
            
            pbar.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}", ssim=f"{ssim_l.item():.4f}")

        scheduler_g.step()
        
        # Save Elite Universal Core
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'disc_state_dict': discriminator.state_dict(),
                'latent_channels': args.latent_channels,
            }, os.path.join(args.checkpoint_dir, 'universal_genesis_core.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox Monster Training")
    parser.add_argument("--batch_size", type=int, default=12) # Reduced for GAN VRAM
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--sample_limit", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--use_hd", type=bool, default=True)
    args = parser.parse_args()
    train(args)
