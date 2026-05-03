"""
train.py — Paradox Genesis: 3-Stage Progressive Training Engine
===============================================================
Stage 1 — MONSTER:      VAE-GAN adversarial training (learns visual grammar)
Stage 2 — FINE-TUNE:    Low-LR quality polish, no GAN, QWM + SSIM focused
Stage 3 — RL MISTAKES:  Reward-weighted training — hard examples get heavier
                         gradients. Reward = PSNR improvement over baseline.

Usage:
  python src/train.py --stage 1 --epochs 100
  python src/train.py --stage 2 --epochs 30  --resume checkpoints/universal_genesis_core.pth
  python src/train.py --stage 3 --epochs 20  --resume checkpoints/stage2_genesis_core.pth
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from model import LatentGenesisCore, EliteDiscriminator, HaarWaveletTransform

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


# ── Quantum Wavelet Matching Loss ────────────────────────────────────────────

class QuantumWaveletLoss(nn.Module):
    """
    Frequency-Resonance Learning.
    Forces the generator to match high-frequency phase signatures per channel.
    """
    def __init__(self, channels=3):
        super().__init__()
        self.wavelet = HaarWaveletTransform(channels)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, recon, target):
        w_recon  = self.wavelet(recon)
        w_target = self.wavelet(target)
        B, _, H, W = w_recon.shape
        w_recon  = w_recon.view(B, 3, 4, H, W)
        w_target = w_target.view(B, 3, 4, H, W)
        ll_loss = F.l1_loss(w_recon[:, :, 0],  w_target[:, :, 0])
        hf_loss = F.l1_loss(w_recon[:, :, 1:], w_target[:, :, 1:])
        return ll_loss + (1.5 * hf_loss)


# ── SSIM Loss ────────────────────────────────────────────────────────────────

def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)


def ssim_loss(x, y, window_size: int = 11):
    C_ch   = x.shape[1]
    window = _gaussian_window(window_size).to(x.device).expand(C_ch, 1, window_size, window_size).contiguous()
    pad    = window_size // 2
    mu_x   = F.conv2d(x,   window, padding=pad, groups=C_ch)
    mu_y   = F.conv2d(y,   window, padding=pad, groups=C_ch)
    sig_xx = F.conv2d(x*x, window, padding=pad, groups=C_ch) - mu_x**2
    sig_yy = F.conv2d(y*y, window, padding=pad, groups=C_ch) - mu_y**2
    sig_xy = F.conv2d(x*y, window, padding=pad, groups=C_ch) - mu_x*mu_y
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sig_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sig_xx + sig_yy + C2))
    return 1.0 - ssim_map.mean()


# ── Master Compression Loss ──────────────────────────────────────────────────

def compression_loss(recon_x, x, mu, logvar, kld_weight=0.001, qwm_model=None):
    mse_l  = F.mse_loss(recon_x, x) * 10.0
    ssim_l = ssim_loss(recon_x, x)
    logvar_c = torch.clamp(logvar, -10, 10)
    kld_l  = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
    qwm_l  = torch.tensor(0.0, device=x.device)
    if qwm_model is not None:
        qwm_l = qwm_model(recon_x, x)
    total = mse_l + (0.7 * ssim_l) + (0.1 * qwm_l) + (kld_weight * kld_l)
    return total, mse_l, ssim_l, qwm_l, kld_l


# ── Per-sample PSNR helper ───────────────────────────────────────────────────

def batch_psnr(recon, target):
    """Returns per-image PSNR tensor of shape (B,)."""
    mse = ((recon - target) ** 2).mean(dim=(1, 2, 3))
    return 20.0 * torch.log10(1.0 / (mse.sqrt() + 1e-8))


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MONSTER VAE-GAN
# ════════════════════════════════════════════════════════════════════════════

def stage1_monster(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("🔥 STAGE 1 — MONSTER VAE-GAN | Device: %s", device)
    torch.backends.cudnn.benchmark = True

    trainloader, _ = get_dataloaders(
        batch_size=args.batch_size, root="./data", num_workers=2,
        pin_memory=(device.type == "cuda"), use_hd=args.use_hd,
        sample_limit=args.sample_limit, stage=1
    )

    model         = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    discriminator = EliteDiscriminator().to(device)
    qwm_model     = QuantumWaveletLoss(channels=3).to(device)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info("   Resumed from %s", args.resume)

    opt_g = optim.AdamW(model.parameters(),         lr=args.lr, betas=(0.5, 0.9))
    opt_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, "universal_genesis_core.pth")

    for epoch in range(args.epochs):
        kld_weight = min(1.0, epoch / max(1, args.epochs // 2)) * 0.001
        model.train(); discriminator.train()
        pbar = tqdm(trainloader, desc=f"[S1] Epoch {epoch+1}/{args.epochs}", leave=True)

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)

            # — Discriminator —
            opt_d.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            loss_d = 0.5 * (
                F.binary_cross_entropy_with_logits(discriminator(images),        torch.ones_like(discriminator(images)))  +
                F.binary_cross_entropy_with_logits(discriminator(recon.detach()), torch.zeros_like(discriminator(recon.detach())))
            )
            loss_d.backward(); opt_d.step()

            # — Generator —
            opt_g.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            loss_rec, mse_l, ssim_l, qwm_l, kld_l = compression_loss(
                recon, images, mu, logvar, kld_weight, qwm_model)
            loss_adv = F.binary_cross_entropy_with_logits(discriminator(recon), torch.ones_like(discriminator(recon)))
            loss_g = loss_rec + 0.01 * loss_adv
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_g.step()

            actual_ssim = 1.0 - ssim_l.item()
            pbar.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}", ssim=f"{actual_ssim:.4f}")

        scheduler_g.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'disc_state_dict':  discriminator.state_dict(),
                'latent_channels':  args.latent_channels,
                'stage': 1, 'epoch': epoch + 1,
            }, save_path)
            log.info("   ✅ Checkpoint saved → %s", save_path)

    log.info("🏁 Stage 1 complete. Next: python src/train.py --stage 2 --resume %s", save_path)


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FINE-TUNE
# ════════════════════════════════════════════════════════════════════════════

def stage2_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("✨ STAGE 2 — FINE-TUNE | Device: %s", device)

    # Stage 2 uses indices [50,000 → 75,000] — data the model has NEVER seen in Stage 1
    trainloader, _ = get_dataloaders(
        batch_size=args.batch_size, root="./data", num_workers=2,
        pin_memory=(device.type == "cuda"), use_hd=True,
        sample_limit=args.sample_limit, stage=2
    )

    model     = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    qwm_model = QuantumWaveletLoss(channels=3).to(device)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info("   Loaded Stage 1 weights from %s", args.resume)
    else:
        log.warning("   No --resume checkpoint given. Fine-tuning from scratch.")

    # 10× lower LR — polishing, not learning
    ft_lr = args.lr * 0.1
    optimizer = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=ft_lr * 0.1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, "stage2_genesis_core.pth")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f"[S2] Epoch {epoch+1}/{args.epochs}", leave=True)
        epoch_psnr = []

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)

            # Fine-tune: full QWM + heavier SSIM, no GAN
            mse_l  = F.mse_loss(recon, images) * 10.0
            ssim_l = ssim_loss(recon, images)
            qwm_l  = qwm_model(recon, images)
            logvar_c = torch.clamp(logvar, -10, 10)
            kld_l  = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
            loss   = mse_l + (1.0 * ssim_l) + (0.3 * qwm_l) + (0.0001 * kld_l)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            with torch.no_grad():
                psnr_batch = batch_psnr(recon, images).mean().item()
            epoch_psnr.append(psnr_batch)
            pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{1-ssim_l.item():.4f}", psnr=f"{psnr_batch:.2f}dB")

        scheduler.step()
        avg_psnr = sum(epoch_psnr) / len(epoch_psnr)
        log.info(f"   [S2 Epoch {epoch+1}] Avg PSNR: {avg_psnr:.2f} dB")

    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_channels':  args.latent_channels,
        'stage': 2, 'epoch': args.epochs,
    }, save_path)
    log.info("   ✅ Stage 2 complete → %s", save_path)
    log.info("🏁 Next: python src/train.py --stage 3 --resume %s", save_path)


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — REINFORCEMENT LEARNING FROM MISTAKES
# ════════════════════════════════════════════════════════════════════════════

def stage3_rl_mistakes(args):
    """
    RL from Mistakes:
      - Compute per-image PSNR after each forward pass.
      - Reward = PSNR - running_baseline  (positive = improved, negative = mistake)
      - Advantage = normalize rewards across batch.
      - Mistake weight = clamp(1 - advantage, 0.5, 4.0)
        → Hard images (low PSNR vs baseline) get up to 4× gradient weight.
        → Easy images (already good) get reduced weight to avoid over-smoothing.
      - Also maintains a MistakeBuffer: images where PSNR < threshold are stored
        and replayed more frequently in subsequent iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("🧠 STAGE 3 — RL FROM MISTAKES | Device: %s", device)

    # Stage 3 uses indices [75,000 → 100,000] — held-out, never seen in stages 1 or 2
    trainloader, _ = get_dataloaders(
        batch_size=args.batch_size, root="./data", num_workers=2,
        pin_memory=(device.type == "cuda"), use_hd=True,
        sample_limit=args.sample_limit, stage=3
    )

    model     = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    qwm_model = QuantumWaveletLoss(channels=3).to(device)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info("   Loaded Stage 2 weights from %s", args.resume)
    else:
        log.warning("   No --resume checkpoint given. RL training from scratch.")

    rl_lr     = args.lr * 0.05   # very small LR — surgical corrections only
    optimizer = optim.AdamW(model.parameters(), lr=rl_lr, weight_decay=1e-5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, "stage3_rl_genesis_core.pth")

    # Running baseline PSNR (exponential moving average)
    baseline_psnr = 20.0   # conservative start
    ema_alpha     = 0.05   # slow baseline drift — model must consistently beat it
    psnr_thresh   = 22.0   # below this = "mistake", eligible for replay

    # Mistake buffer (stores raw CPU tensors to avoid VRAM pressure)
    mistake_buffer = []
    buffer_max     = 256   # max images stored

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f"[S3-RL] Epoch {epoch+1}/{args.epochs}", leave=True)
        epoch_rewards, epoch_psnrs = [], []

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)

            # ── Optional Mistake Replay ──────────────────────────────────────
            if len(mistake_buffer) >= 8 and torch.rand(1).item() < 0.4:
                idxs   = torch.randperm(len(mistake_buffer))[:8]
                replay = torch.stack([mistake_buffer[i] for i in idxs]).to(device)
                images = torch.cat([images, replay], dim=0)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                recon_no_grad, _, _ = model(images)
                psnr_per = batch_psnr(recon_no_grad, images)   # (B,)

            # ── Reward & Advantage ───────────────────────────────────────────
            reward    = psnr_per - baseline_psnr               # +good, −mistake
            advantage = (reward - reward.mean()) / (reward.std() + 1e-8)
            # Weight: mistakes (low reward) get amplified, successes dampened
            mistake_weight = torch.clamp(1.0 - advantage, 0.3, 4.0).detach()

            # ── Forward pass with gradient ───────────────────────────────────
            recon, mu, logvar = model(images)

            # Per-sample weighted MSE
            mse_per = ((recon - images) ** 2).mean(dim=(1, 2, 3))   # (B,)
            weighted_mse = (mse_per * mistake_weight).mean() * 10.0

            # Structure & frequency losses (standard)
            ssim_l = ssim_loss(recon, images)
            qwm_l  = qwm_model(recon, images)
            logvar_c = torch.clamp(logvar, -10, 10)
            kld_l  = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())

            loss = weighted_mse + (0.7 * ssim_l) + (0.2 * qwm_l) + (0.0001 * kld_l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # ── Update baseline & mistake buffer ─────────────────────────────
            mean_psnr = psnr_per.mean().item()
            baseline_psnr = (1 - ema_alpha) * baseline_psnr + ema_alpha * mean_psnr
            epoch_rewards.append(reward.mean().item())
            epoch_psnrs.append(mean_psnr)

            # Store hard examples in mistake buffer (CPU)
            hard_mask = (psnr_per < psnr_thresh)
            for i, is_hard in enumerate(hard_mask):
                if is_hard and i < images.shape[0]:
                    if len(mistake_buffer) >= buffer_max:
                        mistake_buffer.pop(0)          # evict oldest
                    mistake_buffer.append(images[i].cpu().detach())

            pbar.set_postfix(
                psnr=f"{mean_psnr:.2f}dB",
                base=f"{baseline_psnr:.2f}dB",
                reward=f"{reward.mean().item():+.2f}",
                buf=len(mistake_buffer)
            )

        avg_psnr   = sum(epoch_psnrs)   / len(epoch_psnrs)
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        log.info(f"   [S3 Epoch {epoch+1}] Avg PSNR: {avg_psnr:.2f} dB | Avg Reward: {avg_reward:+.3f} | Buffer: {len(mistake_buffer)}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_channels':  args.latent_channels,
        'stage': 3, 'epoch': args.epochs,
        'final_baseline_psnr': baseline_psnr,
    }, save_path)
    log.info("   ✅ Stage 3 complete → %s", save_path)
    log.info("🏆 Training complete! Use 'stage3_rl_genesis_core.pth' for deployment.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox 3-Stage Training Pipeline")
    parser.add_argument("--stage",           type=int,   default=1,      help="Training stage: 1=Monster, 2=FineTune, 3=RL")
    parser.add_argument("--batch_size",      type=int,   default=12)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--latent_channels", type=int,   default=16)
    parser.add_argument("--sample_limit",    type=int,   default=10000)
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints")
    parser.add_argument("--use_hd",          type=bool,  default=True)
    parser.add_argument("--resume",          type=str,   default=None,   help="Path to checkpoint to resume/load from")
    args = parser.parse_args()

    if   args.stage == 1: stage1_monster(args)
    elif args.stage == 2: stage2_finetune(args)
    elif args.stage == 3: stage3_rl_mistakes(args)
    else: raise ValueError(f"Unknown stage {args.stage}. Choose 1, 2, or 3.")
