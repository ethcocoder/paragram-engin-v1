"""
hybrid_enhancer.py — Paradox v3 Sovereign Apex: The Hybrid Engine
==================================================================
Team B: The GAN-Diffusion Hybrid Receiver Enhancer.

This is the pinnacle of the Paradox receiver pipeline. It combines:

1. GAN Realism Path   — A residual attention generator creates sharp,
                        perceptually-convincing textures with adversarial
                        pressure from a multi-scale PatchGAN discriminator.

2. Diffusion Refinement Path — A lightweight conditional DDIM denoiser
                               runs a fixed-step reverse diffusion process
                               over the GAN output, iteratively removing
                               smearing and restoring manifold-correct
                               colour, contrast, and sharpness.

3. Fusion Gate        — A learned α-gate blends both paths per-pixel,
                        giving the diffusion path full control in noisy /
                        low-contrast zones while letting the GAN dominate
                        high-frequency texture regions.

Team A feeds 4 KB bottleneck latent → reconstructed base image.
Team B receives that base image and outputs a retina-grade restoration.

Loss formula (Team B training):
    L = λ_pix · L1_saliency
      + λ_perc · L_perceptual (VGG 3-layer)
      + λ_style· L_style (Gram)
      + λ_adv · L_LSGAN
      + λ_fm  · L_feature_matching
      + λ_iso · L_isotropic_laplacian
      + λ_diff· L_diffusion_denoising
"""

import os
import sys
import math
import argparse
import urllib.request
from pathlib import Path

# ── Advanced Pathing ─────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import LatentGenesisCore
from hd_data import CustomHDDataset, get_hd_dataloaders

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = "PJRT_DEVICE" in os.environ or "TPU_NAME" in os.environ
except ImportError:
    TPU_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 ── Loss Modules
# ═══════════════════════════════════════════════════════════════════════════════

class PerceptualStyleLoss(nn.Module):
    """
    Three-layer VGG-16 perceptual + Gram-matrix style loss.
    Identical to v1's proven PerceptualLoss, extended with Gram style terms.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.s1 = nn.Sequential(*vgg[:4])    # ReLU1_2 – edges
        self.s2 = nn.Sequential(*vgg[4:9])   # ReLU2_2 – textures
        self.s3 = nn.Sequential(*vgg[9:16])  # ReLU3_3 – structure
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    @staticmethod
    def _gram(f):
        b, c, h, w = f.shape
        x = f.view(b, c, h * w)
        return torch.bmm(x, x.transpose(1, 2)) / (c * h * w + 1e-8)

    def forward(self, pred, target):
        def norm(t):
            return (t * 0.5 + 0.5 - self.mean) / self.std
        p, t = norm(pred), norm(target)
        p1, t1 = self.s1(p), self.s1(t)
        p2, t2 = self.s2(p1), self.s2(t1)
        p3, t3 = self.s3(p2), self.s3(t2)
        loss_perc  = F.l1_loss(p1, t1) + F.l1_loss(p2, t2) + F.l1_loss(p3, t3)
        loss_style = (F.l1_loss(self._gram(p1), self._gram(t1)) +
                      F.l1_loss(self._gram(p2), self._gram(t2)))
        return loss_perc, loss_style


class IsotropicLaplacianLoss(nn.Module):
    """
    Direction-neutral sharpness enforcement via circular Laplacian.
    Kills horizontal/vertical streaking and directional blur.
    """
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[1,4,1],[4,-20,4],[1,4,1]]]], dtype=torch.float32)
        self.register_buffer("lap", lap)

    def forward(self, pred, target):
        p_g = pred.mean(1, keepdim=True)
        t_g = target.mean(1, keepdim=True)
        pad = F.pad
        kw  = self.lap
        p_e = F.conv2d(pad(p_g, (1,1,1,1), "reflect"), kw)
        t_e = F.conv2d(pad(t_g, (1,1,1,1), "reflect"), kw)
        return F.mse_loss(p_e, t_e)


class SaliencyWeightedL1(nn.Module):
    """
    L1 loss scaled by an edge-saliency map so complex areas
    (hair, fur, needles) are penalised more heavily than flat skies.
    """
    _EDGE = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32)

    def forward(self, pred, target, device):
        kern = self._EDGE.to(device)
        with torch.no_grad():
            gray = target.mean(1, keepdim=True)
            edge = torch.abs(F.conv2d(gray, kern, padding=1))
            sal  = torch.clamp(edge / (edge.mean() + 1e-8), 0.1, 5.0)
        return (F.l1_loss(pred, target, reduction="none") * sal).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── GAN PATH  (Team B — Generator)
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel gate (ratio=16)."""
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, max(1, ch // ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // ratio), ch, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)


class ResidualAttentionBlock(nn.Module):
    """
    Residual block with channel attention + learnable noise injection.
    Provides biological sharpness via lateral-inhibition sharpening.
    """
    def __init__(self, nf=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
        )
        self.ca          = ChannelAttention(nf)
        self.noise_scale = nn.Parameter(torch.zeros(1, nf, 1, 1))

    def forward(self, x):
        res = self.body(x)
        # Lateral inhibition: sharpening boost
        blurred = F.avg_pool2d(res, kernel_size=3, stride=1, padding=1)
        res = res + (res - blurred) * 0.3
        # Learnable stochastic noise
        res = res + torch.randn_like(res) * self.noise_scale
        res = self.ca(res)
        return x + res * 0.2   # residual scaling prevents explosion


class GANGenerator(nn.Module):
    """
    The Sovereign GAN Generator.
    Input : blurry/compressed base image  (B, 3, H, W)  ∈ [-1, 1]
    Output: sharpened residual image      (B, 3, H, W)  ∈ [-1, 1]

    Architecture: conv_in → N × RAB → conv_out, then x + residual (clamped).
    """
    def __init__(self, nf: int = 64, nb: int = 12):
        super().__init__()
        self.entry = nn.Conv2d(3, nf, 3, 1, 1)
        self.body  = nn.Sequential(*[ResidualAttentionBlock(nf) for _ in range(nb)])
        # Saliency-gated entropy injection (SEI)
        self.sei_strength = nn.Parameter(torch.ones(1, nf, 1, 1) * 0.003)
        self.exit  = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 3, 3, 1, 1),
        )

    def forward(self, x):
        feat = self.entry(x)

        # SEI: inject entropy only in high-complexity zones
        with torch.no_grad():
            gray  = x.mean(1, keepdim=True)
            cpx   = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean(2, keepdim=True)
            gate  = F.interpolate(cpx, size=feat.shape[2:], mode="bilinear", align_corners=False)
            gate  = (gate - gate.min()) / (gate.max() - gate.min() + 1e-8)
        feat = feat + torch.randn_like(feat) * self.sei_strength * gate

        trunk    = self.body(feat)
        residual = self.exit(trunk)
        return torch.clamp(x + residual, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── DIFFUSION PATH  (Team B — Denoiser)
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


class DiffusionUNetBlock(nn.Module):
    """Slim U-Net block with time-step conditioning."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        groups_in = 8 if in_ch % 8 == 0 else (3 if in_ch % 3 == 0 else 1)
        self.conv1 = nn.Sequential(nn.GroupNorm(groups_in, in_ch), nn.SiLU(),
                                   nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        groups_out = 8 if out_ch % 8 == 0 else (3 if out_ch % 3 == 0 else 1)
        self.conv2 = nn.Sequential(nn.GroupNorm(groups_out, out_ch), nn.SiLU(),
                                   nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class ConditionalDiffusionDenoiser(nn.Module):
    """
    Lightweight conditional DDIM denoiser (U-Net style).

    Conditioned on the GAN output (guide) AND the original base image.
    Predicts the noise residual ε = ε̂(x_t, t, condition).
    Channels: [noisy(3) + gan_out(3) + base(3)] → predict ε(3)

    At inference we run T_inf DDIM steps (default 4) for a fast,
    deterministic reverse-diffusion refinement.
    """
    def __init__(self, base_ch: int = 48, time_dim: int = 128,
                 T_train: int = 1000, T_inf: int = 4):
        super().__init__()
        self.T_train    = T_train
        self.T_inf      = T_inf
        self.time_embed = SinusoidalEmbedding(time_dim)

        # Encoder
        self.enc1 = DiffusionUNetBlock(9, base_ch,      time_dim)   # 9 = 3+3+3
        self.enc2 = DiffusionUNetBlock(base_ch, base_ch*2, time_dim)
        self.enc3 = DiffusionUNetBlock(base_ch*2, base_ch*4, time_dim)
        # Bottleneck
        self.mid  = DiffusionUNetBlock(base_ch*4, base_ch*4, time_dim)
        # Decoder  (with skip connections)
        self.dec3 = DiffusionUNetBlock(base_ch*4 + base_ch*4, base_ch*2, time_dim)
        self.dec2 = DiffusionUNetBlock(base_ch*2 + base_ch*2, base_ch,   time_dim)
        self.dec1 = DiffusionUNetBlock(base_ch   + base_ch,   base_ch,   time_dim)
        self.out  = nn.Conv2d(base_ch, 3, 1)

        # Noise schedule (cosine β)
        betas  = self._cosine_betas(T_train)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab",   alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab", (1.0 - alpha_bar).sqrt())

    @staticmethod
    def _cosine_betas(T, s=0.008):
        steps  = torch.arange(T + 1, dtype=torch.float64)
        f      = torch.cos(((steps / T + s) / (1 + s)) * math.pi * 0.5) ** 2
        ab     = f / f[0]
        betas  = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999)
        return betas.float()

    def _unet(self, x_noisy, gan_out, base, t_emb):
        inp  = torch.cat([x_noisy, gan_out, base], dim=1)     # (B, 9, H, W)
        e1   = self.enc1(inp,  t_emb)
        e2   = self.enc2(F.avg_pool2d(e1, 2), t_emb)
        e3   = self.enc3(F.avg_pool2d(e2, 2), t_emb)
        m    = self.mid(e3, t_emb)
        d3   = self.dec3(torch.cat([m,  e3], 1), t_emb)
        d3   = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2   = self.dec2(torch.cat([d3, e2], 1), t_emb)
        d2   = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1   = self.dec1(torch.cat([d2, e1], 1), t_emb)
        return self.out(d1)

    def forward_train(self, x0, gan_out):
        """
        Training forward: randomly sample t, add noise, predict ε.
        Returns predicted epsilon and the true epsilon (for L2 loss).
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T_train, (B,), device=x0.device)
        eps   = torch.randn_like(x0)
        x_t   = self.sqrt_ab[t, None, None, None] * x0 + self.sqrt_1mab[t, None, None, None] * eps
        t_emb = self.time_embed(t)
        eps_pred = self._unet(x_t, gan_out, x0, t_emb)
        return eps_pred, eps

    @torch.no_grad()
    def ddim_refine(self, gan_out, base, steps=None):
        """
        DDIM deterministic refinement (inference only).
        Starts from x_T = slightly noised GAN output for guided denoising.
        Returns the refined output ∈ [-1, 1].
        """
        steps = steps or self.T_inf
        B, _, H, W = gan_out.shape
        device = gan_out.device

        # Build DDIM timestep subsequence
        stride = self.T_train // steps
        ts     = list(range(self.T_train - 1, -1, -stride))[:steps]

        # Start from lightly-noised GAN output
        noise_level = self.sqrt_1mab[ts[0]]
        x = (self.sqrt_ab[ts[0]] * gan_out +
             noise_level * torch.randn_like(gan_out))

        for i, t_val in enumerate(ts):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_emb   = self.time_embed(t_batch)
            eps_hat = self._unet(x, gan_out, base, t_emb)

            ab_t    = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[ts[i+1]] if i+1 < len(ts) else torch.tensor(1.0)
            ab_prev = ab_prev.to(device)

            x0_pred = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt().clamp(min=1e-8)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_hat

        return torch.clamp(x, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── FUSION  (Team B — Alpha Gate)
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaFusionGate(nn.Module):
    """
    Learned per-pixel α-gate that blends the GAN and Diffusion outputs.

    α ≈ 1 in high-frequency / texture zones → GAN dominates (sharpness).
    α ≈ 0 in smooth / uncertain zones       → Diffusion dominates (fidelity).

    Input:  GAN output + Diffusion output + base (9 ch)
    Output: fused image  (B, 3, H, W)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),           # α ∈ [0, 1]
        )

    def forward(self, gan_out, diff_out, base):
        inp   = torch.cat([gan_out, diff_out, base], dim=1)    # (B, 9, H, W)
        alpha = self.net(inp)
        return alpha * gan_out + (1.0 - alpha) * diff_out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── DISCRIMINATOR  (Multi-Scale PatchGAN)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchDiscriminatorScale(nn.Module):
    """Single-scale PatchGAN discriminator with GroupNorm for stability."""
    def __init__(self, in_ch=3):
        super().__init__()
        def _block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1, bias=not norm)]
            if norm:
                groups = 8 if oc % 8 == 0 else (3 if oc % 3 == 0 else 1)
                layers.append(nn.GroupNorm(groups, oc, eps=1e-4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *_block(in_ch, 64,  stride=2, norm=False),
            *_block(64,  128, stride=2),
            *_block(128, 256, stride=2),
            *_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        feats = []
        for layer in self.model[:-1]:
            x = layer(x)
            feats.append(x)
        output = self.model[-1](x)
        return feats, output


class MultiScaleDiscriminator(nn.Module):
    """
    Two-scale PatchGAN. The second scale operates on 2× downsampled input
    to capture both local texture quality and global composition.
    """
    def __init__(self):
        super().__init__()
        self.d1 = PatchDiscriminatorScale(3)
        self.d2 = PatchDiscriminatorScale(3)
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        return self.d1(x), self.d2(self.down(x))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── FULL TEAM-B MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class SovereignHybridEnhancer(nn.Module):
    """
    The complete Team-B Hybrid Enhancer.

    Call modes:
        forward(base)           → fused output  (inference)
        forward_train(base, gt) → (fused, gan_out, diff_eps_pred, diff_eps_true)
    """
    def __init__(self, gan_nf=64, gan_nb=12, diff_base_ch=48,
                 T_train=1000, T_inf=4):
        super().__init__()
        self.generator = GANGenerator(nf=gan_nf, nb=gan_nb)
        self.denoiser  = ConditionalDiffusionDenoiser(base_ch=diff_base_ch,
                                                       T_train=T_train,
                                                       T_inf=T_inf)
        self.gate      = AlphaFusionGate()

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        """Inference: returns the final fused enhanced image."""
        gan_out  = self.generator(base)
        diff_out = self.denoiser.ddim_refine(gan_out, base)
        return self.gate(gan_out, diff_out, base)

    def forward_train(self, base: torch.Tensor, gt: torch.Tensor):
        """
        Training forward.
        Returns:
            fused      – final fused output
            gan_out    – raw GAN output (for adversarial + perceptual losses)
            eps_pred   – denoiser predicted noise
            eps_true   – ground-truth noise (for diffusion loss)
        """
        gan_out             = self.generator(base)
        eps_pred, eps_true  = self.denoiser.forward_train(gt, gan_out)
        diff_out            = self.denoiser.ddim_refine(gan_out, base,
                                                         steps=self.denoiser.T_inf)
        fused               = self.gate(gan_out, diff_out, base)
        return fused, gan_out, eps_pred, eps_true


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_device():
    if TPU_AVAILABLE:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_hybrid(args):
    device = _get_device()
    print(f"\n{'='*60}")
    print(f"  🔱 Paradox v3 Sovereign Hybrid — TEAM B TRAINING")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = get_hd_dataloaders(
        image_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    if loader is None:
        print("[!] No data found. Place HD images in:", args.data_dir)
        return

    # ── Team A (frozen) ───────────────────────────────────────────────────────
    team_a = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    ckpt   = torch.load(args.sender_path, map_location="cpu")
    state  = ckpt.get("model_state_dict", ckpt)
    team_a.load_state_dict(state, strict=True)
    team_a.eval()
    for p in team_a.parameters():
        p.requires_grad = False
    print(f"[✓] Team A loaded from {args.sender_path}  (frozen)")

    # ── Team B ────────────────────────────────────────────────────────────────
    team_b = SovereignHybridEnhancer(
        gan_nf=args.gan_nf, gan_nb=args.gan_nb,
        diff_base_ch=args.diff_base_ch,
        T_train=args.T_train, T_inf=args.T_inf,
    ).to(device)

    netD  = MultiScaleDiscriminator().to(device)

    # Resume if checkpoint exists
    if args.resume and os.path.exists(args.enhancer_path):
        sd = torch.load(args.enhancer_path, map_location="cpu")
        team_b.load_state_dict(sd["model_state_dict"])
        print(f"[✓] Resumed Team B from {args.enhancer_path}")

    # ── Losses ────────────────────────────────────────────────────────────────
    perc_style = PerceptualStyleLoss().to(device).eval()
    iso_loss   = IsotropicLaplacianLoss().to(device).eval()
    sal_l1     = SaliencyWeightedL1()
    criterion_gan = nn.MSELoss()     # LSGAN for stability

    # ── Optimisers ────────────────────────────────────────────────────────────
    optG = optim.Adam(team_b.parameters(),       lr=args.lr_g,
                      betas=(0.5, 0.999), eps=1e-8)
    optD = optim.Adam(netD.parameters(),          lr=args.lr_d,
                      betas=(0.5, 0.999), eps=1e-8)
    schedG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=args.epochs)
    schedD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.enhancer_path), exist_ok=True)

    for epoch in range(args.epochs):
        team_b.train();  netD.train()
        train_iter = (
            pl.ParallelLoader(loader, [device]).per_device_loader(device)
            if TPU_AVAILABLE else loader
        )
        pbar = tqdm(total=len(loader),
                    desc=f"Epoch {epoch+1}/{args.epochs}")

        for real_imgs, _ in train_iter:
            if not TPU_AVAILABLE:
                real_imgs = real_imgs.to(device)

            # ── Team A — produce base (4-KB bottleneck output) ───────────────
            with torch.no_grad():
                base, _, _ = team_a(real_imgs)

            # ── Team B — forward_train ────────────────────────────────────────
            fused, gan_out, eps_pred, eps_true = team_b.forward_train(base, real_imgs)

            # ════════════════════════════════════════════════════════════════
            # Discriminator step
            # ════════════════════════════════════════════════════════════════
            optD.zero_grad()
            (rf1, rf2), (rg1, rg2) = netD(real_imgs)
            (ff1, ff2), (fg1, fg2) = netD(fused.detach())

            loss_D = 0.5 * (
                criterion_gan(rf2, torch.ones_like(rf2))  +
                criterion_gan(ff2, torch.zeros_like(ff2)) +
                criterion_gan(rg2, torch.ones_like(rg2))  +
                criterion_gan(fg2, torch.zeros_like(fg2))
            )
            loss_D.backward()
            nn.utils.clip_grad_norm_(netD.parameters(), 0.5)
            if TPU_AVAILABLE: xm.optimizer_step(optD)
            else:             optD.step()

            # ════════════════════════════════════════════════════════════════
            # Generator step
            # ════════════════════════════════════════════════════════════════
            optG.zero_grad()
            (pf1, pf2), (pg1, pg2) = netD(fused)
            (pr1, pr2), (prg1, prg2) = netD(real_imgs)

            # 1. LSGAN adversarial
            loss_adv = 0.5 * (
                criterion_gan(pf2, torch.ones_like(pf2)) +
                criterion_gan(pg2, torch.ones_like(pg2))
            )

            # 2. Feature matching (multi-scale)
            loss_fm = sum(
                F.l1_loss(pf1[i], pr1[i].detach()) +
                F.l1_loss(pg1[i], prg1[i].detach())
                for i in range(len(pf1))
            ) / max(1, len(pf1))

            # 3. Perceptual + Style
            loss_perc, loss_style = perc_style(fused, real_imgs)

            # 4. Isotropic sharpness
            loss_iso = iso_loss(fused, real_imgs)

            # 5. Saliency-weighted pixel L1
            loss_pix = sal_l1(fused, real_imgs, device)

            # 6. Diffusion denoising (MSE on ε)
            loss_diff = F.mse_loss(eps_pred, eps_true)

            # 7. GAN-path perceptual (intermediate sharpness)
            loss_gan_perc, _ = perc_style(gan_out, real_imgs)

            # ── SOVEREIGN APEX FORMULA ────────────────────────────────────────
            loss_G = (
                args.w_pix   * loss_pix   +
                args.w_perc  * loss_perc  +
                args.w_style * loss_style +
                args.w_adv   * loss_adv   +
                args.w_fm    * loss_fm    +
                args.w_iso   * loss_iso   +
                args.w_diff  * loss_diff  +
                args.w_gp    * loss_gan_perc
            )

            loss_G.backward()
            nn.utils.clip_grad_norm_(team_b.parameters(), 0.5)
            if TPU_AVAILABLE: xm.optimizer_step(optG)
            else:             optG.step()

            pbar.set_postfix(
                D=f"{loss_D.item():.3f}",
                G=f"{loss_G.item():.3f}",
                iso=f"{loss_iso.item():.4f}",
                diff=f"{loss_diff.item():.4f}",
            )
            pbar.update(1)

        pbar.close()
        schedG.step();  schedD.step()

        # Save checkpoint every epoch
        save_fn = xm.save if TPU_AVAILABLE else torch.save
        save_fn({"model_state_dict": team_b.state_dict(),
                 "epoch": epoch + 1},
                args.enhancer_path)
        print(f"[✓] Checkpoint saved  →  {args.enhancer_path}")

    print("\n[🏆] Training complete! Sovereign Hybrid Enhancer is ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── DEMO / INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def demo(args):
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sovereign Demo — {device}")

    # Team A
    team_a = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    ckpt   = torch.load(args.sender_path, map_location="cpu")
    team_a.load_state_dict(ckpt.get("model_state_dict", ckpt))
    team_a.eval()

    # Team B
    team_b = SovereignHybridEnhancer(
        gan_nf=args.gan_nf, gan_nb=args.gan_nb,
        diff_base_ch=args.diff_base_ch,
        T_train=args.T_train, T_inf=args.T_inf,
    ).to(device)
    sd = torch.load(args.enhancer_path, map_location="cpu")
    team_b.load_state_dict(sd["model_state_dict"])
    team_b.eval()

    # Fetch random HD image
    seed = torch.randint(0, 99999, (1,)).item()
    url  = f"https://picsum.photos/seed/{seed}/1024/1024"
    print(f"[*] Fetching: {url}")
    urllib.request.urlretrieve(url, "_demo_input.jpg")

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    img = tfm(Image.open("_demo_input.jpg").convert("RGB")).unsqueeze(0).to(device)

    def unnorm(t):
        return torch.clamp(t[0].cpu() * 0.5 + 0.5, 0, 1).permute(1,2,0).numpy()

    with torch.no_grad():
        base, _, _ = team_a(img)
        enhanced   = team_b(base)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["🖼 Original", "📡 Team A (4-KB Bottleneck)", "🔱 Team B (Sovereign Hybrid)"]
    imgs   = [unnorm(img), unnorm(base), unnorm(enhanced)]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im);  ax.set_title(title, fontsize=14, fontweight="bold");  ax.axis("off")

    plt.suptitle("Paradox v3 — Sovereign Hybrid Enhancer Demo", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = "sovereign_hybrid_demo.png"
    plt.savefig(out, dpi=200)
    print(f"[🏆] Demo saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ── CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Paradox v3 Sovereign Hybrid Enhancer — Team A + Team B"
    )
    p.add_argument("--mode",         type=str, default="train",
                   choices=["train", "demo"])

    # Paths
    p.add_argument("--sender_path",   type=str,
                   default="checkpoints/universal_genesis_core.pth")
    p.add_argument("--enhancer_path", type=str,
                   default="checkpoints/sovereign_hybrid_enhancer.pth")
    p.add_argument("--data_dir",      type=str, default="hd_images")

    # Team A
    p.add_argument("--latent_channels", type=int, default=16)

    # Team B architecture
    p.add_argument("--gan_nf",        type=int, default=64)
    p.add_argument("--gan_nb",        type=int, default=12)
    p.add_argument("--diff_base_ch",  type=int, default=48)
    p.add_argument("--T_train",       type=int, default=1000)
    p.add_argument("--T_inf",         type=int, default=4)

    # Training hyper-params
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr_g",       type=float, default=1e-4)
    p.add_argument("--lr_d",       type=float, default=5e-5)
    p.add_argument("--resume",     action="store_true")

    # Loss weights
    p.add_argument("--w_pix",   type=float, default=5.0)
    p.add_argument("--w_perc",  type=float, default=5.0)
    p.add_argument("--w_style", type=float, default=3.0)
    p.add_argument("--w_adv",   type=float, default=1.0)
    p.add_argument("--w_fm",    type=float, default=10.0)
    p.add_argument("--w_iso",   type=float, default=8.0)
    p.add_argument("--w_diff",  type=float, default=10.0)
    p.add_argument("--w_gp",    type=float, default=2.0)

    args = p.parse_args()

    if args.mode == "train":
        train_hybrid(args)
    else:
        demo(args)
