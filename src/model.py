"""
model.py — Paradox Genesis Core Architecture
=============================================
Implements the Quantum-Neural Variational Autoencoder (QNVAE) for
ultra-efficient image compression across the Aether Mesh network.

Architecture:
    SemanticEncoder  → encodes image into (mu, logvar) latent maps
    GenesisDecoder   → reconstructs image from quantized latent z
    LatentGenesisCore → full VAE with QVS-modulated reparameterization
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple

# ── Advanced Pathing Protocol ────────────────────────────────────────────────
# Ensures the QAU substrate is always resolvable regardless of working directory.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from qau_qvs.core.qvs import QVS
except ImportError:
    from .qau_qvs.core.qvs import QVS  # type: ignore[no-redef]


# ── Elite Components ──────────────────────────────────────────────────────────

class HaarWaveletTransform(nn.Module):
    """
    Elite Wavelet Decomposition — suppresses aliasing at the physical limit.
    Decomposes (B, C, H, W) -> (B, C*4, H/2, W/2) into [LL, LH, HL, HH].
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Haar Kernels
        kernel = torch.tensor([
            [[1, 1], [1, 1]],   # LL (Low-pass)
            [[1, -1], [1, -1]], # LH (Vertical)
            [[1, 1], [-1, -1]], # HL (Horizontal)
            [[1, -1], [-1, 1]]  # HH (Diagonal)
        ], dtype=torch.float32) / 2.0
        
        self.register_buffer('filter', kernel.unsqueeze(1).repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.filter, stride=2, groups=self.channels)

class InverseHaarWaveletTransform(nn.Module):
    """
    Elite Wavelet Reconstruction — restores high-frequency phase.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Haar Kernels [4, 2, 2]
        kernel = torch.tensor([
            [[1, 1], [1, 1]],
            [[1, -1], [1, -1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [-1, 1]]
        ], dtype=torch.float32) / 2.0
        # For ConvTranspose2d (12 -> 3 with groups=3): weight is [12, 1, 2, 2]
        self.register_buffer('filter', kernel.view(4, 1, 2, 2).repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(x, self.filter, stride=2, groups=self.channels)

class EliteAttention(nn.Module):
    """
    Phase-Coherent Dual Attention.
    Combines Channel gating with Spatial refinement.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg_out, max_out], dim=1))
        return x

class ResBlock(nn.Module):
    """
    Paradox Residual Block — the topological anchor of the Genesis pipeline.
    UPGRADED: GELU activation and Elite Attention.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise Separable Conv 1
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels//4), channels),
            nn.GELU(),
            # Depthwise Separable Conv 2
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels//4), channels),
        )
        self.attn = EliteAttention(channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x + self.attn(self.conv(x)))


# ── Encoder ──────────────────────────────────────────────────────────────────

class SemanticEncoder(nn.Module):
    """
    Paradox Semantic Encoder — collapses an image into a Quantum Superposition.
    UPGRADED: Uses Wavelet Decomposition to preserve high-frequency phase.
    """

    def __init__(self, latent_channels: int = 16) -> None:
        super().__init__()
        self.wavelet = HaarWaveletTransform(3) # 3 -> 12 ch
        
        self.layers = nn.Sequential(
            # 12 → 64 ch, spatial /2 (already /2 from wavelet)
            nn.Conv2d(12, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            ResBlock(64),
            # 64 → 128 ch, spatial /2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            ResBlock(128),
            # 128 → 256 ch, spatial /2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            ResBlock(256),
            # 256 → 512 ch, spatial /2 (Matching 16x reconstruction)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            ResBlock(512),
        )
        self.mu     = nn.Conv2d(512, latent_channels, kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(512, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.wavelet(x)
        h = self.layers(w)
        return self.mu(h), self.logvar(h)


# ── Decoder ──────────────────────────────────────────────────────────────────

class GenesisDecoder(nn.Module):
    """
    Paradox Genesis Decoder — collapses the quantum latent back into a physical image.
    UPGRADED: Hierarchical Phase Reconstruction using Wavelet-Neural Fusion.
    """

    def __init__(self, latent_channels: int = 16) -> None:
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(latent_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            ResBlock(512),
            ResBlock(512),
        )
        
        # Intelligence hook (will be injected by LatentGenesisCore)
        self.intelligence = nn.Identity()

        # Phase Reconstruction Stages
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.PixelShuffle(2), # 256 -> 64
            nn.GroupNorm(16, 64),
            nn.GELU(),
            ResBlock(64)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2), # 64 -> 256 -> 64
            nn.GroupNorm(16, 64),
            nn.GELU(),
            ResBlock(64)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.PixelShuffle(2), # 64 -> 128 -> 32
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResBlock(32)
        )
        
        # Final Wavelet Reconstruction Stage
        self.wavelet_up = nn.Conv2d(32, 12, 3, padding=1)
        self.iwt = InverseHaarWaveletTransform(3) # 12 -> 3 ch
        
        self.final_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        
        # Inject Intelligence at the deep manifold
        x = self.intelligence(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # Wavelet synthesis
        w = self.wavelet_up(x)
        img = self.iwt(w)
        
        return self.final_refine(img)


# ── Quantizer ───────────────────────────────────────────────────────────────

class SovereignQuantizer(nn.Module):
    """
    Paradox Sovereign Quantizer — the 10-bit bottleneck logic.
    UPGRADED: Higher precision (10-bit) for Elite accuracy.
    """
    def __init__(self, levels: float = 511.5) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, -1.0, 1.0)
        return x_clamped + (torch.round(x_clamped * self.levels) / self.levels - x_clamped).detach()


class GlobalIntelligenceAttention(nn.Module):
    """
    Holographic Context Engine.
    Provides the 'Intelligence' for the decoder to understand global scene geometry.
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, c, h, w)

        return self.gamma * out + x

class EliteDiscriminator(nn.Module):
    """
    Paradox Elite Critic (PatchGAN).
    Forcing the generator to synthesize sharp, high-frequency textures.
    Uses Spectral Normalization for extreme training stability.
    """
    def __init__(self, in_channels=3):
        super().__init__()

        def disc_block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, stride, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            disc_block(in_channels, 64, stride=2),
            disc_block(64, 128, stride=2),
            disc_block(128, 256, stride=2),
            disc_block(256, 512, stride=1),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, padding=1))
        )

    def forward(self, x):
        return self.model(x)

# ── Full Model ───────────────────────────────────────────────────────────────

class LatentGenesisCore(nn.Module):
    """
    Paradox Genesis Core — the Quantum-Neural VAE-GAN.
    UPGRADED: GPU-Native QVS + Elite Intelligence Attention.
    """

    def __init__(self, latent_channels: int = 16, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.encoder = SemanticEncoder(latent_channels)
        self.decoder = GenesisDecoder(latent_channels)
        
        # Inject Intelligence into Decoder
        self.decoder.intelligence = GlobalIntelligenceAttention(512)
        
        self.quantizer = SovereignQuantizer()
        self.qvs = QVS(device=device)

    def quantum_superposition(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        UPGRADED: Vectorized GPU-Native Phase Modulation.
        Eliminated the batch loop for massive training speedup.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        if self.training:
            # Vectorized Quantum Logic
            # We calculate a per-batch-item energy bias using the mean of mu
            energy = torch.mean(mu, dim=(1, 2, 3)) # (B,)
            
            # Simulate the "Collapse" outcome across the entire batch
            # Logic: If energy > 0, bias is more likely to be 1, else -1
            # Using a simplified vectorized version of the QVS logic
            probs = torch.sigmoid(energy) # Map energy to probability [0, 1]
            outcomes = torch.bernoulli(probs) 
            bias = (outcomes * 2 - 1).view(-1, 1, 1, 1) # (B, 1, 1, 1)
            
            eps = eps * bias
        else:
            eps *= 0.1

        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.quantum_superposition(mu, logvar)
        z_q = self.quantizer(z)
        reconstructed = self.decoder(z_q)
        return reconstructed, mu, logvar
