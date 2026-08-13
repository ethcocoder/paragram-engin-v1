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


# ── Building Block ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Paradox Residual Block — the topological anchor of the Genesis pipeline.

    A standard pre-activation residual connection stabilises gradient flow
    through deep encoder/decoder stacks.

    Args:
        channels: Number of convolutional feature channels (in == out).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


# ── Encoder ──────────────────────────────────────────────────────────────────

class SemanticEncoder(nn.Module):
    """
    Paradox Semantic Encoder — collapses an image into a Quantum Superposition.

    Applies three stride-2 convolutions to downsample 32×32 → 4×4, then
    projects to (mu, logvar) maps representing the latent Gaussian distribution.

    Args:
        latent_channels: Number of channels in the bottleneck latent map.

    Input shape:  (B, 3, H, W)
    Output shape: (mu, logvar) each of shape (B, latent_channels, H/8, W/8)
    """

    def __init__(self, latent_channels: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # 3 → 32 ch, spatial /2
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            # 32 → 64 ch, spatial /2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            # 64 → 128 ch, spatial /2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            # 128 → 256 ch, spatial /2 [New HD Stage]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
        )
        self.mu     = nn.Conv2d(256, latent_channels, kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(256, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layers(x)
        return self.mu(h), self.logvar(h)


# ── Decoder ──────────────────────────────────────────────────────────────────

class GenesisDecoder(nn.Module):
    """
    Paradox Genesis Decoder — collapses the quantum latent back into a physical image.

    Uses PixelShuffle for sub-pixel upsampling (no checkerboard artifacts),
    with a wide 256-channel manifold for high-fidelity reconstruction.

    PixelShuffle(r) math: (B, C*r², H, W) → (B, C, H*r, W*r)
      up1: (B, 256, 4, 4)  → (B, 64, 8, 8)    [r=2, C=64]
      up2: (B, 256, 8, 8)  → (B, 64, 16, 16)  [r=2, C=64]
      up3: (B,  48, 16,16) → (B, 12, 32, 32)  [r=2, C=12] → (B, 3, 32, 32)

    Args:
        latent_channels: Must match the encoder's latent_channels.

    Input shape:  (B, latent_channels, H/8, W/8)
    Output shape: (B, 3, H, W) with values in [-1, 1]
    """

    def __init__(self, latent_channels: int = 16, bottleneck_blocks: int = 4) -> None:
        super().__init__()
        if bottleneck_blocks < 4:
            raise ValueError("bottleneck_blocks must be at least 4 to load legacy Genesis checkpoints.")
        self.bottleneck_blocks = bottleneck_blocks
        # Keep the original six module positions unchanged so all legacy
        # decoder weights retain their state-dict names and can load directly.
        self.expand = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),  # Legacy Elite HD capacity
            ResBlock(256),  # Legacy Elite HD capacity
        )
        # Deep-decoder v2 appends residual reasoning blocks after the legacy
        # stack.  A zero-gamma final BatchNorm makes every added block start
        # as an identity transformation, preserving v1 output at initialization
        # while allowing the block to learn during fine-tuning.
        extras = []
        for _ in range(bottleneck_blocks - 4):
            block = ResBlock(256)
            nn.init.zeros_(block.conv[4].weight)
            nn.init.zeros_(block.conv[4].bias)
            extras.append(block)
        self.extra_bottleneck = nn.Sequential(*extras) if extras else nn.Identity()
        # 256ch → PixelShuffle(2) → 64ch, spatial ×2
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        # 64ch → 256ch → PixelShuffle(2) → 64ch, spatial ×2
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        # 64ch → 256ch → PixelShuffle(2) → 64ch, spatial ×2
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        # 64ch → 48ch → PixelShuffle(2) → 12ch → 3ch, spatial ×2
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.extra_bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


# ── Quantizer ───────────────────────────────────────────────────────────────

class SovereignQuantizer(nn.Module):
    """
    Paradox Sovereign Quantizer — the 8-bit bottleneck logic.

    Uses a Straight-Through Estimator (STE) to allow gradients to flow past the
    non-differentiable rounding operation. This enables the model to 'learn'
    a representation that is natively robust to 8-bit quantization.

    Args:
        levels: Number of steps (127.5 for 8-bit [-1, 1]).
    """
    def __init__(self, levels: float = 127.5) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator: round in forward, identity in backward
        x_clamped = torch.clamp(x, -1.0, 1.0)
        return x_clamped + (torch.round(x_clamped * self.levels) / self.levels - x_clamped).detach()


# ── Full Model ───────────────────────────────────────────────────────────────

class LatentGenesisCore(nn.Module):
    """
    Paradox Genesis Core — the Quantum-Neural VAE.

    Fuses classical deep learning with the Quantum Virtual Substrate (QVS)
    to produce a phase-modulated reparameterization trick that encodes
    latent space geometry into the QAU's Hilbert space.

    Compression Efficiency:
        Native (32x32x3x4 bytes) vs Compressed (16x16xLatent x 1 byte)
    """

    def __init__(self, latent_channels: int = 16, decoder_bottleneck_blocks: int = 4) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.decoder_bottleneck_blocks = decoder_bottleneck_blocks
        self.encoder = SemanticEncoder(latent_channels)
        self.decoder = GenesisDecoder(latent_channels, bottleneck_blocks=decoder_bottleneck_blocks)
        self.quantizer = SovereignQuantizer() # The 'Compression' gate
        self.qvs = QVS()  # Quantum Engine living inside the Neural Core

    def quantum_superposition(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        QVS-modulated reparameterization trick.

        During training, encodes each sample's mean latent intensity into a
        4-state quantum basis, applies a phase WEAVE, and uses the COLLAPSE
        outcome to determine a ±1 phase bias for the noise vector.

        During inference, uses low-temperature noise (σ=0.1) for slight
        stochastic diversity without distributional shift.

        Args:
            mu:     Latent mean,     shape (B, C, H, W).
            logvar: Latent log-var,  shape (B, C, H, W).

        Returns:
            z: Reparameterized latent sample, same shape as mu.
        """
        batch_size = mu.shape[0]
        std = torch.exp(0.5 * logvar)

        if self.training:
            phase_biases = []
            for i in range(batch_size):
                # Encode latent intensity into a 4-state quantum basis
                # (representing the 4 quadrants of the complex Hilbert space)
                asc_id = self.qvs.create_asc(size=2)
                self.qvs.SUPERPOSE(asc_id, [(0, 0), (0, 1), (1, 0), (1, 1)])

                # WEAVE a phase shift proportional to the mean latent energy
                intensity = torch.mean(mu[i]).item()
                self.qvs.WEAVE(asc_id, phase_angle=intensity * np.pi)

                # COLLAPSE to a binary outcome and map to ±1 bias
                outcome = self.qvs.COLLAPSE(asc_id)
                phase_biases.append(1.0 if sum(outcome) % 2 == 0 else -1.0)

                # Release quantum resources
                self.qvs.delete_asc(asc_id)

            bias_tensor = torch.tensor(
                phase_biases, dtype=torch.float32, device=mu.device
            ).view(batch_size, 1, 1, 1)
            eps = torch.randn_like(std) * bias_tensor
        else:
            # Inference: low-temperature sampling for smooth diversity
            eps = torch.randn_like(std) * 0.1

        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode → reparameterize → quantize → decode pipeline.

        Args:
            x: Input image batch, shape (B, 3, H, W), values in [-1, 1].

        Returns:
            reconstructed: Decoded image,  shape (B, 3, H, W).
            mu:            Latent mean,    shape (B, C, H/16, W/16).
            logvar:        Latent log-var, shape (B, C, H/16, W/16).
        """
        mu, logvar = self.encoder(x)
        z = self.quantum_superposition(mu, logvar)

        # Apply 8-bit Sovereign Quantization (Compression Logic)
        z_q = self.quantizer(z)

        reconstructed = self.decoder(z_q)
        return reconstructed, mu, logvar
