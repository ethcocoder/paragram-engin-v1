"""Compatibility checks for the approximately 80 MB deep-decoder v2 architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune_deterministic import load_compatible_weights
from model import LatentGenesisCore


def test_v1_weights_bootstrap_deep_decoder_without_changing_output() -> None:
    checkpoint_path = ROOT / "checkpoints" / "universal_genesis_core_ft_decoder_v1.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    v1 = LatentGenesisCore(latent_channels=16, decoder_bottleneck_blocks=4).eval()
    v1.load_state_dict(checkpoint["model_state_dict"], strict=True)

    v2 = LatentGenesisCore(latent_channels=16, decoder_bottleneck_blocks=14).eval()
    load_compatible_weights(v2, checkpoint, source_blocks=4, target_blocks=14)

    # The added blocks start as identity transformations, so loading a v1
    # checkpoint into v2 must not perturb the initial reconstruction.
    torch.manual_seed(1337)
    latent = torch.rand(2, 16, 16, 16) * 2.0 - 1.0
    with torch.no_grad():
        v1_output = v1.decoder(latent)
        v2_output = v2.decoder(latent)

    torch.testing.assert_close(v1_output, v2_output, rtol=0.0, atol=0.0)
    assert latent[0].numel() == 4096
    assert v2.decoder.bottleneck_blocks == 14


def test_deep_decoder_parameter_budget_is_near_80mb() -> None:
    v2 = LatentGenesisCore(latent_channels=16, decoder_bottleneck_blocks=14)
    parameter_count = sum(parameter.numel() for parameter in v2.parameters())
    float32_megabytes = parameter_count * 4 / 1_000_000

    assert 79.0 <= float32_megabytes <= 81.0


if __name__ == "__main__":
    test_v1_weights_bootstrap_deep_decoder_without_changing_output()
    test_deep_decoder_parameter_budget_is_near_80mb()
    print("deep-decoder v2 compatibility tests passed")
