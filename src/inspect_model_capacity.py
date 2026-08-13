"""Inspect parameter and checkpoint capacity for a Genesis model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import LatentGenesisCore, ResBlock


def parameter_summary(module: torch.nn.Module) -> tuple[int, int]:
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    return parameter_count, parameter_count * 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Genesis model parameter capacity.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/universal_genesis_core_ft_decoder_v1.pth",
    )
    parser.add_argument("--latent_channels", type=int, default=16)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("fine_tuning", {})
    if not isinstance(metadata, dict):
        metadata = {}
    latent_channels = int(checkpoint.get("latent_channels", args.latent_channels))
    decoder_bottleneck_blocks = int(
        checkpoint.get("decoder_bottleneck_blocks", metadata.get("decoder_bottleneck_blocks", 4))
    )
    model = LatentGenesisCore(
        latent_channels=latent_channels,
        decoder_bottleneck_blocks=decoder_bottleneck_blocks,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    sections = {
        "encoder": model.encoder,
        "decoder": model.decoder,
        "quantizer": model.quantizer,
        "full_model": model,
    }
    print(f"checkpoint={checkpoint_path}")
    print(f"latent_channels={latent_channels}")
    print(f"decoder_bottleneck_blocks={decoder_bottleneck_blocks}")
    for name, module in sections.items():
        count, raw_bytes = parameter_summary(module)
        print(f"{name}: {count:,} parameters | {raw_bytes / 1_000_000:.2f} MB float32")

    # Adding residual blocks after the original four keeps the packet interface
    # and decoder-width contract intact.  Estimate variants from the v1 base,
    # or show the loaded size directly for an existing deep checkpoint.
    bottleneck_block_params, _ = parameter_summary(ResBlock(256))
    print(f"bottleneck_resblock_256: {bottleneck_block_params:,} parameters")
    if decoder_bottleneck_blocks == 4:
        full_params, _ = parameter_summary(model)
        print("deep_decoder_variants:")
        for total_blocks in (12, 14, 15):
            added_blocks = total_blocks - 4
            estimated_params = full_params + added_blocks * bottleneck_block_params
            print(
                f"  {total_blocks} total bottleneck blocks: "
                f"{estimated_params:,} parameters | {estimated_params * 4 / 1_000_000:.2f} MB float32"
            )


if __name__ == "__main__":
    main()
