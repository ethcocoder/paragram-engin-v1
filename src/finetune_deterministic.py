"""Fine-tune the Universal Genesis checkpoint on its deployed packet path.

This script deliberately trains the reconstruction path used in ``demo_hd.py``:

    image -> encoder mean (mu) -> 8-bit quantization -> decoder -> reconstruction

It does not change the latent geometry or quantization contract.  For 256x256
inputs with 16 channels, the packet remains a 16x16x16 quantized latent map
(4,096 values / 4,096 raw bytes before transport-container overhead).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from model import LatentGenesisCore
from train import PerceptualLoss, ssim_loss


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


IMAGE_SIZE = 256
QUANTIZATION_BITS = 8


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    """Make data sampling and fine-tuning runs reproducible when possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load and validate a checkpoint produced by the existing training code."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint must be a dictionary containing 'model_state_dict'. "
            f"Received an unsupported checkpoint at {path}."
        )
    return checkpoint


def decoder_bottleneck_blocks_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    """Read the decoder-depth contract, treating legacy checkpoints as four blocks."""
    metadata = checkpoint.get("fine_tuning", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return int(checkpoint.get("decoder_bottleneck_blocks", metadata.get("decoder_bottleneck_blocks", 4)))


def load_compatible_weights(
    model: LatentGenesisCore,
    checkpoint: Dict[str, Any],
    source_blocks: int,
    target_blocks: int,
) -> None:
    """Load v1 weights into an equal or deeper decoder without silent mismatch.

    Only parameters belonging to newly appended identity-initialized decoder
    blocks may be absent.  Any other missing or unexpected weight is treated as
    a versioning error rather than being silently ignored.
    """
    if target_blocks < source_blocks:
        raise ValueError(
            f"Cannot load a {source_blocks}-block checkpoint into a shallower "
            f"{target_blocks}-block decoder."
        )
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing_prefix = "decoder.extra_bottleneck."
    invalid_missing = [key for key in result.missing_keys if not key.startswith(allowed_missing_prefix)]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is not compatible with the requested architecture. "
            f"Missing: {invalid_missing}; unexpected: {result.unexpected_keys}"
        )
    if source_blocks == target_blocks and result.missing_keys:
        raise RuntimeError(
            f"Matching-depth checkpoint unexpectedly missed parameters: {result.missing_keys}"
        )


def deterministic_latent(
    model: LatentGenesisCore, images: torch.Tensor
) -> torch.Tensor:
    """Return precisely the latent representation used by the current demo.

    ``demo_hd.py`` encodes ``mu`` and applies an 8-bit round/clamp operation.
    ``model.quantizer(mu)`` performs the identical forward operation while
    retaining straight-through gradients for full-model fine-tuning.
    """
    mu, _ = model.encoder(images)
    return model.quantizer(mu)


def payload_contract(latent: torch.Tensor, latent_channels: int) -> Dict[str, int]:
    """Validate and describe the fixed wire payload contract."""
    if latent.ndim != 4:
        raise ValueError(f"Expected a four-dimensional latent tensor, got {latent.shape}.")
    _, channels, height, width = latent.shape
    if channels != latent_channels:
        raise ValueError(
            f"Latent channels changed from {latent_channels} to {channels}; refusing to save."
        )
    expected_spatial = IMAGE_SIZE // 16
    if height != expected_spatial or width != expected_spatial:
        raise ValueError(
            "The latent spatial geometry changed. Expected "
            f"{expected_spatial}x{expected_spatial}, received {height}x{width}."
        )
    values_per_image = channels * height * width
    return {
        "image_height": IMAGE_SIZE,
        "image_width": IMAGE_SIZE,
        "latent_channels": channels,
        "latent_height": height,
        "latent_width": width,
        "quantization_bits": QUANTIZATION_BITS,
        "values_per_image": values_per_image,
        "raw_payload_bytes_per_image": values_per_image,
    }


def psnr_from_normalized(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute PSNR after converting model values from [-1, 1] to [0, 1]."""
    reconstruction = torch.clamp(reconstruction * 0.5 + 0.5, 0.0, 1.0)
    target = torch.clamp(target * 0.5 + 0.5, 0.0, 1.0)
    mse = torch.mean((reconstruction - target) ** 2, dim=(1, 2, 3))
    return 20.0 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-12)))


def gradient_detail_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize lost horizontal and vertical image detail without changing payload size."""
    reconstruction_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    reconstruction_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(reconstruction_dx, target_dx) + F.l1_loss(reconstruction_dy, target_dy)


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    perceptual_model: PerceptualLoss | None,
    ssim_weight: float,
    perceptual_weight: float,
    edge_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Use pixel, structural, perceptual, and detail-preserving edge objectives."""
    l1_value = F.l1_loss(reconstruction, target)
    ssim_value = ssim_loss(reconstruction, target)
    edge_value = gradient_detail_loss(reconstruction, target)
    perceptual_value = torch.zeros((), device=target.device)
    if perceptual_model is not None and perceptual_weight > 0:
        perceptual_value = perceptual_model(reconstruction, target)

    total = (
        l1_value
        + ssim_weight * ssim_value
        + perceptual_weight * perceptual_value
        + edge_weight * edge_value
    )
    return total, {
        "l1": float(l1_value.detach().item()),
        "ssim_loss": float(ssim_value.detach().item()),
        "edge_loss": float(edge_value.detach().item()),
        "perceptual_loss": float(perceptual_value.detach().item()),
        "total": float(total.detach().item()),
    }


def evaluate(
    model: LatentGenesisCore,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    latent_channels: int,
    max_batches: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Evaluate deterministic packet decoding on a fixed validation loader."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    sample_count = 0
    contract: Dict[str, int] | None = None

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            latent = deterministic_latent(model, images)
            reconstruction = model.decoder(latent)
            if contract is None:
                contract = payload_contract(latent, latent_channels)

            total_psnr += float(psnr_from_normalized(reconstruction, images).sum().item())
            total_ssim += float((1.0 - ssim_loss(reconstruction, images)).item()) * images.shape[0]
            sample_count += images.shape[0]

    if sample_count == 0 or contract is None:
        raise RuntimeError("Validation loader produced no images.")
    return {
        "samples": float(sample_count),
        "mean_psnr_db": total_psnr / sample_count,
        "mean_ssim": total_ssim / sample_count,
    }, contract


def configure_stage(model: LatentGenesisCore, stage: str) -> None:
    """Set trainable modules for a safe decoder-first or full fine-tuning stage."""
    if stage == "decoder":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    output_path = Path(args.output_checkpoint).expanduser().resolve()

    if not base_path.is_file():
        raise FileNotFoundError(f"Base checkpoint does not exist: {base_path}")
    if base_path == output_path:
        raise ValueError("Output checkpoint must differ from the base checkpoint to protect rollback.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    set_seed(args.seed)
    checkpoint = load_checkpoint(base_path, device)
    checkpoint_latent_channels = int(checkpoint.get("latent_channels", args.latent_channels))
    checkpoint_decoder_blocks = decoder_bottleneck_blocks_from_checkpoint(checkpoint)
    if args.decoder_bottleneck_blocks < checkpoint_decoder_blocks:
        raise ValueError(
            f"Base checkpoint uses {checkpoint_decoder_blocks} decoder bottleneck blocks, "
            f"but --decoder_bottleneck_blocks={args.decoder_bottleneck_blocks} was requested."
        )
    if args.latent_channels is not None and args.latent_channels != checkpoint_latent_channels:
        raise ValueError(
            f"Checkpoint requires {checkpoint_latent_channels} latent channels, "
            f"but --latent_channels={args.latent_channels} was requested."
        )
    latent_channels = checkpoint_latent_channels

    model = LatentGenesisCore(
        latent_channels=latent_channels,
        decoder_bottleneck_blocks=args.decoder_bottleneck_blocks,
    ).to(device)
    load_compatible_weights(
        model,
        checkpoint,
        source_blocks=checkpoint_decoder_blocks,
        target_blocks=args.decoder_bottleneck_blocks,
    )
    configure_stage(model, args.stage)

    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size,
        root=args.data_root,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        use_hd=True,
        sample_limit=args.sample_limit,
    )

    base_metrics, contract = evaluate(
        model, testloader, device, latent_channels, args.val_batches
    )
    log.info(
        "[*] Baseline deterministic packet evaluation: %.2f dB PSNR | %.4f SSIM | %d bytes/image",
        base_metrics["mean_psnr_db"],
        base_metrics["mean_ssim"],
        contract["raw_payload_bytes_per_image"],
    )

    perceptual_model: PerceptualLoss | None = None
    if args.perceptual_weight > 0:
        perceptual_model = PerceptualLoss().to(device).eval()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_metrics = dict(base_metrics)
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.stage == "decoder":
            # Frozen encoder BatchNorm statistics must remain fixed as well.
            model.encoder.eval()
        if perceptual_model is not None:
            perceptual_model.eval()

        running_total = 0.0
        batch_count = 0
        progress = tqdm(trainloader, desc=f"Fine-tune {args.stage} {epoch}/{args.epochs}")
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if args.stage == "decoder":
                with torch.no_grad():
                    latent = deterministic_latent(model, images)
            else:
                latent = deterministic_latent(model, images)
            reconstruction = model.decoder(latent)
            loss, components = reconstruction_loss(
                reconstruction,
                images,
                perceptual_model,
                args.ssim_weight,
                args.perceptual_weight,
                args.edge_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=args.grad_clip)
            optimizer.step()

            running_total += components["total"]
            batch_count += 1
            progress.set_postfix(
                loss=f"{components['total']:.4f}",
                l1=f"{components['l1']:.4f}",
                ssim=f"{components['ssim_loss']:.4f}",
                edge=f"{components['edge_loss']:.4f}",
            )

        scheduler.step()
        val_metrics, epoch_contract = evaluate(
            model, testloader, device, latent_channels, args.val_batches
        )
        if epoch_contract != contract:
            raise RuntimeError("Payload contract changed during fine-tuning; refusing to save checkpoint.")
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_total / max(batch_count, 1),
            "mean_psnr_db": val_metrics["mean_psnr_db"],
            "mean_ssim": val_metrics["mean_ssim"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        is_better = (
            val_metrics["mean_psnr_db"] > best_metrics["mean_psnr_db"]
            or (
                val_metrics["mean_psnr_db"] == best_metrics["mean_psnr_db"]
                and val_metrics["mean_ssim"] > best_metrics["mean_ssim"]
            )
        )
        if is_better:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            log.info("[*] New best deterministic validation checkpoint at epoch %d.", epoch)
        log.info(
            "[*] Epoch %d/%d | train loss %.5f | PSNR %.2f dB | SSIM %.4f",
            epoch,
            args.epochs,
            epoch_record["train_loss"],
            epoch_record["mean_psnr_db"],
            epoch_record["mean_ssim"],
        )

    # Save the best deterministic validation state, not merely the final epoch.
    model.load_state_dict(best_state_dict, strict=True)
    final_metrics, final_contract = evaluate(
        model, testloader, device, latent_channels, args.val_batches
    )
    if final_contract != contract:
        raise RuntimeError("Final payload contract changed during fine-tuning; refusing to save checkpoint.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_version": (
            "universal-genesis-deep-decoder-v2"
            if args.decoder_bottleneck_blocks > 4
            else "universal-genesis-ft-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "stage": args.stage,
        "latent_channels": latent_channels,
        "decoder_bottleneck_blocks": args.decoder_bottleneck_blocks,
        "base_decoder_bottleneck_blocks": checkpoint_decoder_blocks,
        "payload_contract": contract,
        "baseline_metrics": base_metrics,
        "final_metrics": final_metrics,
        "best_epoch": best_epoch,
        "selected_by": "highest mean deterministic validation PSNR; SSIM breaks PSNR ties",
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "ssim_weight": args.ssim_weight,
            "perceptual_weight": args.perceptual_weight,
            "edge_weight": args.edge_weight,
            "sample_limit": args.sample_limit,
            "seed": args.seed,
            "validation_batches": args.val_batches,
            "decoder_bottleneck_blocks": args.decoder_bottleneck_blocks,
        },
        "history": history,
    }
    save_payload = {
        "model_state_dict": model.state_dict(),
        "latent_channels": latent_channels,
        "fine_tuning": metadata,
    }
    torch.save(save_payload, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log.info("[*] Fine-tuned checkpoint saved to %s", output_path)
    log.info("[*] Metrics and payload-contract report saved to %s", metrics_path)
    log.info(
        "[*] Validation delta: %+0.2f dB PSNR | %+0.4f SSIM | packet remains %d bytes/image",
        final_metrics["mean_psnr_db"] - base_metrics["mean_psnr_db"],
        final_metrics["mean_ssim"] - base_metrics["mean_ssim"],
        contract["raw_payload_bytes_per_image"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Universal Genesis checkpoint on its deterministic 8-bit packet path."
    )
    parser.add_argument("--base_checkpoint", required=True, help="Existing .pth checkpoint to preserve and fine-tune from.")
    parser.add_argument("--output_checkpoint", required=True, help="New .pth checkpoint path; must not equal --base_checkpoint.")
    parser.add_argument("--stage", choices=("decoder", "full"), default="decoder", help="decoder is the safe first stage; full unfreezes the encoder after decoder tuning.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ssim_weight", type=float, default=0.5)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--edge_weight", type=float, default=0.0, help="Detail-preserving edge-loss weight; use 0.05 for deep-decoder v2 training.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional cap on STL-10 unlabeled training images; omit for all 100,000 images.")
    parser.add_argument("--val_batches", type=int, default=25, help="Fixed number of STL-10 validation batches; 0 evaluates all batches.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--latent_channels", type=int, default=None, help="Optional safety check; otherwise inferred from checkpoint metadata.")
    parser.add_argument(
        "--decoder_bottleneck_blocks",
        type=int,
        default=4,
        help="Total 256-channel decoder bottleneck residual blocks. Use 14 for the approximately 80 MB deep-decoder v2.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
