"""Verify a Genesis checkpoint on freshly downloaded random images.

The verifier runs the same deployed path as ``demo_hd.py``:
image -> encoder mean -> 8-bit quantization -> decoder -> reconstruction.
It reports per-image and aggregate PSNR while validating that the latent packet
shape remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from hd_data import CustomHDDataset
from model import LatentGenesisCore


IMAGE_SIZE = 256
QUANTIZATION_LEVELS = 127.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_random_images(image_dir: Path, count: int) -> None:
    """Download exactly ``count`` fresh 1024px images with reproducible filenames."""
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        seed = random.randint(0, 1_000_000_000)
        url = f"https://picsum.photos/seed/{seed}/1024/1024"
        target = image_dir / f"random_{index:02d}.jpg"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp(images * 0.5 + 0.5, 0.0, 1.0)


def psnr_per_image(reference: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    reference = unnormalize(reference)
    reconstruction = unnormalize(reconstruction)
    mse = torch.mean((reference - reconstruction) ** 2, dim=(1, 2, 3))
    return 20.0 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-12)))


def run(args: argparse.Namespace) -> None:
    if args.num_images < 1:
        raise ValueError("--num_images must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.model_path).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_path = Path(args.output_image).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if image_dir.exists():
        shutil.rmtree(image_dir)
    print(f"[*] Downloading {args.num_images} fresh random verification images...")
    download_random_images(image_dir, args.num_images)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model_state_dict'.")
    metadata = checkpoint.get("fine_tuning", {})
    if not isinstance(metadata, dict):
        metadata = {}
    checkpoint_blocks = int(
        checkpoint.get("decoder_bottleneck_blocks", metadata.get("decoder_bottleneck_blocks", 4))
    )
    checkpoint_channels = int(checkpoint.get("latent_channels", args.latent_channels))
    if checkpoint_channels != args.latent_channels:
        raise ValueError(
            f"Checkpoint requires {checkpoint_channels} latent channels, "
            f"but {args.latent_channels} was requested."
        )

    model = LatentGenesisCore(
        latent_channels=args.latent_channels,
        decoder_bottleneck_blocks=checkpoint_blocks,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CustomHDDataset(str(image_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=args.num_images, shuffle=False)
    images, _ = next(iter(loader))
    if images.shape[0] != args.num_images:
        raise RuntimeError(f"Expected {args.num_images} downloaded images, received {images.shape[0]}.")
    images = images.to(device)

    with torch.no_grad():
        mu, _ = model.encoder(images)
        latent = torch.round(torch.clamp(mu, -1.0, 1.0) * QUANTIZATION_LEVELS) / QUANTIZATION_LEVELS
        reconstruction = model.decoder(latent)

    _, channels, height, width = latent.shape
    if (channels, height, width) != (args.latent_channels, 16, 16):
        raise RuntimeError(
            "Packet contract changed: expected "
            f"({args.latent_channels}, 16, 16), received ({channels}, {height}, {width})."
        )
    payload_bytes_per_image = channels * height * width
    reduction_factor = (IMAGE_SIZE * IMAGE_SIZE * 3) / payload_bytes_per_image
    psnr_values = psnr_per_image(images, reconstruction).cpu().tolist()

    print("\n--- RANDOM CHECKPOINT VERIFICATION ---")
    print(f"[*] Checkpoint: {checkpoint_path.name}")
    print(f"[*] Device: {device}")
    print(f"[*] Images tested: {len(psnr_values)}")
    print(f"[*] Latent packet: {channels}x{height}x{width} @ 8-bit = {payload_bytes_per_image} raw bytes/image")
    print(f"[*] Reduction factor: {reduction_factor:.1f}X")
    print(f"[*] Mean PSNR: {sum(psnr_values) / len(psnr_values):.2f} dB")
    print(f"[*] Minimum PSNR: {min(psnr_values):.2f} dB")
    print(f"[*] Maximum PSNR: {max(psnr_values):.2f} dB")
    print("--------------------------------------\n")

    figure, axes = plt.subplots(2, args.num_images, figsize=(3.0 * args.num_images, 6.0), squeeze=False)
    figure.suptitle(
        f"Random Checkpoint Verification: {reduction_factor:.1f}x | "
        f"Mean PSNR: {sum(psnr_values) / len(psnr_values):.2f} dB",
        fontsize=18,
    )
    display_source = unnormalize(images.cpu())
    display_reconstruction = unnormalize(reconstruction.cpu())
    for index in range(args.num_images):
        axes[0, index].imshow(display_source[index].permute(1, 2, 0))
        axes[0, index].set_title(f"Source {index + 1}")
        axes[0, index].axis("off")
        axes[1, index].imshow(display_reconstruction[index].permute(1, 2, 0))
        axes[1, index].set_title(f"PSNR {psnr_values[index]:.1f} dB")
        axes[1, index].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "image_source": "https://picsum.photos random 1024x1024 images",
        "images_tested": args.num_images,
        "decoder_bottleneck_blocks": checkpoint_blocks,
        "packet_contract": {
            "input_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "latent_shape_per_image": [channels, height, width],
            "quantization_bits": 8,
            "raw_payload_bytes_per_image": payload_bytes_per_image,
            "reduction_factor_against_raw_rgb": reduction_factor,
        },
        "psnr_db": {
            "per_image": psnr_values,
            "mean": sum(psnr_values) / len(psnr_values),
            "minimum": min(psnr_values),
            "maximum": max(psnr_values),
        },
        "output_image": str(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[*] Comparison image: {output_path}")
    print(f"[*] JSON report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a checkpoint on fresh random images.")
    parser.add_argument(
        "--model_path",
        default="checkpoints/universal_genesis_core_ft_decoder_v1.pth",
        help="Fine-tuned checkpoint to verify.",
    )
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--image_dir", default="verification_random_images")
    parser.add_argument("--output_image", default="random_checkpoint_verification.png")
    parser.add_argument("--report_path", default="random_checkpoint_verification.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
