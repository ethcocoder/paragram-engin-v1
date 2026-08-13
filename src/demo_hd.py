"""
demo_hd.py — Paradox Aether Mesh: Universal HD Demo
===================================================
Demonstrates the Universal Engine's ability to compress and 
reconstruct ANY random HD image from the internet.
"""

import os
import torch
import matplotlib.pyplot as plt
import sys
import random
import urllib.request
from pathlib import Path
from model import LatentGenesisCore
from hd_data import CustomHDDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

# Advanced Pathing Protocol
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

def unnorm(img):
    return torch.clamp(img * 0.5 + 0.5, 0, 1)

def psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0: return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()

def download_random_hd(image_dir, count=4):
    """Pulls fresh random HD images from the web to test generalization."""
    if not os.path.exists(image_dir): os.makedirs(image_dir)
    print(f"[*] Pulling {count} fresh random HD samples from the Aether Mesh...")
    for i in range(count):
        seed = random.randint(0, 1000000)
        url = f"https://picsum.photos/seed/{seed}/1024/1024"
        file_path = os.path.join(image_dir, f"random_test_{i}.jpg")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as res, open(file_path, 'wb') as f:
            f.write(res.read())

def run_hd_simulation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_path = Path(args.model_path).name
    print(f"[*] Paradox Universal Demo: Testing '{log_path}' on {device}")
    
    # 1. Load Universal Model.  Deep-decoder checkpoints include their
    # architecture depth; legacy checkpoints default to the original four
    # bottleneck blocks.
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    metadata = checkpoint.get('fine_tuning', {})
    if not isinstance(metadata, dict):
        metadata = {}
    checkpoint_blocks = int(
        checkpoint.get('decoder_bottleneck_blocks', metadata.get('decoder_bottleneck_blocks', 4))
    )
    checkpoint_channels = int(checkpoint.get('latent_channels', args.latent_channels))
    if checkpoint_channels != args.latent_channels:
        raise ValueError(
            f"Checkpoint requires {checkpoint_channels} latent channels, "
            f"but --latent_channels={args.latent_channels} was requested."
        )
    model = LatentGenesisCore(
        latent_channels=checkpoint_channels,
        decoder_bottleneck_blocks=checkpoint_blocks,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    print(f"[*] Decoder bottleneck blocks: {checkpoint_blocks}")

    # 2. Fresh Patterns for Generalization Test
    if args.random:
        download_random_hd(args.image_dir, count=4)

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomHDDataset(args.image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    images, _ = next(iter(loader))
    images = images.to(device)

    # 3. Compression Logic (16x Spatial + 8-bit Q)
    original_bytes = 256 * 256 * 3
    with torch.no_grad():
        mu, _ = model.encoder(images)
        z_q = torch.round(torch.clamp(mu, -1, 1) * 127.5) / 127.5
        payload_bytes = z_q.nelement() * 1
        reconstructed = model.decoder(z_q)

    compression_ratio = original_bytes / (payload_bytes / images.shape[0])
    
    print("\n--- UNIVERSAL HD RESULTS ---")
    print(f"[-] Reduction Factor: {compression_ratio:.1f}X")
    
    psnr_scores = [psnr(unnorm(images[i].cpu()), unnorm(reconstructed[i].cpu())) for i in range(len(images))]
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"[*] Average Generalization PSNR: {avg_psnr:.2f} dB")
    print("----------------------------\n")

    fig, axes = plt.subplots(2, len(images), figsize=(20, 10), squeeze=False)
    fig.suptitle(f"Universal HD Engine: {compression_ratio:.1f}x Reduction | Generalization PSNR: {avg_psnr:.2f}dB", fontsize=20)

    for i in range(len(images)):
        axes[0, i].imshow(unnorm(images[i].cpu()).permute(1, 2, 0))
        axes[0, i].set_title("Sender (Global Pattern)")
        axes[0, i].axis('off')
        axes[1, i].imshow(unnorm(reconstructed[i].cpu()).permute(1, 2, 0))
        axes[1, i].set_title(f"Receiver (Neural Synth)\nPSNR: {psnr_scores[i]:.1f}dB")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('universal_hd_result.png', dpi=300)
    print("[*] Output saved to 'universal_hd_result.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/universal_genesis_core.pth')
    parser.add_argument('--image_dir', type=str, default='hd_images')
    parser.add_argument('--latent_channels', type=int, default=16)
    parser.add_argument('--random', action='store_true', help="Pull fresh random images for a true generalization test")
    args = parser.parse_args()
    run_hd_simulation(args)
