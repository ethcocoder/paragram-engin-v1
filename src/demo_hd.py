import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import urllib.request
import psutil
from pathlib import Path
from model import LatentGenesisCore
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse

# ── Hardware Profiler ────────────────────────────────────────────────────────

def get_hardware_info():
    gpu_name = "N/A"
    vram_total = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
    return {
        "GPU": gpu_name,
        "VRAM": f"{vram_total:.1f} GB",
        "CPU": f"{cpu_usage}%",
        "RAM": f"{ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB"
    }

# ── .paradox Codec Logic ─────────────────────────────────────────────────────

def encode_to_paradox(latent_tensor, filepath, metadata):
    latent_np = (latent_tensor.cpu().numpy() * 32767).astype(np.int16)
    with open(filepath, 'wb') as f:
        f.write(b'PARADOX!')
        f.write(np.array([metadata['channels'], metadata['h'], metadata['w']], dtype=np.int32).tobytes())
        f.write(latent_np.tobytes())
    return os.path.getsize(filepath)

def decode_from_paradox(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic != b'PARADOX!': raise ValueError("Invalid .paradox file")
        meta = np.frombuffer(f.read(12), dtype=np.int32)
        latent_np = np.frombuffer(f.read(), dtype=np.int16)
    latent_tensor = torch.from_numpy(latent_np.astype(np.float32) / 32767.0)
    latent_tensor = latent_tensor.view(1, meta[0], meta[1], meta[2])
    return latent_tensor

# ── Elite Single Demo ────────────────────────────────────────────────────────

def run_elite_demo(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = get_hardware_info()
    print(f"\n{'='*60}\n[*] PARADOX ELITE DEMO | HW: {hw['GPU']} | VRAM: {hw['VRAM']}\n{'='*60}")

    model = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.image_path and os.path.exists(args.image_path):
        img = Image.open(args.image_path).convert('RGB')
    else:
        seed = random.randint(0, 1000000)
        url = f"https://picsum.photos/seed/{seed}/1024/1024"
        print(f"[*] Fetching random 1024x1024 Master Sample...")
        urllib.request.urlretrieve(url, "master_sample.jpg")
        img = Image.open("master_sample.jpg")

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        mu, _ = model.encoder(input_tensor)
        z_q = model.quantizer(mu)
    encode_time = (time.time() - start_time) * 1000
    
    meta = {'channels': args.latent_channels, 'h': z_q.shape[2], 'w': z_q.shape[3]}
    paradox_size = encode_to_paradox(z_q, "output.paradox", meta)
    
    start_time = time.time()
    with torch.no_grad():
        latent_restored = decode_from_paradox("output.paradox").to(device)
        reconstructed = model.decoder(latent_restored)
    decode_time = (time.time() - start_time) * 1000

    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((input_tensor - reconstructed) ** 2))).item()
    print(f"[*] PSNR: {psnr_val:.2f} dB | Latency: {encode_time+decode_time:.1f}ms | Profit: {(args.size*args.size*3)/paradox_size:.1f}X")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(input_tensor[0].cpu().permute(1,2,0)*0.5+0.5); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(reconstructed[0].cpu().permute(1,2,0)*0.5+0.5); plt.title("Paradox Reconstruction"); plt.axis('off')
    plt.savefig('elite_codec_result.png'); print("[*] Result saved to elite_codec_result.png")

# ── Elite Batch Engine ───────────────────────────────────────────────────────

def run_batch_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = get_hardware_info()
    
    model = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tasks = [
        ("High-Q Elite", 512, "random"),
        ("High-Q Elite", 512, "random"),
        ("High-Q Elite", 512, "random"),
        ("High-Q Elite", 512, "random"),
        ("Low-Q Pattern", 128, "random"),
        ("Low-Q Pattern", 128, "random"),
        ("User Local", 512, "test_local/johan.png")
    ]

    results = []
    print(f"\n{'='*60}\n[*] PARADOX BATCH TEST | HW: {hw['GPU']}\n{'='*60}")

    for i, (name, size, src) in enumerate(tasks):
        print(f"[*] Task {i+1}: {name}")
        if src == "random":
            url = f"https://picsum.photos/seed/{random.randint(0,9999)}/{size}/{size}"
            urllib.request.urlretrieve(url, f"temp_{i}.jpg"); img = Image.open(f"temp_{i}.jpg").convert('RGB')
        else:
            if os.path.exists(src): img = Image.open(src).convert('RGB')
            else: print(f"[!] {src} not found."); continue

        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = model.encoder(input_tensor); z_q = model.quantizer(mu); reconstructed = model.decoder(z_q)
        
        psnr_val = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((input_tensor - reconstructed) ** 2))).item()
        results.append({"name": name, "size": size, "psnr": psnr_val, 
                        "img_orig": input_tensor[0].cpu().permute(1,2,0)*0.5+0.5, 
                        "img_recon": reconstructed[0].cpu().permute(1,2,0)*0.5+0.5})

    fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 8))
    for i, res in enumerate(results):
        axes[0, i].imshow(res['img_orig']); axes[0, i].set_title(f"SENDER\n{res['size']}px"); axes[0, i].axis('off')
        axes[1, i].imshow(res['img_recon']); axes[1, i].set_title(f"RECEIVER\n{res['psnr']:.1f} dB"); axes[1, i].axis('off')
    plt.tight_layout(); plt.savefig('universal_batch_report.png'); print("[*] Batch report saved to universal_batch_report.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/universal_genesis_core.pth')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--latent_channels', type=int, default=16)
    args = parser.parse_args()
    if args.batch: run_batch_test(args)
    else: run_elite_demo(args)
