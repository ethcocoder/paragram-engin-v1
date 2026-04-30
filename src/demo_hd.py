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

import gzip

def encode_to_paradox(latent_tensor, filepath, metadata):
    # Quantize to int16 for storage
    latent_np = (latent_tensor.cpu().numpy() * 32767).astype(np.int16)
    
    # Binary Payload: [Channels (4b), H (4b), W (4b), LatentData (Variable)]
    header = np.array([metadata['channels'], metadata['h'], metadata['w']], dtype=np.int32).tobytes()
    payload = latent_np.tobytes()
    
    with gzip.open(filepath, 'wb') as f:
        f.write(b'PDX-v2!') # Paradox v2 (Compressed)
        f.write(header)
        f.write(payload)
        
    return os.path.getsize(filepath)

def decode_from_paradox(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic = f.read(7)
        if magic != b'PDX-v2!': 
            # Fallback for old version or error
            if magic[:8] == b'PARADOX!':
                raise ValueError("Old .paradox v1 detected. Use legacy decoder.")
            raise ValueError("Invalid .paradox file or corruption.")
            
        meta = np.frombuffer(f.read(12), dtype=np.int32)
        latent_np = np.frombuffer(f.read(), dtype=np.int16)
        
    latent_tensor = torch.from_numpy(latent_np.astype(np.float32) / 32767.0)
    latent_tensor = latent_tensor.view(1, meta[0], meta[1], meta[2])
    return latent_tensor

# ── Elite Single Demo ────────────────────────────────────────────────────────

def run_elite_demo(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = get_hardware_info()
    print(f"\n{'='*70}\n[*] PARADOX ELITE ENGINE | SINGLE IMAGE AUDIT\n{'='*70}")
    print(f"[HW] GPU: {hw['GPU']} | VRAM: {hw['VRAM']}")

    # 1. Load Model
    model = LatentGenesisCore(latent_channels=args.latent_channels, device=str(device)).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Prepare Sample
    if args.image_path and os.path.exists(args.image_path):
        img_raw = Image.open(args.image_path).convert('RGB')
    else:
        seed = random.randint(0, 1000000)
        url = f"https://picsum.photos/seed/{seed}/1024/1024"
        print(f"[*] Fetching 1024x1024 Master Sample (Lorem Picsum)...")
        urllib.request.urlretrieve(url, "master_sample.jpg")
        img_raw = Image.open("master_sample.jpg")

    # 128px Safety Protocol: Enforce minimum 256px to prevent latent collapse
    safe_size = max(args.size, 256)
    if args.size < 256:
        print(f"[!] Safety Net Activated: Upscaling from {args.size}px to 256px.")

    # 128px Safety Protocol: Enforce minimum 256px to prevent latent collapse
    safe_size = max(args.size, 256)
    if args.size < 256:
        print(f"[!] Safety Net Activated: Upscaling from {args.size}px to 256px.")

    transform = transforms.Compose([
        transforms.Resize((safe_size, safe_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(img_raw).unsqueeze(0).to(device)

    # 3. Encoding Phase
    t0 = time.time()
    with torch.no_grad():
        mu, _ = model.encoder(input_tensor)
        z_q = model.quantizer(mu)
    encode_time = (time.time() - t0) * 1000
    
    meta = {'channels': args.latent_channels, 'h': z_q.shape[2], 'w': z_q.shape[3]}
    paradox_size = encode_to_paradox(z_q, "output.paradox", meta)
    
    # 4. Decoding Phase (From .paradox file)
    t1 = time.time()
    with torch.no_grad():
        latent_restored = decode_from_paradox("output.paradox").to(device)
        reconstructed = model.decoder(latent_restored)
    decode_time = (time.time() - t1) * 1000

    # 5. Elite Metric Calculations
    # PSNR
    mse = torch.mean((input_tensor - reconstructed) ** 2).item()
    psnr_val = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
    
    # Simple SSIM Approximation (Structural Correlation)
    from train import ssim_loss
    ssim_val = 1.0 - ssim_loss(input_tensor, reconstructed).item()
    
    # BPP (Bits Per Pixel)
    total_pixels = args.size * args.size
    bpp = (paradox_size * 8) / total_pixels
    profit = (total_pixels * 3 * 8) / (paradox_size * 8) # Uncompressed vs Paradox

    print(f"\n[METRICS]")
    print(f" -> PSNR: {psnr_val:.2f} dB")
    print(f" -> SSIM: {ssim_val:.4f}")
    print(f" -> MSE:  {mse:.6f}")
    print(f" -> BPP:  {bpp:.4f} bits/pixel")
    print(f" -> PROFIT: {profit:.1f}X (Compression)")

    print(f"\n[LATENCY]")
    print(f" -> Encoder: {encode_time:.1f}ms")
    print(f" -> Decoder: {decode_time:.1f}ms")
    print(f" -> TOTAL:   {encode_time + decode_time:.1f}ms")

    # 6. High-Resolution Output
    # Save Standalone Reconstructed Image (High Quality PNG)
    recon_img = reconstructed[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    recon_img = (recon_img.clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(recon_img).save('paradox_high_res.png')
    print(f"\n[*] SUCCESS: Standalone HD result saved to 'paradox_high_res.png'")

    # Save Comparison Dashboard
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor[0].cpu().permute(1,2,0)*0.5+0.5)
    plt.title(f"Original ({args.size}px)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(recon_img)
    plt.title(f"Paradox Reconstruction\nPSNR: {psnr_val:.2f}dB | BPP: {bpp:.3f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('paradox_elite_dashboard.png')
    print(f"[*] Comparison dashboard saved to 'paradox_elite_dashboard.png'")

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

        # 128px Safety Protocol
        safe_size = max(size, 256)
        transform = transforms.Compose([transforms.Resize((safe_size, safe_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
