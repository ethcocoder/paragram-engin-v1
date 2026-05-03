"""
train_hd.py — Paradox Genesis HD Training Engine
================================================
Trains the Genesis Core specifically for High Definition (256+) images.
Utilises the upgraded 4-stage architecture and Perceptual VGG Loss to 
capture high-frequency detail and sharp textures.
"""

import os
import argparse
import logging
import torch
import torch.optim as optim
from pathlib import Path

from hd_data import get_hd_dataloaders
from model import LatentGenesisCore
from train import compression_loss, QuantumWaveletLoss

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

def train_hd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("[*] Paradox HD Engine: Initializing on %s", device)
    
    # 1. Load Data
    loader = get_hd_dataloaders(image_dir=args.image_dir, batch_size=args.batch_size)
    if loader is None: 
        log.error("No data found. Aborting.")
        return

    # 2. Initialize Model (4-stage HD capable)
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    
    # Initialize HD Secret Sauce: Quantum Wavelet Matching
    qwm_model = QuantumWaveletLoss(channels=3).to(device)
    
    # AdamW with weight decay for better edge generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    log.info("\n[*] Executing ELITE HD Synthesis. Capturing high-frequency textures...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # Lower KLD pressure (0.001) lets the model focus on ELITE quality
        kld_weight = min(1.0, epoch / max(1, args.epochs // 2)) * 0.001
        
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            outputs, mu, logvar = model(images)
            
            # Use the integrated HD-Ready loss function
            loss, l1_l, ssim_l, qwm_l, kld_l = compression_loss(
                outputs, images, mu, logvar, kld_weight, qwm_model
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(loader)
        log.info(f"Epoch [{epoch+1}/{args.epochs}] -> HD Genesis Error: {avg_loss:.4f} | SSIM: {ssim_l.item():.4f} | QWM: {qwm_l.item():.4f}")

    log.info("[*] HD Genesis Complete. Saving deployment architecture.")
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_channels': args.latent_channels,
        'epoch': args.epochs,
    }, os.path.join(args.checkpoint_dir, 'hd_genesis_core.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox HD Engine Training")
    parser.add_argument('--image_dir', type=str, default='hd_images')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500) # Increased for Elite Sharpness
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_channels', type=int, default=16) # 16 for Elite HD
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    train_hd(args)
