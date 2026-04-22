# Paradox v3: Sovereign Hybrid Enhancer — Master Training Guide

The Sovereign Hybrid Enhancer is the next evolution of the Paradox Engine. **Team A** (the frozen Genesis Core VAE) compresses any image into a 4-KB bottleneck, and **Team B** (the GAN + Diffusion hybrid) restores it to retina-grade fidelity. You train only Team B.

---

## Step 1: Initialize the Mesh
Clone the repository, switch to the v3-hybrid branch, and prepare the GPU runtime environment.

```bash
!git clone https://github.com/ethcocoder/paradoxnetwork.git
%cd paradoxnetwork/ai-engine-git
!git checkout v3-hybrid-enhancer
!pip install -q torch torchvision torchaudio tqdm matplotlib pillow
```

---

## Step 2: Load the Genesis Core (Team A)
Upload your pre-trained `universal_genesis_core.pth` to the session. This is the frozen 4-KB structural foundation that Team B will learn to enhance.

```bash
from google.colab import files
uploaded = files.upload()   # Select universal_genesis_core.pth

import shutil, os
os.makedirs("checkpoints", exist_ok=True)
shutil.move("universal_genesis_core.pth", "checkpoints/universal_genesis_core.pth")
print("[✓] Team A structural foundation ready.")
```

---

## Step 3: Choose Your Training Path

### Path A: Real-World Diversity (HD Photos)
Pull **200 real HD photos** from the web and generate augmented variants. This creates a high-entropy dataset for handling real-world complexity.

```bash
          !python src/generate_hd_samples.py --mode download --n 200 --res 512 --augments 2 --out ../hd_images
```
***Result**: 600 training images covering real-world diversity.

### Path B: Latent Manifold Synthesis (Offline)
Use the **QVS probability manifold** to synthesize 200 images directly from Team A's latent space. Fully autonomous — no internet required.

```bash
          !python src/generate_hd_samples.py --mode synth --n 200 --augments 1 --sender_path ../checkpoints/universal_genesis_core.pth --latent_channels 16 --out ../hd_images
```
***Result**: 200 AI-synthesized images for pure latent space training.

### Path C: Maximum Generalization (Recommended)
Combines real photos and AI-synthesized images. This is the "Gold Standard" for achieving sovereign-grade reconstruction.

```bash
          !python src/generate_hd_samples.py --mode both --n 300 --res 512 --augments 2 --sender_path ../checkpoints/universal_genesis_core.pth --latent_channels 16 --out ../hd_images
```
***Result**: ~1500 training files for absolute generalization capability.

---

## Step 4: Train the Sovereign Enhancer (Team B)
Launch the GAN + Diffusion hybrid training loop. This teaches Team B to hallucinate missing detail back into the 4-KB bottleneck.

```bash
          !python src/hybrid_enhancer.py --mode train --sender_path ../checkpoints/universal_genesis_core.pth --enhancer_path ../checkpoints/sovereign_hybrid_enhancer.pth --data_dir ../hd_images --latent_channels 16 --gan_nf 64 --gan_nb 12 --diff_base_ch 48 --epochs 30 --batch_size 4
```

---

## Step 5: The Sovereign Test (Inference Check)
Run the demo to prove the model can reconstruct sharp, retina-grade images from the latent memory.

```bash
          !python src/hybrid_enhancer.py --mode demo --sender_path ../checkpoints/universal_genesis_core.pth --enhancer_path ../checkpoints/sovereign_hybrid_enhancer.pth --latent_channels 16 --T_inf 4
```

---

## 💎 Sovereign Architecture Summary
| Feature | Logic | Advantage |
|---|---|---|
| **Genesis Core** | Frozen 4-KB Bottleneck | Zero-cost structural compression |
| **Sovereign GAN** | RAB × 12 + SEI Gate | Sharp textures and perfect edge detail |
| **Hybrid Diffusion** | Conditional DDIM U-Net | Manifold-correct color & contrast |
| **Alpha Fusion** | Learned α Blending | Perfect mix of GAN sharpness + Diffusion smoothness |
| **Isotropic Loss** | Weight 8.0 | Kills directional blur & motion streaking |

---

**Protocol Check**: For best results, use **Path C** to ensure the model learns both real-world textures and AI-synthesized latent patterns! 🌌🛡️
