# Paradox MONSTER Engine: Mobile & Consistency Upgrade Plan

This document outlines the strategic steps to transition the Paradox Engine from a heavy GPU research model into a highly consistent, mobile-friendly production engine.

## 1. The Novel Paradigm: Quantum Wavelet Matching (QWM)
You asked if we can invent a new type of learning. Yes. Traditional AI uses GANs or VGG networks (like our current `PerceptualLoss`), which are heavy, outdated, and cause the texture inconsistencies you see in the wood planks.

We will invent **Quantum Wavelet Matching (QWM)** (Frequency-Resonance Learning):
*   **Task 1.1: Frequency-Domain Loss:** Instead of evaluating pixels, we will use your existing `HaarWaveletTransform` to split the image into 4 quantum states (Low-Low, High-Low, Low-High, High-High). We will train the model to mathematically match the *frequencies* of the textures, not just the colors.
*   **Task 1.2: Discriminator Deactivation:** By mastering frequency matching, the Generator becomes so intelligent that we can actually *shrink or remove* the heavy `EliteDiscriminator`. This solves the texture collapse (wood planks will look perfect) and makes training extremely fast.

## 2. The Mobile-First Refactor (Hardware Efficiency)
Currently, the model uses standard heavy 2D Convolutions, which drain mobile batteries and cause latency.
*   **Task 2.1: Depthwise Separable Convolutions:** We will rewrite the `ResBlock` in `src/model.py`. By splitting convolutions into two lighter steps (depthwise then pointwise), we can reduce the mathematical operations (MACs) by up to **70%** without losing visual quality. This allows real-time decoding on mobile CPUs/NPUs.
*   **Task 2.2: ONNX Mobile Export:** Create an `export_mobile.py` script. Mobile phones do not run native PyTorch efficiently. We must export the `Decoder` to `.onnx` format so it can be deployed directly to iOS (CoreML) and Android (NNAPI).

## 3. The 128px "Safety Net"
*   **Task 3.1: Dynamic Upscaling:** We will update `demo_hd.py` and the future mobile entry points to automatically upscale images that fall below the 256px architectural minimum, preventing the "blindspot" failures seen in the batch report.

---
### Execution Strategy
If you approve this plan, I recommend we execute **Phase 1 (Consistency)** first to ensure the AI logic is perfect and the textures are fixed. Then, we execute **Phase 2 (Mobile Optimization)** to shrink the perfect model down for phones.
