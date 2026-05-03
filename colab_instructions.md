# 🌌 Paradox Elite: 3-Stage Progressive Training
================================================
**VAE-GAN → Fine-Tune → RL from Mistakes** — each stage is tested before proceeding.

---

## Step 1: Environment Setup
```bash
!git clone --single-branch --branch last-elite https://github.com/ethcocoder/ai-engin.git ai-engin
!pip install psutil torch torchvision matplotlib tqdm
%cd ai-engin
```

---

## ⚡ STANDARD TEST (run this first — always)
**5 epochs · 5000 samples · batch 12.**
Enough data for real learning. Fast enough to catch problems before full training.
If a stage **passes** here, the full run will be even better.

```bash
# Stage 1 — 5 epochs, 5000 samples
!python src/train.py --stage 1 --epochs 5 --batch_size 12 --sample_limit 5000
!python src/test_stages.py --stage 1 --model_path checkpoints/universal_genesis_core.pth

# Stage 2 — 5 epochs, 5000 samples
!python src/train.py --stage 2 --epochs 5 --batch_size 12 --sample_limit 5000 --resume checkpoints/universal_genesis_core.pth
!python src/test_stages.py --stage 2 --model_path checkpoints/stage2_genesis_core.pth

# Stage 3 — 5 epochs, 5000 samples
!python src/train.py --stage 3 --epochs 5 --batch_size 12 --sample_limit 5000 --resume checkpoints/stage2_genesis_core.pth
!python src/test_stages.py --stage 3 --model_path checkpoints/stage3_rl_genesis_core.pth

# Full comparison dashboard
!python src/test_stages.py --compare
```
> ✅ If all 3 stages pass — scale up epochs for the full elite run.
> ❌ If a stage fails — fix it here cheaply before investing more GPU time.

---

## 📐 Tunable Parameters

| Parameter | Standard Test ✅ | Full Elite 🏆 |
|-----------|-----------------|---------------|
| `--epochs` S1 | **5** | 100 |
| `--epochs` S2 | **5** | 30 |
| `--epochs` S3 | **5** | 20 |
| `--sample_limit` | **5000** | 10000 |
| `--batch_size` | **12** | 12 |

> 💡 **Standard Test** is not a throwaway — it trains real weights on real data.
> A stage that passes at 5 epochs / 5000 samples will always improve with more.
> On a **T4 GPU**: Standard Test ≈ 20–30 min total. Full Elite ≈ 3–4 hours.

---

## ─── STAGE 1: Monster VAE-GAN 🔥 ───────────────────────────────────────────

### Train
```bash
!python src/train.py \
  --stage 1 \
  --epochs 100 \
  --batch_size 12 \
  --sample_limit 10000 \
  --latent_channels 16
```
**Output:** `checkpoints/universal_genesis_core.pth`

### Test Stage 1
```bash
!python src/test_stages.py --stage 1 --model_path checkpoints/universal_genesis_core.pth
```
**Output:** `stage1_report_card.png`

| Metric | Must Pass | Failing? |
|--------|-----------|----------|
| PSNR   | ≥ 14.0 dB | Add `--epochs 50` more |
| SSIM   | ≥ 0.35    | Increase `--sample_limit` |

---

## ─── STAGE 2: Fine-Tune ✨ ──────────────────────────────────────────────────

### Train
```bash
!python src/train.py \
  --stage 2 \
  --epochs 30 \
  --sample_limit 10000 \
  --resume checkpoints/universal_genesis_core.pth
```
**Output:** `checkpoints/stage2_genesis_core.pth`

### Test Stage 2
```bash
!python src/test_stages.py --stage 2 --model_path checkpoints/stage2_genesis_core.pth
```
**Output:** `stage2_report_card.png`

| Metric | Must Pass | Failing? |
|--------|-----------|----------|
| PSNR   | ≥ 22.0 dB | Add `--epochs 20` more |
| SSIM   | ≥ 0.65    | Increase `--sample_limit` |

---

## ─── STAGE 3: RL from Mistakes 🧠 ──────────────────────────────────────────

### Train
```bash
!python src/train.py \
  --stage 3 \
  --epochs 20 \
  --sample_limit 10000 \
  --resume checkpoints/stage2_genesis_core.pth
```
**Output:** `checkpoints/stage3_rl_genesis_core.pth`

### Test Stage 3
Tests on **hard high-frequency images** — the ones the model historically struggles with.
```bash
!python src/test_stages.py --stage 3 --model_path checkpoints/stage3_rl_genesis_core.pth
```
**Output:** `stage3_report_card.png`

| Metric | Must Pass | Failing? |
|--------|-----------|----------|
| PSNR   | ≥ 26.0 dB | Re-run with `--epochs 40 --resume checkpoints/stage3_rl_genesis_core.pth` |
| SSIM   | ≥ 0.78    | Increase `--sample_limit` |

---

## ─── FULL COMPARISON DASHBOARD 🏆 ──────────────────────────────────────────

Runs all 3 models on the **same 4 images** — side-by-side bar chart of PSNR per stage.
```bash
!python src/test_stages.py --compare \
  --s1 checkpoints/universal_genesis_core.pth \
  --s2 checkpoints/stage2_genesis_core.pth \
  --s3 checkpoints/stage3_rl_genesis_core.pth
```
**Output:** `stage_comparison_dashboard.png`

---

## ─── FINAL DEPLOYMENT TEST 🛡️ ───────────────────────────────────────────────

### Batch Report
```bash
!python src/demo_hd.py --batch --model_path checkpoints/stage3_rl_genesis_core.pth
```

### Single Image .paradox Codec
```bash
!python src/demo_hd.py --size 1024 --image_path test_local/johan.png \
        --model_path checkpoints/stage3_rl_genesis_core.pth
```

---

## 💎 Architecture Summary

| Stage | Name | LR | Loss | Default Epochs |
|-------|------|----|------|----------------|
| 1 | Monster VAE-GAN | 2e-4 | MSE×10 + SSIM×0.7 + QWM×0.1 + GAN×0.01 | 100 |
| 2 | Fine-Tune | 2e-5 | MSE×10 + SSIM×1.0 + QWM×0.3 | 30 |
| 3 | RL Mistakes | 1e-5 | Reward-Weighted MSE + SSIM×0.7 + QWM×0.2 | 20 |

---

## 📦 Data Partitioning Strategy

All stages use **STL-10 Unlabeled** (100,000 images, 96×96px).
Each stage gets its **own fixed, non-overlapping slice** — so fine-tuning and RL
always train on data the model has **never seen before**.

| Stage | Data Slice | Size | Seed | Purpose |
|-------|-----------|------|------|---------|
| 1 — Monster | indices `[0 → 50,000)` | 50k images | 43 | Broad visual grammar |
| 2 — Fine-Tune | indices `[50,000 → 75,000)` | 25k images | 44 | Fresh domain, quality polish |
| 3 — RL Mistakes | indices `[75,000 → 100,000)` | 25k images | 45 | Held-out challenge set |
| Eval / Test | STL-10 `test` split | 800 images | — | Never used in any training |

> 💡 **Why this matters:**
> If all stages train on the same images, fine-tuning just memorizes what Stage 1 already saw.
> With separate partitions, each stage is forced to **generalize** to new visual patterns.
> The RL stage trains on images the model has genuinely **never encountered** — making
> the reward signal honest: improvements are real, not just re-memorization.

> 🔒 All slices use a **fixed deterministic seed** (`seed = 42 + stage`).
> Every run uses the **exact same images** for each stage — fully reproducible.

---

## 🧠 RL Stage — How the Reward Works

```
Reward     = per_image_PSNR − running_baseline_PSNR  (EMA, α=0.05)
Advantage  = (Reward − mean) / std                   (normalized per batch)
Weight     = clamp(1 − Advantage, 0.3, 4.0)

→ Low PSNR image  → Weight up to 4.0 → strong gradient correction
→ High PSNR image → Weight down to 0.3 → protect good reconstructions
```
**MistakeBuffer:** Images with PSNR < 22 dB stored (max 256). Replayed at 40% probability.

---

## 📊 Report Cards Explained

Each `stageN_report_card.png` has **3 rows per test image:**
- **Row 1** — Original input
- **Row 2** — Reconstruction + PSNR / SSIM per image
- **Row 3** — 🔥 Error heat map (bright = high error zones, dark = accurate zones)

**Expected final result:** PSNR 28–34 dB | SSIM > 0.85 | BPP ~0.8 | ~29× compression
