# Paragram Engine — Colab Fine-Tuning Guide

This guide fine-tunes the existing **Universal Genesis** checkpoint to improve reconstruction quality while retaining its current transmission contract.

> **Packet contract preserved:** Each 256 × 256 image is represented by a `16 × 16 × 16` latent map. Each value is 8-bit quantized, so the raw latent payload remains **4,096 values / 4,096 bytes per image** before the transport container, headers, and metadata are added.

The base checkpoint is never overwritten. Every fine-tuning stage writes a separately versioned `.pth` file and a matching `.metrics.json` report.

---

## 1. Set up a GPU runtime

In Google Colab, choose **Runtime → Change runtime type → T4 GPU**, then run:

```bash
!nvidia-smi
```

---

## 2. Clone and install

```bash
!git clone https://github.com/ethcocoder/paragram-engin-v1.git
%cd paragram-engin-v1
!pip install -r requirements.txt
```

The initial perceptual-loss run may download the pre-trained VGG-16 weights and the STL-10 dataset. Allow those downloads to complete before judging the fine-tune.

---

## 3. Confirm the base checkpoint

The fine-tuning script refuses to overwrite the base model. Confirm it exists first:

```bash
!ls -lh checkpoints/universal_genesis_core.pth
!sha256sum checkpoints/universal_genesis_core.pth
```

The current base checkpoint should have this SHA-256 value:

```text
775d691f8cb739f1fe2c20beb726fabf5f868b7c7deb675e98bdc8dcdc5d506e
```

---

## 4. Stage 1 — recommended decoder-only fine-tune

This is the safe quality-improvement step. It freezes the encoder and fine-tunes the decoder to reconstruct from the **same deterministic, 8-bit quantized latent** that is used by `demo_hd.py` during transmission.

```bash
!python src/finetune_deterministic.py \
  --base_checkpoint checkpoints/universal_genesis_core.pth \
  --output_checkpoint checkpoints/universal_genesis_core_ft_decoder_v1.pth \
  --stage decoder \
  --epochs 12 \
  --batch_size 16 \
  --lr 1e-4 \
  --sample_limit 30000 \
  --val_batches 25
```

The script will print the baseline and fine-tuned validation PSNR/SSIM on the same STL-10 validation batches. It saves the best validation epoch rather than blindly saving the final epoch.

### Stage 1 output files

| File | Purpose |
|---|---|
| `checkpoints/universal_genesis_core_ft_decoder_v1.pth` | Fine-tuned checkpoint compatible with the existing demo loader |
| `checkpoints/universal_genesis_core_ft_decoder_v1.metrics.json` | Base-versus-final metrics, payload contract, seed, and training history |

---

## 5. Inspect the validation report

```bash
!cat checkpoints/universal_genesis_core_ft_decoder_v1.metrics.json
```

Use the values in `baseline_metrics` and `final_metrics` to decide whether to keep the checkpoint.

| Gate | Keep the Stage 1 checkpoint only if… |
|---|---|
| Payload | `raw_payload_bytes_per_image` remains `4096` |
| Quality | `final_metrics.mean_psnr_db` is higher than the baseline on the same validation set |
| Structure | `final_metrics.mean_ssim` does not fall materially below the baseline |
| Compatibility | The existing demo loads the new checkpoint without an error |

> Do not compare PSNR values from unrelated random samples. The script's base-versus-final validation values are the meaningful quality comparison because they use the same held-out images.

---

## 6. Test the fine-tuned checkpoint on fresh images

Run the existing packet-decoding demo with the new checkpoint:

```bash
!python src/demo_hd.py \
  --model_path checkpoints/universal_genesis_core_ft_decoder_v1.pth \
  --latent_channels 16 \
  --random
```

This downloads four fresh images, encodes their means, quantizes each latent to 8-bit values, reconstructs them, and writes:

```text
universal_hd_result.png
```

Display the result in Colab:

```python
from IPython.display import Image, display
display(Image(filename="universal_hd_result.png"))
```

---

## 7. Optional Stage 2 — full-model fine-tune

Run this **only if Stage 1 improves quality but still leaves substantial blur**. Stage 2 adjusts both encoder and decoder while retaining the identical `16 × 16 × 16`, 8-bit latent contract. It uses a lower learning rate because changing the encoder is less conservative.

```bash
!python src/finetune_deterministic.py \
  --base_checkpoint checkpoints/universal_genesis_core_ft_decoder_v1.pth \
  --output_checkpoint checkpoints/universal_genesis_core_ft_v1.pth \
  --stage full \
  --epochs 8 \
  --batch_size 16 \
  --lr 2e-5 \
  --sample_limit 30000 \
  --val_batches 25
```

Then evaluate it through the same demo command:

```bash
!python src/demo_hd.py \
  --model_path checkpoints/universal_genesis_core_ft_v1.pth \
  --latent_channels 16 \
  --random
```

---

## 8. Deployment requirement

A receiver must use the **same checkpoint version** that was used by the sender. The image packet retains the same shape and raw payload size, but the learned meaning of those 4,096 values changes as the encoder and/or decoder is fine-tuned.

For the decoder-only Stage 1 checkpoint, distribute the new decoder checkpoint to the receiver. For the full Stage 2 checkpoint, distribute the **same complete new checkpoint to both sender and receiver**.

```bash
# Example: retain the original model and distribute a versioned fine-tuned model.
!cp checkpoints/universal_genesis_core_ft_v1.pth peer_sender/
!cp checkpoints/universal_genesis_core_ft_v1.pth peer_receiver/
```

---

## 9. Rollback

Rollback does not require retraining. Use the untouched original checkpoint:

```bash
!python src/demo_hd.py \
  --model_path checkpoints/universal_genesis_core.pth \
  --latent_channels 16 \
  --random
```

---

## What this workflow changes—and what it does not

| Item | Result |
|---|---|
| Latent channels | Unchanged: 16 |
| Latent spatial size for 256 × 256 input | Unchanged: 16 × 16 |
| Quantization | Unchanged: 8-bit `[-1, 1]` quantization |
| Raw latent payload | Unchanged: 4,096 bytes per image before container overhead |
| Fine-tuned weights | Updated in new, versioned checkpoints only |
| Original checkpoint | Preserved for rollback |
| Reported quality | Measured against the same held-out validation set before and after fine-tuning |

The result is a controlled test of whether the same low-bandwidth image representation can produce visibly sharper, more structurally faithful reconstructions.
