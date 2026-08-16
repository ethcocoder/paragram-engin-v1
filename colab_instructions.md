# Paragram Engine — Deep-Decoder v2 Colab Guide

This branch implements **Deep-Decoder v2**, an approximately 80 MB float32 model that increases decoder capacity while keeping the image packet exactly the same.

> **Packet contract:** Every 256 × 256 image remains a `16 × 16 × 16` latent map quantized to 8-bit values. The raw latent payload remains **4,096 values / 4,096 bytes per image** before the transport container, headers, and metadata.

The extra capacity is installed at the receiver. It improves how the decoder interprets the packet; it does **not** increase the image payload.

---

## 1. Use a T4 GPU runtime

In Google Colab, select **Runtime → Change runtime type → T4 GPU**, then confirm it:

```bash
!nvidia-smi
```

---

## 2. Clone the deep-decoder v2 branch

This must clone the implementation branch, not `main`:

```bash
!git clone --branch feature/deep-decoder-80m-v2 --single-branch \
  https://github.com/ethcocoder/paragram-engin-v1.git
%cd paragram-engin-v1
!pip install -r requirements.txt
```

Confirm the active branch:

```bash
!git branch --show-current
```

Expected output:

```text
feature/deep-decoder-80m-v2
```

---

## 3. Confirm the v1 base checkpoint and v2 capacity

The deep decoder bootstraps from the verified v1 model. It will not overwrite it.

```bash
!ls -lh checkpoints/universal_genesis_core_ft_decoder_v1.pth
!sha256sum checkpoints/universal_genesis_core_ft_decoder_v1.pth
!python src/inspect_model_capacity.py \
  --checkpoint checkpoints/universal_genesis_core_ft_decoder_v1.pth
```

The `14 total bottleneck blocks` estimate should be approximately **80 MB float32**. The initial four blocks load from v1; ten new blocks are appended as identity transformations so the starting v2 reconstruction matches v1 before fine-tuning begins.

---

## 4. Train Deep-Decoder v2 — decoder-only stage

This is the required first stage. It freezes the v1 encoder and trains the larger decoder using the same deterministic 8-bit latent path used during image transmission. It uses the complete 100,000-image STL-10 unlabeled set by omitting `--sample_limit`. The default experiment is now **5 epochs**, with a validation test and recovery checkpoint after every epoch. The script automatically stops after two stale epochs when validation quality stops improving.

```bash
!python src/finetune_deterministic.py \
  --base_checkpoint checkpoints/universal_genesis_core_ft_decoder_v1.pth \
  --output_checkpoint checkpoints/universal_genesis_deep_decoder_80m_v2.pth \
  --stage decoder \
  --decoder_bottleneck_blocks 14 \
  --epochs 5 \
  --batch_size 32 \
  --early_stop_patience 2 \
  --min_psnr_delta 0.02 \
  --lr 1e-4 \
  --ssim_weight 0.5 \
  --perceptual_weight 0.1 \
  --edge_weight 0.05 \
  --val_batches 25
```

The loss includes pixel accuracy, SSIM, perceptual similarity, and an edge-preservation term. The edge term is intended to improve the fine detail that is soft in the current reconstruction while preserving the structural layout already carried by the latent.

### Per-epoch safety checks

At the end of every epoch, the script evaluates deterministic packet reconstruction on the fixed validation subset and prints PSNR, SSIM, payload bytes, and the number of stale epochs. It also writes these recovery artifacts:

```text
checkpoints/universal_genesis_deep_decoder_80m_v2.latest.pth
checkpoints/universal_genesis_deep_decoder_80m_v2.best.pth
checkpoints/universal_genesis_deep_decoder_80m_v2.progress.json
```

The `.best.pth` file contains the best validation model seen so far, `.latest.pth` contains the latest completed epoch, and `.progress.json` can be read while training is still running. This means a long run produces useful evidence after epoch 1 instead of requiring the entire experiment to finish.

### T4 memory fallback

The 80 MB decoder is substantially larger than v1. AMP is enabled automatically for the T4. If CUDA reports an out-of-memory error, restart the runtime and change only `--batch_size`:

| First attempt | First fallback | Second fallback |
|---:|---:|---:|
| 32 | 24 | 16 |

Do **not** change `--decoder_bottleneck_blocks`, `--latent_channels`, or quantization settings. Those define the model and packet contract.

---

## 5. Inspect the controlled validation result

```bash
!cat checkpoints/universal_genesis_deep_decoder_80m_v2.metrics.json
```

Keep the v2 model only if it meets all gates below on the same held-out validation images.

| Gate | Required result |
|---|---|
| Model size | `decoder_bottleneck_blocks` is `14` |
| Packet contract | `raw_payload_bytes_per_image` remains `4096` |
| Visual fidelity | `final_metrics.mean_psnr_db` exceeds the v1 baseline for the same run |
| Structure | `final_metrics.mean_ssim` does not fall materially below the baseline |
| Detail objective | `training_config.edge_weight` is `0.05` |
| Version safety | A new v2 checkpoint is written; v1 remains untouched |

> The script saves the best validation epoch by deterministic PSNR, not merely the final epoch. Five epochs is a useful first experiment; continue beyond five only if the progress report shows that PSNR/SSIM are still improving.

---

## 6. Verify v2 on ten fresh random images

The verifier downloads ten new images, checks the latent shape and payload size, reports individual and aggregate PSNR, and writes a comparison image plus JSON report.

```bash
!python src/verify_random_checkpoint.py \
  --model_path checkpoints/universal_genesis_deep_decoder_80m_v2.pth \
  --latent_channels 16 \
  --num_images 10 \
  --image_dir verification_random_images_v2 \
  --output_image deep_decoder_v2_ten_image_verification.png \
  --report_path deep_decoder_v2_ten_image_verification.json
```

Display the comparison artifact:

```python
from IPython.display import Image, display
display(Image(filename="deep_decoder_v2_ten_image_verification.png"))
```

The verification must report:

```text
Latent packet: 16x16x16 @ 8-bit = 4096 raw bytes/image
Reduction factor: 48.0X
Images tested: 10
```

A ten-image random run verifies that the checkpoint loads and works end-to-end. Do not use it as a before/after quality comparison unless v1 and v2 are evaluated on the same fixed images.

---

## 7. Optional Stage 2 — full-model fine-tuning

Run this only after Stage 1 passes the validation gates. It unfreezes the encoder and decoder at a low learning rate, while retaining the identical 4 KB packet contract.

```bash
!python src/finetune_deterministic.py \
  --base_checkpoint checkpoints/universal_genesis_deep_decoder_80m_v2.pth \
  --output_checkpoint checkpoints/universal_genesis_deep_decoder_80m_v2_full.pth \
  --stage full \
  --decoder_bottleneck_blocks 14 \
  --epochs 10 \
  --batch_size 8 \
  --lr 2e-5 \
  --ssim_weight 0.5 \
  --perceptual_weight 0.1 \
  --edge_weight 0.05 \
  --val_batches 25
```

Then repeat the ten-image verification using `universal_genesis_deep_decoder_80m_v2_full.pth`.

---

## 8. Download the v2 model and evidence

```python
from google.colab import files

files.download("checkpoints/universal_genesis_deep_decoder_80m_v2.pth")
files.download("checkpoints/universal_genesis_deep_decoder_80m_v2.best.pth")
files.download("checkpoints/universal_genesis_deep_decoder_80m_v2.progress.json")
files.download("checkpoints/universal_genesis_deep_decoder_80m_v2.metrics.json")
files.download("deep_decoder_v2_ten_image_verification.png")
files.download("deep_decoder_v2_ten_image_verification.json")
```

If you completed Stage 2, replace the two checkpoint paths with the `_full.pth` checkpoint and its matching metrics file.

---

## 9. Deployment and rollback

Each receiver must use the exact checkpoint version encoded in the message header. Use a distinct model identifier such as:

```text
model_id: genesis-deep-decoder-80m-v2
latent_shape: [16, 16, 16]
dtype: int8
quantization_scale: 1 / 127.5
payload_version: 1
```

The v2 model accepts the same latent shape as v1, but it is a separately versioned decoder. For predictable behavior, distribute the same complete checkpoint to both sender and receiver.

Rollback requires no retraining. Return to the published v1 checkpoint:

```bash
!python src/demo_hd.py \
  --model_path checkpoints/universal_genesis_core_ft_decoder_v1.pth \
  --latent_channels 16 \
  --image_dir test_local
```

---

## What Deep-Decoder v2 changes—and what it does not

| Item | v1 decoder | Deep-Decoder v2 |
|---|---:|---:|
| Decoder bottleneck residual blocks | 4 | **14** |
| Full-model float32 capacity | 32.91 MB | **approximately 80.14 MB** |
| Latent channels | 16 | 16 |
| Latent spatial size | 16 × 16 | 16 × 16 |
| Quantization | 8-bit `[-1, 1]` | 8-bit `[-1, 1]` |
| Raw latent payload | 4,096 bytes | 4,096 bytes |
| Fine-detail objective | None | Edge-preservation loss enabled |
| Original v1 checkpoint | Preserved | Preserved for rollback |

The v2 branch therefore explores a more capable receiver-side reconstruction model without increasing the cost of each transmitted image.
