# SCALENet — Efficient Low-Light Image Enhancement
**NTIRE 2026 ELLIE Challenge · Team VARCHASVI_SVNIT · Overall Rank 13/26**

SCALENet is a lightweight (184K params, 0.70 MB) deep network for low-light image enhancement. Its core idea is *illumination-awareness*: a global luminance scalar from the input conditions every normalisation layer and drives a multi-scale exposure fusion module, letting the model adapt its behaviour based on how dark the scene actually is.

| Metric | Score | Rank |
|--------|-------|------|
| SSIM ↑ | 0.5575 | 17/26 |
| LPIPS ↓ | 0.4409 | **5/26** |
| DISTS ↓ | 0.2415 | 11/26 |
| MUSIQ ↑ | 62.243 | **7/26** |
| Q-Align ↑ | 3.004 | 16/26 |
| LIQE ↑ | 1.5017 | 26/26 |

---

## Repository Structure

```
├── model_v3.py        # SCALENet architecture (v3)
├── losses_v3.py       # CompetitionLoss — all training loss terms
├── train_v3.py        # Training script with progressive resolution
├── inference_v3.py    # Inference script — single image or batch directory
└── requirements.txt   # Dependencies
```

---

## Setup

**Requirements:** Python 3.8+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/CodingIsCodeine/Efficient_LLIE_v2
cd Efficient_LLIE_v2
pip install -r requirements.txt
```

Core dependencies: `torch>=1.9.0`, `torchvision>=0.10.0`, `lpips>=0.1.4`, `pillow`, `numpy`, `tqdm`.

---

## Inference

### Single image
```bash
python inference_v3.py \
  --model  checkpoints_v3/best_model.pth \
  --input  path/to/low_light.png \
  --output path/to/output.png
```

### Batch directory
```bash
python inference_v3.py \
  --model  checkpoints_v3/best_model.pth \
  --input  path/to/input_dir/ \
  --output path/to/output_dir/ \
  --batch
```

### With TTA (recommended for best quality)
TTA runs the model on the original and its horizontal flip, then averages the two outputs. Gains approximately +1–3% SSIM at the cost of 2× inference time.
```bash
# Single image with TTA
python inference_v3.py \
  --model  checkpoints_v3/best_model.pth \
  --input  path/to/low_light.png \
  --output path/to/output.png \
  --tta

# Batch with TTA
python inference_v3.py \
  --model  checkpoints_v3/best_model.pth \
  --input  path/to/input_dir/ \
  --output path/to/output_dir/ \
  --batch --tta
```

### All inference flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to `.pth` checkpoint |
| `--input` | required | Input image or directory |
| `--output` | required | Output image or directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--tile_size` | `512` | Tile size for images > 1024px |
| `--overlap` | `128` | Overlap between tiles (cosine-blended) |
| `--batch` | off | Enable directory batch mode |
| `--tta` | off | Horizontal-flip test-time augmentation (+1–3% SSIM, 2× slower) |
| `--sharpen` | off | Mild unsharp mask post-processing (amount=0.3) |

**Note on checkpoints:** The training script saves three files to `--checkpoint_dir`:
- `best_model.pth` — EMA weights only; use this for inference
- `best.pth` — full checkpoint (best LPIPS epoch)
- `latest.pth` — full checkpoint (most recent epoch)

Pass `best_model.pth` to `--model` for inference.

---

## Training

### Data layout

The training script expects the following folder structure:

```
data/
└── train/
    ├── low/    ← low-light input images (.jpg or .png)
    └── high/   ← ground-truth images (same filenames)
```

The dataset is split 80/20 into train/val automatically.

### Run training
```bash
python train_v3.py \
  --data_root      ./data \
  --checkpoint_dir ./checkpoints_v3 \
  --epochs         200 \
  --lr             2e-4
```

### Resume from checkpoint
```bash
python train_v3.py \
  --data_root      ./data \
  --checkpoint_dir ./checkpoints_v3 \
  --epochs         200 \
  --resume         ./checkpoints_v3/latest.pth
```

### All training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `./data` | Root folder containing `train/low` and `train/high` |
| `--checkpoint_dir` | `./checkpoints_v3` | Where to save checkpoints |
| `--epochs` | `50` | Total training epochs |
| `--lr` | `2e-4` | Initial learning rate |
| `--resume` | None | Path to checkpoint to resume from |

### Training details

| Detail | Value |
|--------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, wd=1e-4) |
| LR schedule | CosineAnnealingLR → η_min=1e-6 |
| Gradient clipping | 1.0 |
| EMA decay | 0.999 |
| Mixed precision | `torch.amp.autocast` + `GradScaler` |
| Multi-GPU | `nn.DataParallel` (auto-detected) |
| Progressive stages | 256px/bs16 (ep 0–39) → 384px/bs8 (ep 40–89) → 512px/bs4 (ep 90+) |

---

## Model Overview

SCALENet has three stages:

1. **Illumination-Conditioned Normalisation (ICN)** — a two-layer MLP maps the scalar mean luminance of the input to per-channel scale/shift parameters (FiLM-style), so normalisation behaves differently for very dark vs. moderately dark inputs.

2. **Multi-Scale Exposure Fusion (MSEF)** — three parallel branches process under-exposed, normal, and over-exposed virtual versions of the input (via learnable gamma). A spatial softmax fusion network blends them, biasing dark regions toward the over-exposed branch.

3. **Deep Refinement + Spatial Attention** — a 4-layer conv head with a mid-network skip connection. A `SpatialAttentionGate` focuses on dark patches; a channel attention module corrects residual color bias.

**Model complexity:**

| Property | Value |
|----------|-------|
| Trainable parameters | 184,064 |
| Model size (fp32) | 0.70 MB |
| GFLOPs @ 256×256 | 21.85 |
| GFLOPs @ 512×512 | 87.40 |

---

## Team

**Team:** VARCHASVI_SVNIT  
**Members:** Arya Shah, Hriday Kamlesh Samdani — SVNIT Surat, India  
**Supervisors:** Prof. Kishor Upla (SVNIT), Prof. Kiran Raja (NTNU, Norway)  
**Contact:** hridaysamdani2330@gmail.com
