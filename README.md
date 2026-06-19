# Biomechanical Digital Twin — Deep Learning for IMU-to-Pose Estimation

**TFM · Carla Moyano Segado · Universidad Alfonso X el Sabio (UAX) · 2026**

---

## Overview

This repository contains the full experimental pipeline developed for the Master's Thesis
*"Biomechanical Digital Twin: Deep Learning-based Human Pose Estimation from Inertial Measurement Units"*.

Five progressive models are implemented to reconstruct full-body SMPL pose parameters
(135 values, 45 joints × 3) from raw IMU signals (9 sensors → 60 features per frame:
45 orientation + 15 acceleration), using the DIP-IMU dataset augmented with synthetic
data from AMASS.

The key research questions are:
- Can a Transformer Encoder outperform a linear baseline for IMU-driven pose estimation?
- Do physics-informed (PINN) or temporal convolutional (U-Net) inductive biases help?
- Can **few-shot calibration** (K = 5–20 windows) adapt a pre-trained model to a new subject
  — including S06, the only female subject in the dataset — at negligible cost?

---

## Repository Structure

```
biomecánico-digital-twin-DL/
│
├── linear_regression_baseline.py    # M1: Ridge regression baseline
├── modelo_m5_fewshot.py             # M5: Few-shot calibration (K=5/10/20)
├── sintetizar_imu_amass.py          # IMU synthesis from AMASS MoCap
├── combinar_datasets.py             # Merge datasets into unified .npz
├── generar_figuras_m1_m2_m5.py     # Comparative figures (bar charts, Bland-Altman)
├── M2_Transformer_Encoder_TFM.ipynb # M2: Transformer training notebook (Colab)
├── M3_PINN_TFM.ipynb                # M3: PINN training notebook (Colab)
├── M4_UNet_TFM.ipynb                # M4: U-Net training notebook (Colab)
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.11 (CPU or CUDA) |
| NumPy | 2.0 |
| scikit-learn | 1.5 |
| matplotlib | ≥ 3.8 |
| scipy | ≥ 1.13 |

### Setup

```bash
# Clone the repository
git clone https://github.com/CarMoy/biomecanico-digital-twin-DL.git
cd biomecanico-digital-twin-DL

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch==2.11.0 numpy==2.0.0 scikit-learn==1.5.0 matplotlib scipy
```

> **Note on PyTorch version:** `torch==2.11.0+cpu` is used for CPU-only environments.
> For GPU training, install the CUDA variant matching your driver:
> `pip install torch==2.11.0+cu121 --index-url https://download.pytorch.org/whl/cu121`

---

## Data

### DIP-IMU

The primary dataset is **DIP-IMU** (Huang et al., 2018), which provides synchronized
IMU recordings and SMPL ground-truth poses for 10 subjects performing everyday motions.

- **Download:** [https://dip.is.tue.mpg.de/](https://dip.is.tue.mpg.de/) (registration required)
- Subjects S01–S08 → training; S09–S10 → test
- **S06** is the only **female** subject in the dataset and is included in the training split
  (`imu_own_training.npz`, field `data_id == 's6'`, 111 sequences, ~33 K frames)

### AMASS

Synthetic IMU signals are generated from the **AMASS** motion-capture collection
(Mahmood et al., 2019) using `sintetizar_imu_amass.py` to augment the training set.

- **Download:** [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/) (registration required)
- Only the body-model parameters (`poses`, `betas`, `trans`) are needed
- Run `sintetizar_imu_amass.py` → `combinar_datasets.py` to build the `.npz` splits

### Input / Output format

Each `.npz` split contains:

| Field | Shape per sequence | Description |
|-------|--------------------|-------------|
| `orientation` | `(T, 45)` | 9 IMUs × 5-element quaternion-derived features |
| `acceleration` | `(T, 15)` | 9 IMUs × 3-axis linear acceleration |
| `smpl_pose` | `(T, 135)` | 45 joints × 3 (axis-angle rotation) — ground truth |
| `data_id` | scalar string | Subject identifier (`'s1'`…`'s8'` in training) |
| `statistics` | dict | Per-channel mean and std for z-score normalization |

---

## Usage

### M1 — Ridge Regression Baseline

Per-frame linear mapping from IMU features to SMPL pose (no temporal context).

```bash
python M1-Baseline/linear_regression_baseline.py
```

Outputs a results plot to `M1-Baseline/linear_regression_results.png`.

---

### M2 — Transformer Encoder

5-layer Transformer Encoder with pre-norm, positional encoding,
`d_model=192`, 6 attention heads, FFN=384, trained for 30 epochs on windowed sequences (T=60 frames).

Open and run **`M2_Transformer_Encoder_TFM.ipynb`** (compatible with Google Colab and local Jupyter).

Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Window length T | 60 frames |
| d_model | 192 |
| Encoder layers | 5 |
| Attention heads | 6 |
| FFN dim | 384 |
| Dropout | 0.1 |
| Optimizer | Adam, lr=1e-4 |
| Batch size | 32 |

---

### M3 — Physics-Informed Neural Network (PINN)

MLP augmented with biomechanical constraint losses (joint-angle limits, symmetry).
Run **`M3_PINN_TFM.ipynb`**.

---

### M4 — Temporal U-Net (1D)

Encoder-decoder with skip connections operating on temporal windows (T=60).
Run **`M4_UNet_TFM.ipynb`**.

---

### M5 — Few-Shot Calibration

Adapts the pre-trained M2 checkpoint to a new subject using only K calibration windows,
via selective fine-tuning of the last encoder layer and output projection head,
regularized with L2-SP (toward the original M2 weights).

**Includes an S06 experiment** — the only female subject — using sequences from
the training split (`data_id == 's6'`) to test cross-sex generalization.

```bash
python M5-Fewshot/modelo_m5_fewshot.py
```

Key calibration settings:

| Parameter | Value |
|-----------|-------|
| K values evaluated | 5, 10, 20 windows |
| Fine-tuned layers | `encoder.layers[-1]` + `output_proj` |
| Trainable parameters | 323,079 / 1,522,887 total |
| Learning rate | 1e-5 |
| L2-SP strength (α) | 0.01 |
| Calibration epochs | 10 |
| Seed | 42 |

Outputs:
- `M5_fewshot_results.txt` — full comparative table (s09-s10 and S06)
- `M5_fewshot_s06.txt` — S06-specific results
- `M5_fewshot_results.png` — bar chart with both subject groups

---

### Generating Figures

```bash
python generar_figuras_m1_m2_m5.py
```

Produces `fig6_comparativa_5modelos.png`, `fig_boxplot_mae.png`, and `fig_bland_altman.png`.

---

## Model Checkpoints

Trained checkpoints are **not included** in this repository due to file size constraints.

| Checkpoint | Size | Notes |
|------------|------|-------|
| `M2_epoch29_best.pt` | ~24 MB | Required by M5 |
| `M3_PINN_best.pt` | ~12 MB | — |
| `M4_UNet_best.pt` | ~18 MB | — |

**Checkpoints are available upon request** — please open a GitHub Issue or contact
the author directly (see below). Place them in their respective model folders
before running the scripts.

---

## Results

All metrics are computed on the **DIP-IMU test set** (subjects S09–S10, 859 windows of 60 frames)
in the **original (denormalized) space**, averaged uniformly across all 135 SMPL pose channels.

### Main comparison (test split: S09–S10)

| Model | Architecture | RMSE ↓ | nRMSE (%) ↓ | R² ↑ |
|-------|-------------|--------|------------|------|
| **M1** | Ridge Regression | 0.1293 | 11.66 % | 0.261 |
| **M2** | Transformer Encoder | 0.1097 | 10.01 % | 0.418 |
| **M3** | MLP-PINN | 0.1442 | 7.21 % | 0.298 |
| **M4** | U-Net 1D | 0.1539 | 7.70 % | 0.238 |
| **M5** K=5 | M2 + few-shot | 0.1096 | 10.01 % | 0.420 |
| **M5** K=10 | M2 + few-shot | 0.1088 | 9.92 % | 0.429 |
| **M5** K=20 | M2 + few-shot | **0.1083** | **9.88 %** | **0.436** |

> M2 is the strongest single model. M5 (K=20) achieves the best overall result with
> only 323 K trainable parameters updated on 20 calibration windows (~1,200 frames).

### S06 experiment (female subject — training split)

S06 is evaluated separately because she belongs to the training split,
raising the question of whether M2 already generalises to her or benefits from calibration.

| Model | RMSE ↓ | nRMSE (%) ↓ | R² ↑ | n windows |
|-------|--------|------------|------|-----------|
| M2 (no calibration) on S06 | 0.0435 | 4.96 % | 0.864 | 552 |
| M5 K=20 on S06 | 0.0436 | 4.97 % | 0.864 | 532 |

> M2 already predicts S06 with high accuracy (R² = 0.864 vs 0.418 on unseen subjects),
> suggesting S06's motion patterns are well covered by the training distribution.
> Few-shot calibration yields no meaningful gain in this case.

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@mastersthesis{moyano2026biomechanical,
  author  = {Moyano Segado, Carla},
  title   = {Biomechanical Digital Twin: Deep Learning-based Human Pose
             Estimation from Inertial Measurement Units},
  school  = {Universidad Alfonso X el Sabio (UAX)},
  year    = {2026},
  address = {Madrid, Spain},
}
```

Please also cite the original datasets:

```bibtex
@inproceedings{huang2018dip,
  title     = {Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time},
  author    = {Huang, Yinghao and Kaufmann, Manuel and Aksan, Emre and Black, Michael J. and Hilliges, Otmar and Pons-Moll, Gerard},
  booktitle = {ACM SIGGRAPH Asia},
  year      = {2018},
}

@inproceedings{mahmood2019amass,
  title     = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  author    = {Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {ICCV},
  year      = {2019},
}
```

---

## Contact

**Carla Moyano Segado**
Master's in Artificial Intelligence — Universidad Alfonso X el Sabio (UAX), Madrid
carmose1503@gmail.com

---

*This project was developed as part of a Master's Thesis (TFM) at UAX, 2026.*
