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
- Can **few-shot calibration** (K = 20 windows) adapt a pre-trained model to a new subject
  — including S06, the only female subject in the dataset — at negligible cost?

---

## Repository Structure

```
biomechanical-digital-twin-DL/
│
├── linear_regression_baseline.py        # M1 — Ridge regression (α=1.0), per-frame prediction
├── M2_Transformer_Encoder_TFM.ipynb     # M2 — Transformer Encoder training notebook (Colab / local)
├── M3_PINN_TFM.ipynb                    # M3 — Physics-informed MLP notebook
├── M4_UNet_TFM.ipynb                    # M4 — 1D temporal U-Net notebook
├── modelo_m5_fewshot.py                 # M5 — Few-shot calibration of M2 (L2-SP, K=20)
│
├── sintetizar_imu_amass.py              # Synthesize IMU signals from AMASS MoCap sequences
├── combinar_datasets.py                 # Merge DIP-IMU + AMASS into unified .npz splits
├── generar_figuras_m1_m2_m5.py         # Comparative bar charts, boxplots, Bland-Altman
├── generar_figuras_finales.ipynb        # Publication-quality figures for the thesis
│
├── M5_fewshot_results.txt               # Full comparative table M5 (s09-s10 + S06)
├── M5_fewshot_s06.txt                   # S06-specific calibration results
│
├── requirements.txt
└── README.md
```

> **Note on model checkpoints:** Trained checkpoints (`M2_epoch29_best.pt`, `M3_PINN_best.pt`,
> `M4_UNet_best.pt`) are not included in this repository due to file size constraints.
> They are **available upon request** — please open a GitHub Issue or contact the author directly.

---

## Installation

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 (CPU or CUDA) |
| NumPy | 1.26.4 |
| scikit-learn | 1.4.2 |
| matplotlib | ≥ 3.8 |
| scipy | ≥ 1.13 |

### Setup

```bash
# Clone the repository
git clone https://github.com/Carla-Moyano/biomechanical-digital-twin-DL
cd biomecanico-digital-twin-DL

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch==2.1.2 numpy==1.26.4 scikit-learn==1.4.2 matplotlib scipy
```

> **Note on PyTorch version:** `torch==2.1.2+cpu` is used for CPU-only environments.
> For GPU training, install the CUDA variant matching your driver:
> `pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121`

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
| `orientation` | `(T, 45)` | 5 IMUs × 9 valores (matriz rotación 3×3) |
| `acceleration` | `(T, 15)` | 5 IMUs × 3-axis linear acceleration |
| `smpl_pose` | `(T, 135)` | 45 joints × 3 (axis-angle rotation) — ground truth |
| `data_id` | scalar string | Subject identifier (`'s1'`…`'s8'` in training) |
| `statistics` | dict | Per-channel mean and std for z-score normalization |

Total: **60 features de entrada por frame** (45 orientation + 15 acceleration).

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

## Results

| Model | RMSE (rad) | nRMSE (%) | R² | Latency (CPU) |
|-------|------------|-----------|-----|---------------|
| M1 Ridge baseline | 0.1293 | 12.93 | 0.261 | 0.06 ms |
| M2 Transformer encoder | 0.1097 | 10.01 | 0.418 | 3.20 ms |
| M3 PINN biomechanical | 0.1442 | 7.21 | 0.298 | 0.47 ms |
| M4 U-Net 1D temporal | 0.1539 | 11.87 | 0.238 | 1.29 ms |
| M5 Few-shot K=20 | 0.1083 | 9.89 | 0.436 | 3.84 ms |

All models operate well below the 16.7 ms/window real-time threshold at 60 Hz. M5 (few-shot calibration) achieves the best overall performance. Latency measured on CPU with 2000 iterations and 200 warmup runs (torch.no_grad()).

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
