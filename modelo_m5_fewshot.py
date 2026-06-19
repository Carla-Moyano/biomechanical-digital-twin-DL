"""
M5 — Few-shot calibration of Transformer Encoder M2
====================================================
Loads the best M2 checkpoint (M2_epoch29_best.pt),
applies selective fine-tuning (last encoder layer + output_proj)
on K calibration windows from the test set (subjects s09-s10),
and evaluates RMSE / nRMSE / R² on the remaining test windows.

K is evaluated for K in [5, 10, 20].
Regularization: L2-SP (toward original M2 weights, alpha=0.01).

Additional experiment: S06 calibration
---------------------------------------
S06 is the only female subject in the dataset and lives in the
training split (imu_own_training.npz, data_id == 's6').
We calibrate M2 with K=20 windows drawn from S06 and evaluate on the
remaining S06 windows, then compare against uncalibrated M2 on S06.
Results are saved to M5_fewshot_s06.txt and merged into the final table.
"""

import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = r"C:\Users\34693\DIP_IMU_nn"
CKPT_PATH     = os.path.join(DATA_DIR, "M2-Transformer", "M2_epoch29_best.pt")
TEST_PATH     = os.path.join(DATA_DIR, "imu_own_test.npz")
TRAIN_PATH    = os.path.join(DATA_DIR, "imu_own_training.npz")
OUT_TABLE     = os.path.join(DATA_DIR, "M5_fewshot_results.txt")
OUT_TABLE_S06 = os.path.join(DATA_DIR, "M5_fewshot_s06.txt")
OUT_PLOT      = os.path.join(DATA_DIR, "M5_fewshot_results.png")

# ── Model hyperparameters (identical to M2 notebook) ──────────────────────────
T        = 60     # window length (frames)
IN_DIM   = 60     # ori(45) + acc(15)
OUT_DIM  = 135    # smpl_pose
D_MODEL  = 192
NHEAD    = 6
N_LAYERS = 5
DIM_FFN  = 384
DROPOUT  = 0.1

# ── Calibration hyperparameters ───────────────────────────────────────────────
K_VALUES     = [5, 10, 20]   # number of calibration windows
CALIB_LR     = 1e-5
CALIB_ALPHA  = 0.01          # L2-SP strength toward original weights
CALIB_EPOCHS = 10
CALIB_BATCH  = 8
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Architecture ───────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerEncoderM2(nn.Module):
    """d_model=192, 5 encoder layers, 6 heads, FFN=384, dropout=0.1."""

    def __init__(
        self,
        in_dim: int = IN_DIM,
        out_dim: int = OUT_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FFN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        return self.output_proj(x)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_norm_params():
    """Extract per-channel normalization params from training split."""
    print("Loading normalization stats from training data (lazy)...")
    train_raw = np.load(TRAIN_PATH, allow_pickle=True)
    stats = train_raw["statistics"].item()

    ori_mu  = stats["orientation"]["mean_channel"].astype(np.float32)
    ori_std = stats["orientation"]["std_channel"].astype(np.float32)
    acc_mu  = stats["acceleration"]["mean_channel"].astype(np.float32)
    acc_std = stats["acceleration"]["std_channel"].astype(np.float32)
    y_mu    = stats["smpl_pose"]["mean_channel"].astype(np.float32)
    y_std   = stats["smpl_pose"]["std_channel"].astype(np.float32)

    X_mu  = np.concatenate([ori_mu,  acc_mu])
    X_std = np.concatenate([ori_std, acc_std])
    X_std = np.where(X_std < 1e-8, 1.0, X_std)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    return X_mu, X_std, y_mu, y_std


def build_windows(
    seqs_ori,
    seqs_acc,
    seqs_pose,
    X_mu: np.ndarray,
    X_std: np.ndarray,
    y_mu: np.ndarray,
    y_std: np.ndarray,
    stride: int = T,
):
    """Slice raw sequences into normalized windows of length T."""
    X_wins, y_wins = [], []
    for ori, acc, pose in zip(seqs_ori, seqs_acc, seqs_pose):
        L = len(ori)
        X_seq = np.concatenate([ori, acc], axis=-1).astype(np.float32)
        y_seq = pose.astype(np.float32)
        X_seq = (X_seq - X_mu) / X_std
        y_seq = (y_seq - y_mu) / y_std
        for s in range(0, L - T + 1, stride):
            X_wins.append(X_seq[s : s + T])
            y_wins.append(y_seq[s : s + T])
    if not X_wins:
        return np.empty((0, T, IN_DIM), dtype=np.float32), \
               np.empty((0, T, OUT_DIM), dtype=np.float32)
    return np.stack(X_wins).astype(np.float32), np.stack(y_wins).astype(np.float32)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds_n: np.ndarray,
    trues_n: np.ndarray,
    y_mu: np.ndarray,
    y_std: np.ndarray,
) -> tuple[float, float, float]:
    """Return (RMSE, nRMSE, R²) in original (denormalized) space."""
    preds = preds_n * y_std + y_mu
    trues = trues_n * y_std + y_mu

    rmse_per = np.sqrt(np.mean((trues - preds) ** 2, axis=0))
    rmse = float(rmse_per.mean())

    y_range = trues.max(0) - trues.min(0)
    y_range = np.where(y_range < 1e-8, 1.0, y_range)
    nrmse = float((rmse_per / y_range).mean())

    r2 = float(r2_score(trues, preds, multioutput="uniform_average"))
    return rmse, nrmse, r2


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    y_mu: np.ndarray,
    y_std: np.ndarray,
    batch_size: int = 64,
) -> tuple[float, float, float]:
    """Run inference and return (RMSE, nRMSE, R²)."""
    model.eval()
    ds     = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_n, trues_n = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            pred = model(X_b.to(DEVICE)).cpu().numpy()
            preds_n.append(pred.reshape(-1, OUT_DIM))
            trues_n.append(y_b.numpy().reshape(-1, OUT_DIM))

    return compute_metrics(
        np.concatenate(preds_n),
        np.concatenate(trues_n),
        y_mu, y_std,
    )


# ── Few-shot calibration ───────────────────────────────────────────────────────

def fewshot_calibrate(
    base_model: nn.Module,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    lr: float = CALIB_LR,
    alpha: float = CALIB_ALPHA,
    epochs: int = CALIB_EPOCHS,
    batch_size: int = CALIB_BATCH,
) -> nn.Module:
    """
    Deep-copies base_model and fine-tunes ONLY:
      - encoder.layers[-1]  (last TransformerEncoderLayer)
      - output_proj         (Linear head)

    Loss = MSE + alpha * ||theta - theta_0||^2   (L2-SP regularization)
    alpha pulls calibrated weights back toward the original M2 values.
    Returns the calibrated model; base_model is NOT modified.
    """
    model = copy.deepcopy(base_model)
    model.to(DEVICE)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last encoder layer + output head
    for p in model.encoder.layers[-1].parameters():
        p.requires_grad = True
    for p in model.output_proj.parameters():
        p.requires_grad = True

    # Anchor: frozen reference weights for L2-SP regularization
    anchor = {
        name: param.detach().clone().to(DEVICE)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(trainable, lr=lr)
    criterion  = nn.MSELoss()

    ds     = TensorDataset(torch.from_numpy(X_calib), torch.from_numpy(y_calib))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    n_trainable = sum(p.numel() for p in trainable)
    print(f"    Trainable params: {n_trainable:,}  "
          f"({len(X_calib)} calib windows, batch={batch_size})")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred     = model(X_b)
            mse      = criterion(pred, y_b)

            # L2-SP: sum of squared deviations from anchor weights
            l2_sp = sum(
                torch.sum((param - anchor[name]) ** 2)
                for name, param in model.named_parameters()
                if param.requires_grad
            )
            loss = mse + alpha * l2_sp

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:>2}/{epochs}  "
                  f"loss={epoch_loss / max(len(ds), 1):.6f}")

    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    # ── Validate inputs ─────────────────────────────────────────────────────
    for path, label in [(CKPT_PATH, "M2 checkpoint"), (TEST_PATH, "test data"),
                        (TRAIN_PATH, "training data (for stats)")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "  For the checkpoint: run the M2 notebook and copy "
                "M2_epoch29_best.pt to the DATA_DIR."
            )

    # ── Load model ──────────────────────────────────────────────────────────
    print("\nLoading M2 checkpoint...")
    try:
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        # PyTorch < 2.0 does not have weights_only parameter
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)  # noqa: S614
    base_model = TransformerEncoderM2().to(DEVICE)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Epoch {ckpt.get('epoch', '?')} | params={n_params:,}")

    # ── Normalization stats ─────────────────────────────────────────────────
    X_mu, X_std, y_mu, y_std = load_norm_params()

    # ── Load test data ──────────────────────────────────────────────────────
    print("\nLoading test data (s09-s10)...")
    test_raw = np.load(TEST_PATH, allow_pickle=True)
    print(f"  Keys      : {list(test_raw.files)}")

    n_seqs = len(test_raw["orientation"])
    print(f"  Sequences : {n_seqs}")
    for i in range(min(3, n_seqs)):
        print(f"    seq[{i}] len={len(test_raw['orientation'][i])}")

    # Determine per-subject indices.
    # DIP-IMU test set (subjects 09 and 10) does not embed subject IDs in the
    # npz file.  Sequences are ordered subject-first, so the first half belong
    # to s09 and the second half to s10.
    if "subject" in test_raw.files:
        subj_ids = test_raw["subject"]
        unique_s = np.unique(subj_ids)
        print(f"  Subjects found: {unique_s}")
        idx_s09 = np.where(subj_ids == unique_s[0])[0]
        idx_s10 = np.where(subj_ids == unique_s[1])[0] if len(unique_s) > 1 else np.array([], dtype=int)
    else:
        mid     = n_seqs // 2
        idx_s09 = np.arange(0, mid)
        idx_s10 = np.arange(mid, n_seqs)
        print(f"  No 'subject' key found - splitting: s09=[0,{mid}), s10=[{mid},{n_seqs})")

    ori_arr  = test_raw["orientation"]
    acc_arr  = test_raw["acceleration"]
    pose_arr = test_raw["smpl_pose"]

    X_s09, y_s09 = build_windows(
        ori_arr[idx_s09], acc_arr[idx_s09], pose_arr[idx_s09],
        X_mu, X_std, y_mu, y_std,
    )
    X_s10, y_s10 = build_windows(
        ori_arr[idx_s10], acc_arr[idx_s10], pose_arr[idx_s10],
        X_mu, X_std, y_mu, y_std,
    )
    X_all = np.concatenate([X_s09, X_s10], axis=0)
    y_all = np.concatenate([y_s09, y_s10], axis=0)
    print(f"\n  Windows - s09: {len(X_s09)}, s10: {len(X_s10)}, total: {len(X_all)}")

    # ── M2 baseline (no calibration) ────────────────────────────────────────
    print("\nEvaluating M2 baseline (no calibration)...")
    rmse_base, nrmse_base, r2_base = evaluate_model(
        base_model, X_all, y_all, y_mu, y_std
    )
    print(f"  RMSE={rmse_base:.6f}  nRMSE={nrmse_base*100:.2f}%  R²={r2_base:.4f}")

    # ── Few-shot calibration for each K ─────────────────────────────────────
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []  # results table

    for K in K_VALUES:
        sep = "=" * 60
        print(f"\n{sep}\nK = {K} calibration windows\n{sep}")

        n_total = len(X_all)
        if K >= n_total:
            print(f"  Skipping: K={K} >= available windows={n_total}")
            continue

        calib_idx = rng.choice(n_total, size=K, replace=False)
        eval_mask = np.ones(n_total, dtype=bool)
        eval_mask[calib_idx] = False
        eval_idx  = np.where(eval_mask)[0]

        X_calib = X_all[calib_idx]
        y_calib = y_all[calib_idx]
        X_eval  = X_all[eval_idx]
        y_eval  = y_all[eval_idx]

        print(f"  Calib={len(X_calib)} windows  Eval={len(X_eval)} windows  "
              f"(each window = {T} frames)")

        cal_model = fewshot_calibrate(base_model, X_calib, y_calib)

        rmse_k, nrmse_k, r2_k = evaluate_model(
            cal_model, X_eval, y_eval, y_mu, y_std
        )
        print(f"  Result K={K}: RMSE={rmse_k:.6f}  "
              f"nRMSE={nrmse_k*100:.2f}%  R²={r2_k:.4f}")

        rows.append({
            "K":      K,
            "RMSE":   rmse_k,
            "nRMSE":  nrmse_k,
            "R2":     r2_k,
            "n_eval": len(X_eval),
        })

    # ── S06 experiment (female subject, from training split) ────────────────
    print(f"\n{'=' * 68}")
    print("S06 EXPERIMENT  (only female subject - imu_own_training.npz)")
    print(f"{'=' * 68}")

    train_raw = np.load(TRAIN_PATH, allow_pickle=True)
    data_id   = train_raw["data_id"]           # dtype <U2, values 's1'..'s8'
    s06_mask  = data_id == "s6"
    idx_s06   = np.where(s06_mask)[0]
    print(f"  S06 sequences in training split: {len(idx_s06)}")

    ori_tr   = train_raw["orientation"]
    acc_tr   = train_raw["acceleration"]
    pose_tr  = train_raw["smpl_pose"]

    X_s06, y_s06 = build_windows(
        ori_tr[idx_s06], acc_tr[idx_s06], pose_tr[idx_s06],
        X_mu, X_std, y_mu, y_std,
    )
    print(f"  Total S06 windows: {len(X_s06)}")

    # M2 baseline on S06 (no calibration)
    print("\n  M2 baseline (no calibration) on S06...")
    rmse_s06_base, nrmse_s06_base, r2_s06_base = evaluate_model(
        base_model, X_s06, y_s06, y_mu, y_std
    )
    print(f"  RMSE={rmse_s06_base:.6f}  "
          f"nRMSE={nrmse_s06_base*100:.2f}%  R²={r2_s06_base:.4f}")

    # Few-shot calibration with K=20 on S06
    K_S06 = 20
    rng_s06 = np.random.default_rng(SEED)
    n_s06 = len(X_s06)
    calib_idx_s06 = rng_s06.choice(n_s06, size=K_S06, replace=False)
    eval_mask_s06 = np.ones(n_s06, dtype=bool)
    eval_mask_s06[calib_idx_s06] = False
    eval_idx_s06  = np.where(eval_mask_s06)[0]

    X_calib_s06 = X_s06[calib_idx_s06]
    y_calib_s06 = y_s06[calib_idx_s06]
    X_eval_s06  = X_s06[eval_idx_s06]
    y_eval_s06  = y_s06[eval_idx_s06]

    print(f"\n  Few-shot K={K_S06}: calib={len(X_calib_s06)}  eval={len(X_eval_s06)}")
    cal_model_s06 = fewshot_calibrate(base_model, X_calib_s06, y_calib_s06)

    rmse_s06_k, nrmse_s06_k, r2_s06_k = evaluate_model(
        cal_model_s06, X_eval_s06, y_eval_s06, y_mu, y_std
    )
    print(f"  Result K={K_S06}: RMSE={rmse_s06_k:.6f}  "
          f"nRMSE={nrmse_s06_k*100:.2f}%  R²={r2_s06_k:.4f}")

    # Save S06-specific results
    def fmt_row(label: str, K: str, rmse: float, nrmse: float, r2: float, n: int) -> str:
        return (f"{label:<28}  {K:>5}  {rmse:10.6f}  "
                f"{nrmse*100:8.2f}  {r2:8.4f}  {n:>7}")

    header_s06 = (f"{'Model':<28}  {'K':>5}  {'RMSE':>10}  "
                  f"{'nRMSE%':>8}  {'R²':>8}  {'n_eval':>7}")
    sep_s06    = "-" * 74

    lines_s06 = [
        "M5 Few-shot Calibration - S06 (female subject)",
        f"Checkpoint  : {CKPT_PATH}",
        f"lr          : {CALIB_LR}  |  alpha : {CALIB_ALPHA}  |  epochs : {CALIB_EPOCHS}",
        f"Window T    : {T}  |  seed : {SEED}  |  K_S06 : {K_S06}",
        f"Source      : imu_own_training.npz  (data_id == 's6')",
        "",
        header_s06,
        sep_s06,
        fmt_row("M2 (no calib) on S06", "-",
                rmse_s06_base, nrmse_s06_base, r2_s06_base, len(X_s06)),
        fmt_row(f"M5 few-shot S06", str(K_S06),
                rmse_s06_k, nrmse_s06_k, r2_s06_k, len(X_eval_s06)),
    ]

    print(f"\n{'=' * 74}")
    for line in lines_s06:
        print(line)

    with open(OUT_TABLE_S06, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_s06) + "\n")
    print(f"\nS06 table saved : {OUT_TABLE_S06}")

    # ── Comparative table (s09-s10 + S06) ──────────────────────────────────
    print(f"\n{'=' * 74}")
    print("FULL COMPARATIVE TABLE: M2 baseline vs M5 few-shot (s09-s10 & S06)")
    print(f"{'=' * 74}")

    header = (f"{'Model':<28}  {'K':>5}  {'RMSE':>10}  "
              f"{'nRMSE%':>8}  {'R2':>8}  {'n_eval':>7}")
    sep    = "-" * 74

    lines = [
        "M5 Few-shot Calibration - Full Comparative Results",
        f"Checkpoint  : {CKPT_PATH}",
        f"lr          : {CALIB_LR}  |  alpha : {CALIB_ALPHA}  |  epochs : {CALIB_EPOCHS}",
        f"Window T    : {T}  |  seed : {SEED}",
        "",
        "-- s09-s10 (test split) ---------------------------------------------",
        header,
        sep,
        fmt_row("M2 (no calib) s09-s10", "-",
                rmse_base, nrmse_base, r2_base, len(X_all)),
    ]

    for row in rows:
        lines.append(fmt_row(f"M5 few-shot s09-s10", str(row["K"]),
                              row["RMSE"], row["nRMSE"], row["R2"], row["n_eval"]))

    lines += [
        "",
        "-- S06 (training split - only female subject) -----------------------",
        header,
        sep,
        fmt_row("M2 (no calib) on S06", "-",
                rmse_s06_base, nrmse_s06_base, r2_s06_base, len(X_s06)),
        fmt_row("M5 few-shot S06", str(K_S06),
                rmse_s06_k, nrmse_s06_k, r2_s06_k, len(X_eval_s06)),
    ]

    for line in lines:
        print(line)

    with open(OUT_TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nFull table saved : {OUT_TABLE}")

    # ── Plot ────────────────────────────────────────────────────────────────
    K_labels   = [f"K={r['K']}" for r in rows]
    # s09-s10 bars
    labels_910  = ["M2\n(base\ns09-s10)"] + K_labels
    rmse_910    = [rmse_base]  + [r["RMSE"]  for r in rows]
    nrmse_910   = [nrmse_base] + [r["nRMSE"] for r in rows]
    r2_910      = [r2_base]    + [r["R2"]    for r in rows]
    # S06 bars
    labels_s06  = ["M2\n(base\nS06)", f"M5\nS06\nK={K_S06}"]
    rmse_s06v   = [rmse_s06_base, rmse_s06_k]
    nrmse_s06v  = [nrmse_s06_base, nrmse_s06_k]
    r2_s06v     = [r2_s06_base, r2_s06_k]

    all_labels = labels_910 + [""] + labels_s06
    all_rmse   = rmse_910   + [None] + rmse_s06v
    all_nrmse  = nrmse_910  + [None] + nrmse_s06v
    all_r2     = r2_910     + [None] + r2_s06v

    # Build display arrays (skip None separator)
    disp_labels = [l for l, v in zip(all_labels, all_rmse) if v is not None]
    disp_rmse   = [v for v in all_rmse  if v is not None]
    disp_nrmse  = [v for v in all_nrmse if v is not None]
    disp_r2     = [v for v in all_r2    if v is not None]

    n_910 = len(labels_910)
    n_s06 = len(labels_s06)
    palette_910 = ["#999999", "#4C72B0", "#DD8452", "#55A868"]
    palette_s06 = ["#BBBBBB", "#C44E52"]
    colors = palette_910[:n_910] + palette_s06[:n_s06]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"M5 Few-shot Calibration  (lr={CALIB_LR}, α={CALIB_ALPHA}, "
        f"epochs={CALIB_EPOCHS}, T={T})\n"
        f"Left group: s09-s10 (test split) · Right group: S06 (female, training split)",
        fontsize=10, fontweight="bold",
    )

    x_pos = list(range(n_910)) + [n_910 + 0.6 + j for j in range(n_s06)]

    metric_cfg = [
        (axes[0], disp_rmse,                      "RMSE (lower ↓)",     "RMSE",      lambda v: f"{v:.5f}"),
        (axes[1], [v * 100 for v in disp_nrmse],  "nRMSE % (lower ↓)", "nRMSE (%)", lambda v: f"{v:.2f}%"),
        (axes[2], disp_r2,                         "R² (higher ↑)",      "R²",         lambda v: f"{v:.4f}"),
    ]

    for ax, values, title, ylabel, fmt_fn in metric_cfg:
        bars = ax.bar(x_pos, values, color=colors, edgecolor="black", linewidth=0.6,
                      width=0.7)
        # Reference lines: M2 baseline for s09-s10 and for S06
        ax.axhline(values[0], color="#555555", ls="--", lw=1.0, alpha=0.6,
                   label="M2 base s09-s10")
        ax.axhline(values[n_910], color="#AA2222", ls=":", lw=1.0, alpha=0.6,
                   label="M2 base S06")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(disp_labels, fontsize=8)
        ax.legend(fontsize=7)
        if "R²" in title:
            ax.axhline(0, color="black", lw=0.5)
        # Add a vertical separator between groups
        ax.axvline(n_910 - 1 + 0.8, color="black", lw=0.8, ls="-", alpha=0.3)
        spread = max(values) - min(values)
        offset = spread * 0.02 if spread > 1e-9 else 0.001
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt_fn(val),
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved  : {OUT_PLOT}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
