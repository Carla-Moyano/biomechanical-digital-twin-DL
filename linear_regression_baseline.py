"""
Linear regression baseline for IMU → SMPL pose estimation.
Input:  orientation (45) + acceleration (15) = 60 features per frame
Output: smpl_pose (135 features per frame)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\34693\DIP_IMU_nn"
SPLITS = {
    "train": f"{DATA_DIR}/imu_own_training.npz",
    "val":   f"{DATA_DIR}/imu_own_validation.npz",
    "test":  f"{DATA_DIR}/imu_own_test.npz",
}
OUT_PLOT = f"{DATA_DIR}/linear_regression_results.png"


# ── load ───────────────────────────────────────────────────────────────────────
def load_split(path):
    data = np.load(path, allow_pickle=True)
    print(f"\n  Keys: {list(data.files)}")
    for k in ["orientation", "acceleration", "smpl_pose"]:
        arr = data[k]
        print(f"  {k}: {arr.shape[0]} sequences, e.g. seq[0]={arr[0].shape}")
    return data


print("=" * 60)
print("LOADING DATA")
print("=" * 60)
splits_raw = {}
for name, path in SPLITS.items():
    print(f"\n{name.upper()} ({path})")
    splits_raw[name] = load_split(path)


# ── statistics from training set ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTICS (from training set)")
print("=" * 60)
stats = splits_raw["train"]["statistics"].item()
for feat in ["orientation", "acceleration", "smpl_pose"]:
    mu  = stats[feat]["mean_channel"]
    std = stats[feat]["std_channel"]
    print(f"  {feat}: mean [{mu.min():.3f}, {mu.max():.3f}]  "
          f"std [{std.min():.4f}, {std.max():.4f}]")


# ── flatten sequences → frame matrices ────────────────────────────────────────
def flatten(data):
    """Stack all sequences → (total_frames, channels)."""
    return np.concatenate([seq for seq in data], axis=0)


def build_XY(split_data):
    ori = flatten(split_data["orientation"])   # (T, 45)
    acc = flatten(split_data["acceleration"])  # (T, 15)
    pose = flatten(split_data["smpl_pose"])    # (T, 135)
    X = np.concatenate([ori, acc], axis=1)     # (T, 60)
    return X, pose


print("\n" + "=" * 60)
print("BUILDING FRAME MATRICES")
print("=" * 60)
X_train, y_train = build_XY(splits_raw["train"])
X_val,   y_val   = build_XY(splits_raw["val"])
X_test,  y_test  = build_XY(splits_raw["test"])
for name, X, y in [("train", X_train, y_train),
                    ("val",   X_val,   y_val),
                    ("test",  X_test,  y_test)]:
    print(f"  {name:5s}  X={X.shape}  y={y.shape}")


# ── normalisation (per-channel z-score, training stats) ───────────────────────
print("\n" + "=" * 60)
print("NORMALISING")
print("=" * 60)

ori_mean = stats["orientation"]["mean_channel"]   # (45,)
ori_std  = stats["orientation"]["std_channel"]
acc_mean = stats["acceleration"]["mean_channel"]   # (15,)
acc_std  = stats["acceleration"]["std_channel"]

X_mean = np.concatenate([ori_mean, acc_mean])     # (60,)
X_std  = np.concatenate([ori_std,  acc_std])

# replace near-zero std to avoid division by zero
X_std = np.where(X_std < 1e-8, 1.0, X_std)

X_train_n = (X_train - X_mean) / X_std
X_val_n   = (X_val   - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std

# output: normalise y using training smpl_pose stats
y_mean = stats["smpl_pose"]["mean_channel"]       # (135,)
y_std  = stats["smpl_pose"]["std_channel"]
y_std  = np.where(y_std < 1e-8, 1.0, y_std)

y_train_n = (y_train - y_mean) / y_std
y_val_n   = (y_val   - y_mean) / y_std
y_test_n  = (y_test  - y_mean) / y_std

print(f"  X range after norm: [{X_train_n.min():.2f}, {X_train_n.max():.2f}]")
print(f"  y range after norm: [{y_train_n.min():.2f}, {y_train_n.max():.2f}]")


# ── train ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING (Ridge regression alpha=1.0)")
print("=" * 60)
model = Ridge(alpha=1.0, fit_intercept=True, max_iter=5000)
model.fit(X_train_n, y_train_n)
print("  Done.")


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true_n, y_pred_n, y_std, label):
    # denormalise
    y_true = y_true_n * y_std + y_mean
    y_pred = y_pred_n * y_std + y_mean

    rmse_per_joint = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))   # (135,)
    rmse  = rmse_per_joint.mean()

    y_range = y_true.max(axis=0) - y_true.min(axis=0)
    y_range = np.where(y_range < 1e-8, 1.0, y_range)
    nrmse = (rmse_per_joint / y_range).mean()

    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

    print(f"\n  {label}")
    print(f"    RMSE  = {rmse:.6f}")
    print(f"    nRMSE = {nrmse:.6f}  ({nrmse*100:.2f} %)")
    print(f"    R2    = {r2:.6f}")
    return rmse, nrmse, r2, rmse_per_joint, y_true, y_pred


print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

y_val_pred_n  = model.predict(X_val_n)
y_test_pred_n = model.predict(X_test_n)
y_train_pred_n = model.predict(X_train_n)

rmse_tr, nrmse_tr, r2_tr, _, _, _ = compute_metrics(
    y_train_n, y_train_pred_n, y_std, "TRAIN")
rmse_v, nrmse_v, r2_v, rmse_per_v, y_val_true, y_val_pred = compute_metrics(
    y_val_n, y_val_pred_n, y_std, "VALIDATION")
rmse_t, nrmse_t, r2_t, rmse_per_t, y_test_true, y_test_pred = compute_metrics(
    y_test_n, y_test_pred_n, y_std, "TEST")


# ── plot ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING PLOT")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Linear Regression Baseline — IMU → SMPL Pose", fontsize=14, fontweight="bold")

# 1) Bar chart: metrics across splits
ax = axes[0, 0]
splits_labels = ["Train", "Val", "Test"]
rmse_vals  = [rmse_tr,  rmse_v,  rmse_t]
nrmse_vals = [nrmse_tr, nrmse_v, nrmse_t]
r2_vals    = [r2_tr,    r2_v,    r2_t]
x = np.arange(3)
w = 0.25
ax.bar(x - w, rmse_vals,  w, label="RMSE", color="#4C72B0")
ax.bar(x,     nrmse_vals, w, label="nRMSE", color="#DD8452")
ax.bar(x + w, r2_vals,    w, label="R²", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(splits_labels)
ax.set_title("Metrics by split")
ax.legend()
ax.set_ylabel("Value")
ax.axhline(0, color="k", linewidth=0.5)

# 2) RMSE per output channel (val)
ax = axes[0, 1]
ax.plot(rmse_per_v, linewidth=0.8, color="#4C72B0", label="Validation")
ax.plot(rmse_per_t, linewidth=0.8, color="#DD8452", alpha=0.7, label="Test")
ax.set_title("RMSE per output channel")
ax.set_xlabel("Channel index (0-134)")
ax.set_ylabel("RMSE")
ax.legend()

# 3) Scatter: predicted vs true for first 5 channels (val, first 500 frames)
ax = axes[1, 0]
n = min(500, len(y_val_true))
for ch in range(5):
    ax.scatter(y_val_true[:n, ch], y_val_pred[:n, ch],
               s=2, alpha=0.3, label=f"ch{ch}")
lims = [min(y_val_true[:n, :5].min(), y_val_pred[:n, :5].min()),
        max(y_val_true[:n, :5].max(), y_val_pred[:n, :5].max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_title("Predicted vs True (val, first 5 channels, 500 frames)")
ax.set_xlabel("True"); ax.set_ylabel("Predicted")

# 4) Time-series for channel 0 (test, first 300 frames)
ax = axes[1, 1]
n = min(300, len(y_test_true))
ax.plot(y_test_true[:n, 0], label="True",      linewidth=1.2)
ax.plot(y_test_pred[:n, 0], label="Predicted", linewidth=1.0, linestyle="--")
ax.set_title("Test — time series channel 0 (300 frames)")
ax.set_xlabel("Frame"); ax.set_ylabel("SMPL pose value")
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT_PLOT}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
