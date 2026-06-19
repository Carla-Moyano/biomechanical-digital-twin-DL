#!/usr/bin/env python3
"""
Genera 3 figuras TFM/artículo para los modelos M1-M5.
  fig6_comparativa_5modelos.png  — barras agrupadas RMSE/nRMSE/R² (números exactos)
  fig_boxplot_mae.png            — boxplot MAE real M1/M2/M5
  fig_bland_altman.png           — Bland-Altman M2 vs M5 (lado a lado)

Datos: C:\\Users\\34693\\DIP_IMU_nn\\
"""
import os, re, copy, math, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR  = r'C:\Users\34693\DIP_IMU_nn'
CKPT_M2   = r"C:\Users\34693\DIP_IMU_nn\M2-Transformer\M2_epoch29_best.pt"
TRAIN_NPZ = os.path.join(BASE_DIR, 'imu_own_training.npz')
TEST_NPZ  = os.path.join(BASE_DIR, 'imu_own_test.npz')
OUT_DIR   = BASE_DIR

T        = 60
K_SHOTS  = 20
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Números exactos para fig6 (los 5 modelos)
EXACT = {
    'M1 Ridge':       {'RMSE': 0.1293, 'nRMSE': 11.66, 'R2': 0.261},
    'M2 Transformer': {'RMSE': 0.1097, 'nRMSE': 10.01, 'R2': 0.418},
    'M3 MLP-PINN':    {'RMSE': 0.1442, 'nRMSE':  7.21, 'R2': 0.298},
    'M4 UNet1D':      {'RMSE': 0.1539, 'nRMSE':  7.70, 'R2': 0.238},
    'M5 Fine-tune':   {'RMSE': 0.1083, 'nRMSE':  9.89, 'R2': 0.436},
}
PALETTE_5 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
PALETTE_3 = ['#1f77b4', '#ff7f0e', '#9467bd']
LABELS_3  = ['M1 Ridge', 'M2 Transformer', 'M5 Fine-tune']

# ── MODELO ────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerEncoderM2(nn.Module):
    def __init__(self, in_dim=60, out_dim=135,
                 d_model=192, nhead=6, num_layers=5,
                 dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(in_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.output_proj(self.encoder(x))


def load_model_from_ckpt(path):
    """Carga el checkpoint detectando automáticamente d_model y num_layers."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state', ck.get('model_state_dict', ck.get('state_dict', ck)))

    # Auto-detectar arquitectura desde el state dict
    d_model    = sd['input_proj.weight'].shape[0]
    num_layers = sum(1 for k in sd
                     if re.match(r'encoder\.layers\.\d+\.norm1\.weight', k))
    dim_ff     = sd['encoder.layers.0.linear1.weight'].shape[0]

    # Elegir nhead: el especificado por el usuario si divide d_model, si no el primero válido
    for nh in [6, 4, 8, 3, 2, 1]:
        if d_model % nh == 0:
            nhead = nh
            break

    print(f'  → d_model={d_model}, num_layers={num_layers}, dim_ff={dim_ff}, nhead={nhead}')
    model = TransformerEncoderM2(d_model=d_model, nhead=nhead,
                                  num_layers=num_layers, dim_feedforward=dim_ff)
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model


# ── DATOS ─────────────────────────────────────────────────────────────────────
def load_data():
    d_tr  = np.load(TRAIN_NPZ, allow_pickle=True)
    stats = d_tr['statistics'].item()

    def _arr(sub, key):
        return np.array(stats[sub][key], dtype=np.float32)

    ori_mu  = _arr('orientation',  'mean_channel')
    ori_std = _arr('orientation',  'std_channel')
    acc_mu  = _arr('acceleration', 'mean_channel')
    acc_std = _arr('acceleration', 'std_channel')
    y_mu    = _arr('smpl_pose',    'mean_channel')
    y_std   = _arr('smpl_pose',    'std_channel')

    X_mu  = np.concatenate([ori_mu,  acc_mu])
    X_std = np.concatenate([ori_std, acc_std])
    X_std = np.where(X_std < 1e-8, 1.0, X_std)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    def make_windows(d, stride):
        Xw, yw = [], []
        for ori, acc, pose in zip(d['orientation'], d['acceleration'], d['smpl_pose']):
            ori  = np.array(ori,  dtype=np.float32)
            acc  = np.array(acc,  dtype=np.float32)
            pose = np.array(pose, dtype=np.float32)
            L    = len(ori)
            Xs   = (np.concatenate([ori, acc], axis=-1) - X_mu) / X_std
            ys   = (pose - y_mu) / y_std
            for s in range(0, L - T + 1, stride):
                Xw.append(Xs[s:s+T])
                yw.append(ys[s:s+T])
        return np.stack(Xw), np.stack(yw)

    print('  Ventanas entrenamiento (stride=30)...')
    X_tr, y_tr = make_windows(d_tr, stride=30)

    print('  Ventanas test (stride=T)...')
    d_te = np.load(TEST_NPZ, allow_pickle=True)
    X_te, y_te = make_windows(d_te, stride=T)

    print(f'  train X={X_tr.shape}  test X={X_te.shape}')
    return X_tr, y_tr, X_te, y_te, y_mu, y_std


def infer_center(model, X, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(DEVICE)
            out.append(model(xb)[:, T//2, :].cpu().numpy())
    return np.concatenate(out)


# ── FIGURA 1: barras agrupadas (números exactos, 5 modelos) ──────────────────
def fig6_comparativa():
    labels     = list(EXACT.keys())
    rmse_vals  = [EXACT[l]['RMSE']  for l in labels]
    nrmse_vals = [EXACT[l]['nRMSE'] for l in labels]
    r2_vals    = [EXACT[l]['R2']    for l in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        (rmse_vals,  'RMSE',     '#1f77b4', '%.4f'),
        (nrmse_vals, 'nRMSE (%)', '#ff7f0e', '%.2f'),
        (r2_vals,   'R²',        '#2ca02c', '%.3f'),
    ]
    for ax, (vals, metric, color, fmt) in zip(axes, specs):
        bars = ax.bar(x, vals, color=color, alpha=0.82,
                      edgecolor='white', linewidth=0.8)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=9)
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        offset = max(abs(max(vals)), 1e-4) * 0.025
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    fmt % v, ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    # Añadir leyenda de color por modelo (parches en la primera barra de cada subplot)
    for ax, (vals, *_) in zip(axes, specs):
        for rect, color in zip(ax.patches, PALETTE_5):
            rect.set_color(color)
            rect.set_alpha(0.82)

    fig.suptitle('Comparativa M1–M5 — RMSE · nRMSE (%) · R²',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig6_comparativa_5modelos.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardada: {p}')


# ── FIGURA 2: boxplot MAE real ────────────────────────────────────────────────
def fig_boxplot_mae(mae_m1, mae_m2, mae_m5):
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [mae_m1, mae_m2, mae_m5],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
        widths=0.5,
    )
    for patch, color in zip(bp['boxes'], PALETTE_3):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LABELS_3, fontsize=11)
    ax.set_ylabel('MAE (rad · matriz-R)', fontsize=11)
    ax.set_title('Distribución del MAE por modelo\n(evaluación ventana a ventana)',
                 fontsize=13, fontweight='bold')
    for i, ms in enumerate([mae_m1, mae_m2, mae_m5], 1):
        med = float(np.median(ms))
        ax.text(i, med, f'{med:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig_boxplot_mae.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardada: {p}')


# ── FIGURA 3: Bland-Altman M2 vs M5 ─────────────────────────────────────────
def _ba_ax(ax, pred, gt, title, color):
    mean_v = (pred.ravel() + gt.ravel()) / 2.0
    diff_v = pred.ravel() - gt.ravel()
    bias   = float(np.mean(diff_v))
    sd     = float(np.std(diff_v))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    ax.scatter(mean_v, diff_v, s=3, alpha=0.20, color=color, rasterized=True)
    ax.axhline(bias, color='black',   linewidth=1.4, linestyle='-',
               label=f'Bias = {bias:.4f}')
    ax.axhline(hi,   color='crimson', linewidth=1.2, linestyle='--',
               label=f'+1.96σ = {hi:.4f}')
    ax.axhline(lo,   color='crimson', linewidth=1.2, linestyle='--',
               label=f'−1.96σ = {lo:.4f}')
    ax.axhline(0,    color='gray',    linewidth=0.6, linestyle=':')
    ax.set_xlabel('Media (pred + real) / 2', fontsize=10)
    ax.set_ylabel('Diferencia (pred − real)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.annotate(
        f'Bias = {bias:.4f}\n±1.96σ = [{lo:.4f}, {hi:.4f}]',
        xy=(0.02, 0.05), xycoords='axes fraction', fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85),
    )


def fig_bland_altman(pred_m2, pred_m5, gt):
    fig, (ax2, ax5) = plt.subplots(1, 2, figsize=(14, 6))
    _ba_ax(ax2, pred_m2, gt, 'Bland-Altman — M2 Transformer',    PALETTE_3[1])
    _ba_ax(ax5, pred_m5, gt, 'Bland-Altman — M5 Fine-tune (K=20)', PALETTE_3[2])
    fig.suptitle('Análisis de Bland-Altman: acuerdo predicción vs. referencia',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig_bland_altman.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardada: {p}')


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    for style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid'):
        try:
            plt.style.use(style)
            break
        except Exception:
            pass
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13,
        'axes.labelsize': 11, 'figure.dpi': 100,
    })

    print(f'Dispositivo: {DEVICE}')
    print(f'Salida:      {OUT_DIR}\n')

    # ── Fig 6: usa números exactos, sin inferencia ───────────────────────────
    print('[1/3] fig6_comparativa_5modelos.png (números exactos)...')
    fig6_comparativa()

    # ── Cargar datos ─────────────────────────────────────────────────────────
    print('\nCargando datos...')
    X_tr, y_tr, X_te, y_te, y_mu, y_std = load_data()

    # Split test: K primeras ventanas → adaptación M5; resto → evaluación
    X_adapt   = X_te[:K_SHOTS]
    y_adapt_c = y_te[:K_SHOTS, T//2, :]        # (K, 135) normalizado
    X_eval    = X_te[K_SHOTS:]
    GT        = y_te[K_SHOTS:, T//2, :] * y_std + y_mu  # denormalizado

    # ── Cargar M2 ────────────────────────────────────────────────────────────
    print(f'\nCargando M2: {CKPT_M2}')
    model_m2 = load_model_from_ckpt(CKPT_M2)

    # ── Inferencia M2 ────────────────────────────────────────────────────────
    print('Inferencia M2...')
    pred_m2 = infer_center(model_m2, X_eval) * y_std + y_mu

    # ── M1 Ridge ─────────────────────────────────────────────────────────────
    print('Entrenando M1 Ridge...')
    m1 = Ridge(alpha=1.0)
    m1.fit(X_tr.reshape(len(X_tr), -1), y_tr[:, T//2, :])
    pred_m1 = m1.predict(X_eval.reshape(len(X_eval), -1)) * y_std + y_mu

    # ── M5 fine-tuning K=20 ──────────────────────────────────────────────────
    print('Fine-tuning M5 (K=20, 300 épocas)...')
    n_enc  = len(list(model_m2.encoder.layers))
    tags   = [f'encoder.layers.{n_enc-2}', f'encoder.layers.{n_enc-1}', 'output_proj']
    m5     = copy.deepcopy(model_m2)
    for name, p in m5.named_parameters():
        p.requires_grad_(any(t in name for t in tags))
    trainable = sum(p.numel() for p in m5.parameters() if p.requires_grad)
    print(f'  Parámetros entrenables: {trainable:,}')

    opt5 = torch.optim.Adam(filter(lambda p: p.requires_grad, m5.parameters()), lr=1e-4)
    mse  = nn.MSELoss()
    Xa   = torch.from_numpy(X_adapt).to(DEVICE)
    ya   = torch.from_numpy(y_adapt_c).to(DEVICE)

    m5.train()
    for ep in range(1, 301):
        opt5.zero_grad()
        loss = mse(m5(Xa)[:, T//2, :], ya)
        loss.backward()
        opt5.step()
        if ep % 100 == 0:
            print(f'  ep {ep:3d}/300  loss={loss.item():.6f}')

    pred_m5 = infer_center(m5, X_eval) * y_std + y_mu

    # ── MAE por ventana ───────────────────────────────────────────────────────
    mae_m1 = np.mean(np.abs(pred_m1 - GT), axis=1)
    mae_m2 = np.mean(np.abs(pred_m2 - GT), axis=1)
    mae_m5 = np.mean(np.abs(pred_m5 - GT), axis=1)
    print(f'\nMAE mediana  M1={np.median(mae_m1):.4f}  '
          f'M2={np.median(mae_m2):.4f}  M5={np.median(mae_m5):.4f}')

    # ── Fig 2: boxplot ────────────────────────────────────────────────────────
    print('\n[2/3] fig_boxplot_mae.png...')
    fig_boxplot_mae(mae_m1, mae_m2, mae_m5)

    # ── Fig 3: Bland-Altman ───────────────────────────────────────────────────
    print('\n[3/3] fig_bland_altman.png...')
    fig_bland_altman(pred_m2, pred_m5, GT)

    print('\n✓ Listo.')


if __name__ == '__main__':
    main()
