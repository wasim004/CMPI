"""
Figure 4 — Training Curves
===========================
Wasim Ahmad, BIT

Shows the interesting training dynamics from history.json:
  Left : Predictor loss (train + val) — drops as predictors learn coupling
  Middle: C_i AUROC (val) — rises as C_i becomes discriminative
  Right : Balanced accuracy (val) — classification performance

Detection AUROC is 1.0 from epoch 1 (warm-start) so not shown.

Usage
-----
python generate_fig4_training_curves.py \
    --history_json results/training_v2/history.json \
    --output       figures/fig4_training_curves.pdf
"""

import argparse, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'text.usetex':       False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.04,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
})

C_TRAIN = '#1565C0'   # blue  — train
C_VAL   = '#C62828'   # red   — val
C_CI    = '#2E7D32'   # green — C_i AUROC
C_PRED  = '#6A1B9A'   # purple — pred loss
C_SHADE = 0.10        # fill alpha


def generate(args):
    with open(args.history_json) as f:
        h = json.load(f)

    epochs = np.arange(1, len(h['train_loss']) + 1)

    train_pred = np.array(h['train_pred_loss'])
    val_pred   = np.array(h['val_pred_loss'])
    val_ci_auc = np.array(h['val_ci_auroc'])
    val_auroc  = np.array(h['val_auroc'])
    val_bal    = np.array(h['val_bal_acc'])
    train_bal  = np.array(h['train_bal_acc'])

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9))
    axes = list(axes)
    fig.subplots_adjust(left=0.09, right=0.97,
                        top=0.85, bottom=0.17, wspace=0.42)

    # ── Panel 1: Predictor loss ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_pred, color=C_TRAIN, lw=1.6,
            label='Train', zorder=3)
    ax.plot(epochs, val_pred,   color=C_VAL,   lw=1.6,
            label='Val',   zorder=3, ls='--')
    ax.fill_between(epochs, train_pred, val_pred,
                    alpha=C_SHADE, color='#888888')

    # Mark convergence point
    conv_ep = np.argmin(val_pred) + 1
    ax.axvline(conv_ep, color='#888888', lw=0.7, ls=':', zorder=2)
    ax.text(conv_ep + 0.5, val_pred.max()*0.85,
            f'Min val\n@ep {conv_ep}',
            fontsize=6.5, color='#555555')

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('MSE Loss', fontsize=9)
    ax.set_title('Predictor Loss  $\\mathcal{L}_{pred}$',
                 fontsize=9.5, pad=4)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_xlim(1, len(epochs))

    # ── Panel 2: C_i AUROC ───────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, val_ci_auc, color=C_CI, lw=1.8,
            label='$\\mathcal{C}_i$ AUROC (val)', zorder=3)
    ax.fill_between(epochs, val_ci_auc, val_ci_auc.min(),
                    alpha=0.12, color=C_CI)

    # Mark where C_i AUROC first exceeds 0.999
    thresh = 0.999
    cross  = np.where(val_ci_auc >= thresh)[0]
    if len(cross):
        ep_cross = epochs[cross[0]]
        ax.axvline(ep_cross, color='#888888', lw=0.7, ls=':', zorder=2)
        ax.text(ep_cross + 0.5,
                val_ci_auc.min() + (val_ci_auc.max()-val_ci_auc.min())*0.12,
                f'>{thresh}\n@ep {ep_cross}',
                fontsize=6.5, color='#555555')

    ax.axhline(1.0, color='#BDBDBD', lw=0.6, ls='--', zorder=1)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('AUROC', fontsize=9)
    ax.set_title('$\\mathcal{C}_i$ AUROC (unsupervised)',
                 fontsize=9.5, pad=4)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_xlim(1, len(epochs))
    # Set y range to show variation
    ymin = max(0, val_ci_auc.min() - 0.01)
    ax.set_ylim(ymin, 1.005)



    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(
        'Training Dynamics — CMPI  (FakeAVCeleb, 50 epochs, warm-start from AMDD)',
        fontsize=9, y=0.98, color='#1A1A1A')

    # ── Save ─────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format='pdf', dpi=300)
    png = out.with_suffix('.png')
    fig.savefig(str(png), format='png', dpi=300)
    print(f"PDF: {out}\nPNG: {png}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--history_json',
                   default='results/training_v2/history.json')
    p.add_argument('--output',
                   default='figures/fig4_training_curves.pdf')
    return p.parse_args()

if __name__ == '__main__':
    generate(parse_args())