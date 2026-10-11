"""
Figure 3 — C_i Distribution (Publication Quality)
===================================================
Wasim Ahmad, BIT

Extracts C_i scores from the best CMPI checkpoint on the test set
and generates a publication-quality distribution figure showing
the separation between real and fake samples.

Layout:
  Left  : Overlapping KDE + histogram (real vs fake)
  Right : Box-and-whisker comparison
  Bottom: ROC curve using C_i as score

Usage
-----
python generate_fig3_ci_distribution.py \
    --checkpoint  results/training_v2/best_model.pt \
    --metadata_csv AMDD/data/metadata.csv \
    --preprocessed_dir AMDD/data/preprocessed \
    --output figures/fig3_ci_distribution.pdf
"""

import sys, argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── AMDD imports ──────────────────────────────────────────────────────────
AMDD_ROOT = Path(__file__).resolve().parent.parent / "AMDD"
for _p in [AMDD_ROOT, AMDD_ROOT/"models", AMDD_ROOT/"data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from encoders import VisualEncoder, AudioEncoder
except ModuleNotFoundError:
    from models.encoders import VisualEncoder, AudioEncoder
try:
    from dataset import DeepfakeDataset
except ModuleNotFoundError:
    sys.path.insert(0, str(AMDD_ROOT/"training"))
    from dataset import DeepfakeDataset

# ── Publication style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     9.5,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8.5,
    'text.usetex':        False,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.03,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.7,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
})

COL_REAL = '#2E7D32'
COL_FAKE = '#C62828'
ALPHA    = 0.45


# ── Model (must match train_ccmpm.py exactly) ─────────────────────────────
class CrossModalPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class DetectionHead(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class CMPIModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim, dropout=dropout)
        self.audio_encoder  = AudioEncoder(embed_dim=embed_dim,  dropout=dropout)
        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.register_buffer('ci_running_mean', torch.tensor(1.0))
        self.register_buffer('ci_running_std',  torch.tensor(1.0))
        self.register_buffer('ci_n_updates',    torch.tensor(0))
        self.detection_head = DetectionHead(embed_dim * 2 + 1, dropout)

    def forward(self, frames, audio):
        z_v    = self.visual_encoder(frames)['video_embedding']
        z_a    = self.audio_encoder(audio)['audio_embedding']
        pred_v = self.T_av(z_a)
        pred_a = self.T_va(z_v)
        err_av = torch.norm(pred_v - z_v, dim=1)
        err_va = torch.norm(pred_a - z_a, dim=1)
        C_i    = err_av + err_va
        C_i_norm = ((C_i - self.ci_running_mean) /
                    self.ci_running_std.clamp(min=1e-6)).unsqueeze(1)
        det_input  = torch.cat([z_v, z_a, C_i_norm], dim=1)
        det_logits = self.detection_head(det_input)
        return {'C_i': C_i, 'det_probs': torch.sigmoid(det_logits),
                'err_av': err_av, 'err_va': err_va}


@torch.no_grad()
def extract_scores(model, loader, device):
    model.eval()
    ci_list, prob_list, label_list = [], [], []
    av_list, va_list = [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label']
        out    = model(frames, audio)
        ci_list.append(out['C_i'].cpu().numpy())
        prob_list.append(out['det_probs'].cpu().numpy())
        av_list.append(out['err_av'].cpu().numpy())
        va_list.append(out['err_va'].cpu().numpy())
        label_list.append(labels.numpy())
    return (np.concatenate(ci_list),
            np.concatenate(prob_list),
            np.concatenate(av_list),
            np.concatenate(va_list),
            np.concatenate(label_list))


def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = CMPIModel().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')})")

    # Load test data
    test_ds = DeepfakeDataset(args.metadata_csv, 'test', args.preprocessed_dir)
    loader  = DataLoader(test_ds, batch_size=8, shuffle=False,
                         num_workers=4, pin_memory=True)

    print("Extracting C_i scores...")
    ci, probs, err_av, err_va, labels = extract_scores(model, loader, device)

    real_ci = ci[labels == 0]
    fake_ci = ci[labels == 1]
    real_av = err_av[labels == 0]; fake_av = err_av[labels == 1]
    real_va = err_va[labels == 0]; fake_va = err_va[labels == 1]

    auroc    = roc_auc_score(labels, ci)
    fpr, tpr, _ = roc_curve(labels, ci)
    mw_stat, mw_p = mannwhitneyu(fake_ci, real_ci, alternative='greater')
    sep = fake_ci.mean() / real_ci.mean()

    print(f"  Real C_i: {real_ci.mean():.2f} ± {real_ci.std():.2f}")
    print(f"  Fake C_i: {fake_ci.mean():.2f} ± {fake_ci.std():.2f}")
    print(f"  Separation: {sep:.2f}×  AUROC: {auroc:.4f}  p={mw_p:.2e}")

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.16, 4.8))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            width_ratios=[2.2, 1.0, 1.4],
                            height_ratios=[1, 1],
                            hspace=0.42, wspace=0.38,
                            left=0.08, right=0.97,
                            top=0.91, bottom=0.10)

    ax_kde  = fig.add_subplot(gs[:, 0])   # left: full height KDE
    ax_box  = fig.add_subplot(gs[0, 1])   # top middle: boxplot
    ax_arm  = fig.add_subplot(gs[1, 1])   # bottom middle: per-arm
    ax_roc  = fig.add_subplot(gs[:, 2])   # right: full height ROC

    # ── KDE + Histogram ───────────────────────────────────────────────────
    all_ci  = np.concatenate([real_ci, fake_ci])
    ci_min  = max(0, all_ci.min() - 2)
    ci_max  = all_ci.max() + 5
    x_grid  = np.linspace(ci_min, ci_max, 300)

    # Histograms
    bins = np.linspace(ci_min, ci_max, 35)
    ax_kde.hist(real_ci, bins=bins, density=True, alpha=ALPHA,
                color=COL_REAL, label='Real', zorder=2)
    ax_kde.hist(fake_ci, bins=bins, density=True, alpha=ALPHA,
                color=COL_FAKE, label='Fake', zorder=2)

    # KDE curves
    kde_real = gaussian_kde(real_ci, bw_method=0.3)
    kde_fake = gaussian_kde(fake_ci, bw_method=0.3)
    ax_kde.plot(x_grid, kde_real(x_grid), color=COL_REAL, lw=2.0, zorder=3)
    ax_kde.plot(x_grid, kde_fake(x_grid), color=COL_FAKE, lw=2.0, zorder=3)

    # Fill under KDE
    ax_kde.fill_between(x_grid, kde_real(x_grid), alpha=0.12,
                        color=COL_REAL, zorder=1)
    ax_kde.fill_between(x_grid, kde_fake(x_grid), alpha=0.12,
                        color=COL_FAKE, zorder=1)

    # Mean lines
    ax_kde.axvline(real_ci.mean(), color=COL_REAL, lw=1.2, ls='--',
                   alpha=0.85, zorder=4)
    ax_kde.axvline(fake_ci.mean(), color=COL_FAKE, lw=1.2, ls='--',
                   alpha=0.85, zorder=4)

    # Mean annotations at bottom of dashed lines
    ax_kde.text(real_ci.mean(), 0.001,
                f'$\\mu$={real_ci.mean():.1f}',
                ha='center', va='bottom', fontsize=7.5,
                color=COL_REAL, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8))
    ax_kde.text(fake_ci.mean(), 0.001,
                f'$\\mu$={fake_ci.mean():.1f}',
                ha='center', va='bottom', fontsize=7.5,
                color=COL_FAKE, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8))

    ax_kde.set_xlabel('Predictive Inconsistency Score  $C_i$', fontsize=9)
    ax_kde.set_ylabel('Density', fontsize=9)
    ax_kde.set_title('$C_i$ Distribution — Real vs Fake (test set)', fontsize=9.5, pad=4)
    ax_kde.legend(frameon=False, loc='upper right', ncol=1,
                  handlelength=1.2, handletextpad=0.4)

    # Stats annotation in the gap between the two peaks
    ax_kde.text(0.42, 0.68,
                f'Separation: {sep:.2f}\u00d7\n'
                f'$C_i$ AUROC: {auroc:.4f}\n'
                f'$p$ < {mw_p:.0e}',
                transform=ax_kde.transAxes,
                ha='center', va='center', fontsize=7.5,
                color='#424242',
                bbox=dict(boxstyle='round,pad=0.35', fc='#FAFAFA',
                          ec='#BDBDBD', lw=0.6))

    # ── Boxplot ───────────────────────────────────────────────────────────
    bp = ax_box.boxplot([real_ci, fake_ci],
                        tick_labels=['Real', 'Fake'],
                        patch_artist=True, notch=True, widths=0.5,
                        medianprops=dict(color='white', lw=2),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        flierprops=dict(marker='o', markersize=2,
                                        alpha=0.4, lw=0))
    bp['boxes'][0].set_facecolor(COL_REAL); bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_facecolor(COL_FAKE); bp['boxes'][1].set_alpha(0.75)
    bp['boxes'][0].set_edgecolor(COL_REAL)
    bp['boxes'][1].set_edgecolor(COL_FAKE)
    bp['fliers'][0].set_markerfacecolor(COL_REAL)
    bp['fliers'][1].set_markerfacecolor(COL_FAKE)

    ax_box.set_ylabel('$C_i$', fontsize=9)
    ax_box.set_title('Score distribution', fontsize=8.5)

    # ── Per-arm bar chart ─────────────────────────────────────────────────
    arms   = ['$T_{a\\rightarrow v}$', '$T_{v\\rightarrow a}$']
    r_vals = [real_av.mean(), real_va.mean()]
    f_vals = [fake_av.mean(), fake_va.mean()]
    x      = np.array([0, 1])
    w      = 0.32

    ax_arm.bar(x - w/2, r_vals, w, color=COL_REAL, alpha=0.8,
               label='Real', zorder=3)
    ax_arm.bar(x + w/2, f_vals, w, color=COL_FAKE, alpha=0.8,
               label='Fake', zorder=3)
    ax_arm.set_xticks(x); ax_arm.set_xticklabels(arms, fontsize=8)
    ax_arm.set_ylabel('Mean error', fontsize=8)
    ax_arm.set_title('Per-arm errors', fontsize=8.5)
    ax_arm.legend(frameon=False, fontsize=7, loc='upper left')
    ax_arm.grid(axis='y', alpha=0.3, lw=0.5)

    # ── ROC Curve ─────────────────────────────────────────────────────────
    ax_roc.plot(fpr, tpr, color='#1565C0', lw=2.0,
                label=f'C_i score\n(AUC={auroc:.4f})')
    ax_roc.fill_between(fpr, tpr, alpha=0.08, color='#1565C0')
    ax_roc.plot([0,1],[0,1],'--', color='#9E9E9E', lw=0.8, label='Random')

    # Mark operating point at max Youden's J
    j_scores = tpr - fpr
    j_idx    = np.argmax(j_scores)
    ax_roc.plot(fpr[j_idx], tpr[j_idx], 'o',
                color='#AD1457', markersize=5, zorder=5,
                label=f'Best op. point\n(FPR={fpr[j_idx]:.3f})')

    ax_roc.set_xlabel('False Positive Rate', fontsize=9)
    ax_roc.set_ylabel('True Positive Rate', fontsize=9)
    ax_roc.set_title('ROC — Unsupervised $C_i$', fontsize=9.5)
    ax_roc.legend(frameon=True, fontsize=7, loc='lower right',
                  framealpha=0.9, edgecolor='#BDBDBD')
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.set_aspect('equal')

    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(
        'Predictive Inconsistency as a Forensic Signal  '
        '(FakeAVCeleb test set,  n=220)',
        fontsize=9, y=0.975, color='#212121'
    )

    # ── Save ─────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format='pdf', dpi=300)
    print(f"PDF: {out}")
    png = out.with_suffix('.png')
    fig.savefig(str(png), format='png', dpi=300)
    print(f"PNG: {png}")
    plt.close(fig)

    # Save scores for reuse
    np.save(str(out.parent/'test_ci_scores.npy'), ci)
    np.save(str(out.parent/'test_labels.npy'), labels)
    print("Scores saved.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',
                   default='results/training_v2/best_model.pt')
    p.add_argument('--metadata_csv',
                   default='AMDD/data/metadata.csv')
    p.add_argument('--preprocessed_dir',
                   default='AMDD/data/preprocessed')
    p.add_argument('--output',
                   default='figures/fig3_ci_distribution.pdf')
    return p.parse_args()


if __name__ == '__main__':
    generate(parse_args())