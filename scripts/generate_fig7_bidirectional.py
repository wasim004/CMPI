"""
Figure 7 — Forward/Backward Error Non-Redundancy Analysis
Usage
-----
python generate_fig7_bidirectional.py \
    --data_dir results/further_experiments \
    --output   figures/fig7_bidirectional.pdf
"""

import argparse
from pathlib import Path
import numpy as np
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
    'savefig.pad_inches': 0.05,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.7,
})

C_REAL = '#1565C0'
C_FAKE = '#C62828'
GEN_ORDER = ['real', 'faceswap', 'wav2lip', 'fsgan']
GEN_COLORS = {'real': '#2E7D32', 'faceswap': '#C62828',
              'wav2lip': '#EF6C00', 'fsgan': '#6A1B9A'}


def generate(args):
    d = Path(args.data_dir)
    eps_f = np.load(d / 'eps_f.npy')
    eps_b = np.load(d / 'eps_b.npy')
    labels = np.load(d / 'labels.npy')
    gen_id = np.load(d / 'generator_id.npy')
    gen_names = {0: 'real', 1: 'faceswap', 2: 'wav2lip', 3: 'fsgan'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4.0),
                                    gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.38})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.14)

    # ── Left panel: eps_f vs eps_b scatter, real vs fake ────────────────────
    real_m, fake_m = labels == 0, labels == 1
    ax1.scatter(eps_f[real_m], eps_b[real_m], s=14, c=C_REAL, alpha=0.7,
                label=f'Real (n={real_m.sum()})', zorder=3, edgecolors='none')
    ax1.scatter(eps_f[fake_m], eps_b[fake_m], s=14, c=C_FAKE, alpha=0.55,
                label=f'Fake (n={fake_m.sum()})', zorder=3, edgecolors='none')
    lims = [0, max(eps_f.max(), eps_b.max()) * 1.05]
    ax1.plot(lims, lims, ls='--', color='#9E9E9E', lw=0.9, zorder=1,
             label='$\\varepsilon_f = \\varepsilon_b$ (symmetric)')
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel('$\\varepsilon_f$ (forward error)')
    ax1.set_ylabel('$\\varepsilon_b$ (backward error)')
    ax1.set_title('Forward vs. backward error\n(low within-class correlation)', fontsize=9)
    ax1.legend(frameon=True, framealpha=0.95, edgecolor='#DDDDDD',
               fontsize=7, loc='upper left', handlelength=1.2)
    ax1.grid(alpha=0.2, lw=0.5, zorder=0)
    ax1.text(0.97, 0.06, 'r(real)=0.10 (n.s.)\nr(fake)=0.32 (p<1e-4)',
              transform=ax1.transAxes, ha='right', va='bottom', fontsize=7,
              color='#424242',
              bbox=dict(fc='white', ec='#DDDDDD', boxstyle='round,pad=0.3'))

    # ── Right panel: per-generator signed asymmetry A ───────────────────────
    x = np.arange(len(GEN_ORDER))
    means, stds = [], []
    for g in GEN_ORDER:
        gid = {v: k for k, v in gen_names.items()}[g]
        m = gen_id == gid
        A = eps_f[m] - eps_b[m]
        means.append(A.mean()); stds.append(A.std())

    cols = [GEN_COLORS[g] for g in GEN_ORDER]
    bars = ax2.bar(x, means, 0.55, yerr=stds, capsize=3, color=cols, alpha=0.85,
                    zorder=3, error_kw=dict(lw=0.9, ecolor='#616161'))
    ax2.axhline(0.0, color='#616161', lw=0.8, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Real', 'FaceSwap\n(video)', 'Wav2Lip\n(video)', 'FSGAN\n(video)'],
                         fontsize=7.6)
    ax2.set_ylabel('$\\mathcal{A} = \\varepsilon_f - \\varepsilon_b$  (mean $\\pm$ std)')
    ax2.set_title('Asymmetry by manipulation type\n(all FakeAVCeleb generators are visual)', fontsize=9)
    ax2.grid(axis='y', alpha=0.2, lw=0.5, zorder=0)

    fig.suptitle('Non-Redundancy of Bidirectional Prediction Errors  (FakeAVCeleb test set, n=220)',
                 fontsize=10, y=0.995, color='#1A1A1A')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format='pdf', dpi=300)
    png = out.with_suffix('.png')
    fig.savefig(str(png), format='png', dpi=300)
    print(f"PDF: {out}\nPNG: {png}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='results/further_experiments')
    p.add_argument('--output', default='figures/fig7_bidirectional.pdf')
    return p.parse_args()


if __name__ == '__main__':
    generate(parse_args())
