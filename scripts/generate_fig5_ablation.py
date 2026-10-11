"""
Figure 5 — Ablation Study (final clean version)
Usage
-----
python generate_fig5_ablation.py \
    --results_json results/ablation/ablation_results.json \
    --output       figures/fig5_ablation.pdf
"""

import argparse, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

LABELS = [
    'Full\nCMPI',
    'w/o\n$T_{a\\rightarrow v}$',
    'w/o\n$T_{v\\rightarrow a}$',
    'w/o\n$\\mathcal{C}_i$ input',
    'w/o pred. loss\n($\\lambda$=0)',
]
ORDER = ['full','no_T_av','no_T_va','no_ci_input','no_pred_loss']

# Distinct colours — no overlap
C_DET = '#1565C0'   # blue solid  — detection AUROC
C_CI  = '#E65100'   # orange hatched — C_i AUROC
C_SEP_FULL = '#2E7D32'  # green — full model sep bar
C_SEP_NORM = '#78909C'  # grey  — other sep bars
C_SEP_BAD  = '#C62828'  # red   — no_pred_loss sep bar


def generate(args):
    with open(args.results_json) as f:
        results = {r['variant']: r for r in json.load(f)}

    aurocs  = [results[v]['test']['auroc']     for v in ORDER]
    ci_aucs = [results[v]['test']['ci_auroc']  for v in ORDER]
    seps    = [results[v]['test']['separation'] for v in ORDER]

    x = np.arange(len(ORDER))
    w = 0.30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6),
                                    gridspec_kw={'width_ratios':[1.65, 1],
                                                 'wspace': 0.40})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18)

    # ── Left panel: two grouped bar series ───────────────────────────────
    b1 = ax1.bar(x - w/2, aurocs,  w, color=C_DET, alpha=0.90,
                 label='Detection AUROC', zorder=3)
    b2 = ax1.bar(x + w/2, ci_aucs, w, color=C_CI,  alpha=0.85,
                 label='$\\mathcal{C}_i$ AUROC (unsup.)',
                 zorder=3, hatch='...', edgecolor='#BF360C')

    # Value labels — small, above each bar
    for bar, val in zip(b1, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 min(bar.get_height() + 0.005, 1.03),
                 f'{val:.4f}', ha='center', va='bottom',
                 fontsize=5.8, color='#212121', rotation=90)

    for bar, val in zip(b2, ci_aucs):
        ypos = bar.get_height() + 0.005
        ax1.text(bar.get_x() + bar.get_width()/2,
                 min(ypos, 1.03),
                 f'{val:.4f}', ha='center', va='bottom',
                 fontsize=5.8, color='#212121', rotation=90)

    # Annotate the C_i collapse
    npred = ORDER.index('no_pred_loss')
    ax1.annotate(
        '$\\mathcal{C}_i$ collapses\n(0.0665)',
        xy=(x[npred] + w/2, ci_aucs[npred] + 0.03),
        xytext=(x[npred] - 0.85, 0.42),
        fontsize=7, color='#880E4F',
        arrowprops=dict(arrowstyle='->', color='#880E4F',
                        lw=0.9, mutation_scale=8),
        ha='center')

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, fontsize=7.8)
    ax1.set_ylabel('AUROC', fontsize=9)
    ax1.set_ylim(0.0, 1.28)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.axhline(1.0, color='#BDBDBD', lw=0.6, ls='--', zorder=1)
    ax1.grid(axis='y', alpha=0.2, lw=0.5, zorder=0)
    ax1.set_title('Detection and $\\mathcal{C}_i$ AUROC per variant',
                  fontsize=9.5, pad=5)
    ax1.legend(frameon=True, framealpha=0.95, edgecolor='#DDDDDD',
               fontsize=7.5, loc='upper right',
               handlelength=1.4, handletextpad=0.5)

    # ── Right panel: separation ratio bars ───────────────────────────────
    sep_cols = [C_SEP_FULL if v=='full' else
                C_SEP_BAD  if v=='no_pred_loss' else
                C_SEP_NORM for v in ORDER]

    b3 = ax2.bar(x, seps, 0.52, color=sep_cols, alpha=0.88, zorder=3)

    for bar, val in zip(b3, seps):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.06,
                 f'{val:.2f}\u00d7',
                 ha='center', va='bottom', fontsize=7.5, color='#212121')

    ax2.axhline(1.0, color='#C62828', lw=1.0, ls='--', zorder=2,
                label='No separation (1\u00d7)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, fontsize=7.8)
    ax2.set_ylabel('$\\mathcal{C}_i$ separation  (fake/real mean)',
                   fontsize=8.5)
    ax2.set_ylim(0, max(seps)*1.30)
    ax2.grid(axis='y', alpha=0.2, lw=0.5, zorder=0)
    ax2.set_title('$\\mathcal{C}_i$ Separation ratio', fontsize=9.5, pad=5)
    ax2.legend(frameon=False, fontsize=7.5, loc='upper right')

    # Colour legend for sep bars
    leg_els = [
        Patch(fc=C_SEP_FULL, label='Full CMPI'),
        Patch(fc=C_SEP_NORM, label='Ablated'),
        Patch(fc=C_SEP_BAD,  label='No pred. loss'),
    ]
    ax2.legend(handles=leg_els, frameon=False, fontsize=7,
               loc='upper right', handlelength=1.0)

    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle('Ablation Study — FakeAVCeleb Test Set  (n=220)',
                 fontsize=10, y=0.98, color='#1A1A1A')

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
    p.add_argument('--results_json', default='results/ablation/ablation_results.json')
    p.add_argument('--output',       default='figures/fig5_ablation.pdf')
    return p.parse_args()

if __name__ == '__main__':
    generate(parse_args())