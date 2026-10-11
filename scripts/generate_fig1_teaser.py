"""
Figure 1 — CMPI Teaser Figure (v2)
=====================================
Wasim Ahmad, BIT

Fixes from v1:
  - Coupling arrows stop before mel column (no crossing)
  - Coupling strip height reduced
  - Mel spectrogram height matches frame height exactly
  - Row labels aligned to row centres
  - Bottom caption moved down with proper padding
  - Time/Freq axis labels removed from mel (clean look)
  - Title size reduced for balance
  - C_i box proportions fixed

Usage
-----
python generate_fig1_teaser.py \
    --samples_dir  figures/raw_samples \
    --ci_real      13.85 \
    --ci_fake      39.42 \
    --output       figures/fig1_teaser.pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          9,
    'axes.labelsize':     8,
    'axes.titlesize':     8,
    'text.usetex':        False,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

COL_REAL  = '#2E7D32'
COL_FAKE  = '#C62828'
COL_AV    = '#1565C0'
COL_VA    = '#6A1B9A'
COL_GREY  = '#9E9E9E'


def load_frames(samples_dir, label, n=4):
    arr     = np.load(str(Path(samples_dir) / f"{label}_frames.npy"))
    indices = np.linspace(0, arr.shape[0] - 1, n, dtype=int)
    return [arr[i] for i in indices]


def load_mel(samples_dir, label):
    return np.load(str(Path(samples_dir) / f"{label}_mel.npy"))


def draw_ci_box(ax, value, is_real):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    col   = COL_REAL if is_real else COL_FAKE
    arrow = '↓' if is_real else '↑'
    word  = 'Low' if is_real else 'High'

    box = FancyBboxPatch((0.08, 0.06), 0.84, 0.88,
                         boxstyle="round,pad=0.04",
                         facecolor='#FAFAFA',
                         edgecolor=col, linewidth=1.8)
    ax.add_patch(box)

    # C_i symbol
    ax.text(0.50, 0.80, r'$\mathcal{C}_i$',
            ha='center', va='center',
            fontsize=10, color='#424242', fontweight='bold')
    # Divider
    ax.plot([0.18, 0.82], [0.65, 0.65], color=COL_GREY,
            lw=0.6, transform=ax.transAxes)
    # Numeric value
    ax.text(0.50, 0.43, f'{value:.1f}',
            ha='center', va='center',
            fontsize=15, color=col, fontweight='bold')
    # Arrow + word
    ax.text(0.50, 0.18, f'{arrow} {word}',
            ha='center', va='center',
            fontsize=7.5, color=col, fontstyle='italic')


def generate(args):
    sd       = Path(args.samples_dir)
    n_frames = 4

    real_frames = load_frames(sd, 'real', n_frames)
    fake_frames = load_frames(sd, 'fake', n_frames)
    real_mel    = load_mel(sd, 'real')
    fake_mel    = load_mel(sd, 'fake')

    # ── Layout ────────────────────────────────────────────────────────────
    # 3 rows: real | coupling | fake
    # 7 cols: f0 f1 f2 f3 | gap | mel | Ci
    # Key fix: mel column width matches frame columns exactly

    fig = plt.figure(figsize=(7.16, 3.6))   # double-column width

    gs = gridspec.GridSpec(
        3, 7,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 0.04, 1.05, 0.62],
        height_ratios=[1, 0.13, 1],
        hspace=0.0,
        wspace=0.05,
        left=0.055, right=0.985,
        top=0.89, bottom=0.10,
    )

    # ── Helper: add coloured border to axes ───────────────────────────────
    def colour_border(ax, col, lw=1.4):
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(col)
            sp.set_linewidth(lw)

    # ── Real frames ───────────────────────────────────────────────────────
    for i, frame in enumerate(real_frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        colour_border(ax, COL_REAL)
        ax.set_title(f'$t_{{{i+1}}}$', fontsize=7,
                     color='#757575', pad=2)

    # Real mel — no axis labels, clean border
    ax_mr = fig.add_subplot(gs[0, 5])
    ax_mr.imshow(real_mel, aspect='auto', origin='lower',
                 cmap='magma', interpolation='bilinear')
    ax_mr.set_xticks([]); ax_mr.set_yticks([])
    colour_border(ax_mr, COL_REAL)
    ax_mr.set_title('Mel spectrogram', fontsize=7,
                    color='#757575', pad=2)

    # Real C_i
    ax_cr = fig.add_subplot(gs[0, 6])
    draw_ci_box(ax_cr, args.ci_real, is_real=True)

    # ── Coupling strip ────────────────────────────────────────────────────
    ax_mid = fig.add_subplot(gs[1, :5])   # spans only frames + gap cols
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis('off')

    # Horizontal separator lines
    ax_mid.axhline(1.0, color='#E0E0E0', lw=0.7)
    ax_mid.axhline(0.0, color='#E0E0E0', lw=0.7)

    kw_av = dict(arrowstyle='->', color=COL_AV,
                 lw=0.9, mutation_scale=9)
    kw_va = dict(arrowstyle='->', color=COL_VA,
                 lw=0.9, mutation_scale=9)

    # T_{a→v}: left to right (top half of strip)
    ax_mid.annotate('', xy=(0.97, 0.78), xytext=(0.03, 0.78),
                    xycoords='axes fraction',
                    arrowprops=kw_av)
    ax_mid.text(0.50, 0.82,
                r'$T_{a \rightarrow v}$: audio $\rightarrow$ visual prediction',
                ha='center', va='bottom', fontsize=6.8,
                color=COL_AV, transform=ax_mid.transAxes)

    # T_{v→a}: right to left (bottom half of strip)
    ax_mid.annotate('', xy=(0.03, 0.22), xytext=(0.97, 0.22),
                    xycoords='axes fraction',
                    arrowprops=kw_va)
    ax_mid.text(0.50, 0.18,
                r'$T_{v \rightarrow a}$: visual $\rightarrow$ audio prediction',
                ha='center', va='top', fontsize=6.8,
                color=COL_VA, transform=ax_mid.transAxes)

    # ── Fake frames ───────────────────────────────────────────────────────
    for i, frame in enumerate(fake_frames):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        colour_border(ax, COL_FAKE)

    # Fake mel
    ax_mf = fig.add_subplot(gs[2, 5])
    ax_mf.imshow(fake_mel, aspect='auto', origin='lower',
                 cmap='magma', interpolation='bilinear')
    ax_mf.set_xticks([]); ax_mf.set_yticks([])
    colour_border(ax_mf, COL_FAKE)

    # Fake C_i
    ax_cf = fig.add_subplot(gs[2, 6])
    draw_ci_box(ax_cf, args.ci_fake, is_real=False)

    # ── Row labels (left of figure) ───────────────────────────────────────
    # Computed from gs positions — placed in figure fraction coordinates
    # Real row centre ≈ top - (top-coupling_top)/2
    fig.text(0.024, 0.720, 'Real',
             ha='center', va='center', rotation=90,
             fontsize=8.5, fontweight='bold', color=COL_REAL)
    fig.text(0.024, 0.235, 'Fake',
             ha='center', va='center', rotation=90,
             fontsize=8.5, fontweight='bold', color=COL_FAKE)

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        'Bidirectional cross-modal predictive coupling as a forensic signal',
        fontsize=8.5, fontweight='bold', y=0.97, color='#212121'
    )

    # ── Bottom caption ────────────────────────────────────────────────────
    fig.text(
        0.50, 0.015,
        (r'Real pairs: strong coupling ($\mathcal{C}_i \approx$'
         f' {args.ci_real:.1f})'
         r';  Deepfakes: disrupted coupling ($\mathcal{C}_i \approx$'
         f' {args.ci_fake:.1f})'),
        ha='center', va='bottom', fontsize=7, color='#424242'
    )

    # ── Save ─────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(out), format='pdf', dpi=300)
    print(f"Saved PDF: {out}")

    png = out.with_suffix('.png')
    fig.savefig(str(png), format='png', dpi=300)
    print(f"Saved PNG: {png}")

    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir', default='figures/raw_samples')
    p.add_argument('--ci_real',     type=float, default=13.85)
    p.add_argument('--ci_fake',     type=float, default=39.42)
    p.add_argument('--output',      default='figures/fig1_teaser.pdf')
    return p.parse_args()


if __name__ == '__main__':
    generate(parse_args())