"""
Figure 6 — GradCAM++ Publication Figure (hand-picked samples)
==============================================================
Wasim Ahmad, BIT

Uses pre-computed overlays to assemble the final publication figure.
Hand-picked samples:
  Real: real_00 (compact mouth), real_01 (bilateral nose/cheek)
  Fake: fake_03 (split mouth+temple), fake_08 (bilateral eyes)

Layout: 2×2 grid, each cell = [original | overlay]
Rows: Real (top) | Fake (bottom)

Usage
-----
python generate_fig6_final.py \
    --gradcam_dir figures/gradcam \
    --output      figures/fig6_gradcam.pdf
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          9,
    'text.usetex':        False,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.04,
})

COL_REAL = '#2E7D32'
COL_FAKE = '#C62828'

# Hand-picked sample files and their descriptions
SAMPLES = [
    # (file_prefix, label, ci_val, p_val, description)
    ('real_00', 'Real', 10.8, 0.109, 'Compact mouth-region\nattention'),
    ('real_01', 'Real', 12.6, 0.100, 'Bilateral nose/cheek\nattention'),
    ('fake_03', 'Fake', 42.4, 0.980, 'Split activation:\nmouth + temple boundary'),
    ('fake_08', 'Fake', 40.4, 0.985, 'Bilateral eye-region\nactivation'),
]


def load_sample_images(gradcam_dir, prefix):
    """Load original, heatmap, overlay from individual overlay PNG."""
    # The individual overlay files contain 3 panels side by side
    p = Path(gradcam_dir) / f"gradcam_overlay_{prefix}.png"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    img = np.array(Image.open(p))
    h, w = img.shape[:2]
    # 3 equal panels side by side
    pw = w // 3
    original = img[:, :pw, :]
    heatmap  = img[:, pw:2*pw, :]
    overlay  = img[:, 2*pw:, :]
    return original, heatmap, overlay


def generate(args):
    gd  = Path(args.gradcam_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Figure: 2 rows (real, fake) × 4 cols (orig_r0, ovl_r0, orig_r1, ovl_r1)
    # Plus a gap column between real and fake pairs
    fig = plt.figure(figsize=(7.16, 4.0))

    # Outer: 2 rows (real, fake), 2 cols (left sample, right sample)
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.55, wspace=0.10,
        left=0.05, right=0.98,
        top=0.88, bottom=0.12)

    for i, (prefix, label, ci, p_val, desc) in enumerate(SAMPLES):
        row = i // 2
        col = i % 2

        try:
            original, heatmap, overlay_img = load_sample_images(gd, prefix)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        col_border = COL_REAL if label == 'Real' else COL_FAKE

        # Each outer cell splits into 2 inner columns: original | overlay
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row, col],
            wspace=0.04)

        # Original frame
        ax_orig = fig.add_subplot(inner[0])
        ax_orig.imshow(original)
        ax_orig.axis('off')
        for sp in ax_orig.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(col_border)
            sp.set_linewidth(1.3)

        # Overlay
        ax_ovl = fig.add_subplot(inner[1])
        ax_ovl.imshow(overlay_img)
        ax_ovl.axis('off')
        for sp in ax_ovl.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(col_border)
            sp.set_linewidth(1.3)

        # Column headers (only top row, only left and right outer labels)
        if row == 0 and col == 0:
            ax_orig.set_title('Frame', fontsize=8, color='#37474F', pad=4)
            ax_ovl.set_title('GradCAM++ Overlay', fontsize=8, color='#37474F', pad=4)
        elif row == 0 and col == 1:
            ax_orig.set_title('Frame', fontsize=8, color='#37474F', pad=4)
            ax_ovl.set_title('GradCAM++ Overlay', fontsize=8, color='#37474F', pad=4)

        # Score + description below each pair
        score_txt = (f'$\\mathcal{{C}}_i$={ci:.1f}   '
                     f'$\\hat{{p}}$={p_val:.3f}   {desc.replace(chr(10), "  ")}')
        pos = outer[row, col].get_position(fig)
        fig.text(pos.x0 + pos.width/2,
                 pos.y0 - 0.030,
                 score_txt,
                 ha='center', va='top',
                 fontsize=7.2, color=col_border,
                 transform=fig.transFigure)

    # Row labels on the left
    fig.text(0.005, 0.72, 'Real',
             ha='left', va='center', rotation=90,
             fontsize=9, fontweight='bold', color=COL_REAL,
             transform=fig.transFigure)
    fig.text(0.005, 0.30, 'Fake',
             ha='left', va='center', rotation=90,
             fontsize=9, fontweight='bold', color=COL_FAKE,
             transform=fig.transFigure)

    # Dividing line between real and fake rows
    line_y = 0.525
    fig.add_artist(plt.Line2D([0.02, 0.98], [line_y, line_y],
                               transform=fig.transFigure,
                               color='#BDBDBD', lw=0.8, ls='--'))

    # Title
    fig.suptitle(
        'GradCAM++ Analysis — Visual Encoder Attention (CMPI)',
        fontsize=9.5, fontweight='bold', y=0.97, color='#1A1A1A')

    # Caption note
    fig.text(0.50, 0.005,
             'Real: compact face-region attention  |  '
             'Fake: diffuse/multi-focal activation '
             '(face-swap boundary artefacts)',
             ha='center', va='bottom', fontsize=7.2, color='#424242',
             transform=fig.transFigure)

    # Save
    fig.savefig(str(out), format='pdf', dpi=300)
    print(f"PDF: {out}")
    png = out.with_suffix('.png')
    fig.savefig(str(png), format='png', dpi=300)
    print(f"PNG: {png}")
    plt.close(fig)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gradcam_dir', default='figures/gradcam')
    p.add_argument('--output',      default='figures/fig6_gradcam.pdf')
    return p.parse_args()


if __name__ == '__main__':
    generate(parse_args())