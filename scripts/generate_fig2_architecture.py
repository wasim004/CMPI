"""
Figure 2 — CMPI+DIAM+TIL Architecture
CVIU-quality vector PDF output via matplotlib.

Usage:
  python generate_fig2_architecture.py \
      --samples_dir figures/raw_samples \
      --output      figures/fig2_architecture
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Embed fonts as TrueType in the PDF (no Type 3 / bitmap fonts) ──────────
matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
    'font.family':  'DejaVu Sans',
    'font.size':    11.0,
    'mathtext.fontset': 'dejavusans',
})

# ── Colour palette ──────────────────────────────────────────────────────────
#   Visual    → deep blue
#   Audio     → warm amber
#   Predictor → violet / indigo
#   DIAM      → crimson  (C_i, A)
#   TIL       → magenta / rose
#   Detection → dark teal/emerald
#   Output    → slate

VF, VE = '#DBEAFE', '#1D4ED8'   # visual  fill / edge
AF, AE = '#FEF3C7', '#B45309'   # audio   fill / edge
PF, PE = '#EDE9FE', '#5B21B6'   # predictor
CF, CE = '#FEE2E2', '#B91C1C'   # DIAM (C_i, A)
TF, TE = '#FCE7F3', '#9D174D'   # TIL
DF, DE = '#ECFDF5', '#065F46'   # detection head
OF, OE = '#F1F5F9', '#334155'   # output

ARROW_VIS  = VE
ARROW_AUD  = AE
ARROW_PRED = PE
ARROW_CI   = CE
ARROW_TIL  = TE
ARROW_DET  = DE
ARROW_SKIP = '#94A3B8'   # dashed skip arrows

LW_BOX   = 2.0    # outer box line width
LW_INNER = 1.0    # inner row line width
LW_ARR   = 1.5    # main arrow line width
LW_SKIP  = 1.05   # skip-connection arrow
FS       = 11.0   # base font size
FS_SM    = 9.8     # small label size
FS_TTL   = 12.5   # box title size
FS_MATH  = 13.0   # math symbol size

FIG_W = 17.6
FIG_H =  7.9


# ── Primitive helpers ───────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, fc, ec, lw=LW_BOX, rpad=0.035, zorder=3, alpha=1.0,
        dashed=False):
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={rpad}',
        linewidth=lw, edgecolor=ec, facecolor=fc,
        alpha=alpha, zorder=zorder,
        linestyle='--' if dashed else '-',
    )
    ax.add_patch(p)
    return p


def inner_box(ax, cx, cy, w, h, ec, zorder=4):
    """White inner row box."""
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.018',
        linewidth=LW_INNER, edgecolor=ec, facecolor='white',
        zorder=zorder
    )
    ax.add_patch(p)


def txt(ax, x, y, s, size=FS, color='#1E293B', ha='center', va='center',
        bold=False, italic=False, zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            zorder=zorder)


def arr(ax, x0, y0, x1, y1, col, lw=LW_ARR, dash=False,
        cs='arc3,rad=0.0', shrinkA=4, shrinkB=4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=col, lw=lw,
                    linestyle='--' if dash else '-',
                    connectionstyle=cs,
                    shrinkA=shrinkA, shrinkB=shrinkB
                ), zorder=6)


def hline(ax, x0, x1, y, col, lw=LW_ARR, dash=False):
    """Arrow-terminated horizontal line (no bend)."""
    ls = (0, (4, 3)) if dash else '-'
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(
                    arrowstyle='->', color=col, lw=lw,
                    linestyle=ls,
                    connectionstyle='arc3,rad=0.0',
                    shrinkA=4, shrinkB=4
                ), zorder=6)


# ── Compound components ─────────────────────────────────────────────────────

def encoder_block(ax, cx, cy, bw, bh, title, tc, rows, ec, fc):
    """
    Draw an encoder box containing stacked inner rows with down-arrows.

    rows: list of (label_string, inner_edge_color)
    """
    box(ax, cx, cy, bw, bh, fc, ec, lw=LW_BOX)
    txt(ax, cx, cy + bh/2 - 0.30, title, size=FS_TTL, color=tc, bold=True)

    n = len(rows)
    row_h = 0.46
    gap   = 0.12
    top_row_y = cy + bh/2 - 0.60 - row_h/2

    for i, (label, rc) in enumerate(rows):
        ry = top_row_y - i * (row_h + gap)
        inner_box(ax, cx, ry, bw - 0.28, row_h, rc)
        txt(ax, cx, ry, label, size=FS_SM, color='#374151')
        if i < n - 1:
            y_tail = ry - row_h / 2
            y_head = ry - row_h / 2 - gap + 0.01
            ax.annotate('', xy=(cx, y_head), xytext=(cx, y_tail),
                        arrowprops=dict(arrowstyle='->', color=rc, lw=0.85,
                                        shrinkA=0, shrinkB=0), zorder=5)


def film_strip(ax, cx, cy, w, h, ec, frames_rgb=None):
    """Film strip icon.  frames_rgb: list of 3 (H,W,3) uint8 arrays."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle='round,pad=0.02', lw=1.6,
                                edgecolor=ec, facecolor='#1F2937', zorder=3))
    n_holes = 6
    for side_y in [cy - h/2 + 0.10, cy + h/2 - 0.10]:
        for k in range(n_holes):
            hx = cx - w/2 + 0.10 + k * (w - 0.20) / (n_holes - 1)
            ax.add_patch(plt.Circle((hx, side_y), 0.055,
                                    color='#E5E7EB', zorder=5))
    slot_w = (w - 0.24) / 3 - 0.04
    slot_h = h - 0.38
    for k in range(3):
        sx = cx - w/2 + 0.10 + k * (slot_w + 0.04)
        sy = cy - slot_h / 2
        if frames_rgb is not None and k < len(frames_rgb):
            ax.imshow(frames_rgb[k],
                      extent=[sx, sx + slot_w, sy, sy + slot_h],
                      aspect='auto', zorder=6, origin='upper',
                      interpolation='bilinear')
        else:
            ax.add_patch(FancyBboxPatch((sx, sy), slot_w, slot_h,
                                        boxstyle='round,pad=0.01', lw=0.4,
                                        edgecolor='#9CA3AF', facecolor='#6B7280',
                                        zorder=4))


def mel_icon(ax, cx, cy, w, h, ec, mel_data=None):
    """Mel spectrogram icon.  mel_data: (128,128) float32 array."""
    if mel_data is not None:
        mn, mx = mel_data.min(), mel_data.max()
        mel_norm = (mel_data - mn) / (mx - mn + 1e-8)
        mel_rgb = (plt.cm.magma(mel_norm[::-1])[:, :, :3] * 255).astype(np.uint8)
        ax.imshow(mel_rgb,
                  extent=[cx - w/2, cx + w/2, cy - h/2, cy + h/2],
                  aspect='auto', zorder=4, origin='upper',
                  interpolation='bilinear')
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.02', lw=1.6,
                                    edgecolor=ec, facecolor='none', zorder=5))
    else:
        n_bands = 22
        for k in range(n_bands):
            col = plt.cm.magma(0.15 + 0.78 * k / n_bands)
            ax.add_patch(mpatches.Rectangle(
                (cx - w/2 + k * w / n_bands, cy - h/2),
                w / n_bands, h, facecolor=col, edgecolor='none', zorder=3))
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.02', lw=1.6,
                                    edgecolor=ec, facecolor='none', zorder=4))


# ── Main figure ─────────────────────────────────────────────────────────────

def generate(args):

    # ── Load real samples ────────────────────────────────────────────────
    sd = Path(args.samples_dir)
    frames_rgb = None
    mel_data   = None
    if sd.exists():
        frames_path = sd / 'real_frames.npy'
        mel_path    = sd / 'real_mel.npy'
        if frames_path.exists():
            all_frames = np.load(str(frames_path), allow_pickle=True)
            idxs = np.linspace(0, len(all_frames) - 1, 3, dtype=int)
            frames_rgb = [all_frames[i] for i in idxs]
            print(f"Loaded {len(all_frames)} frames -> using indices {list(idxs)}")
        if mel_path.exists():
            mel_data = np.load(str(mel_path), allow_pickle=True).astype(np.float32)
            print(f"Loaded mel spectrogram: shape={mel_data.shape}")
    else:
        print(f"Warning: samples_dir '{sd}' not found -- using schematic icons.")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Key y-coordinates ────────────────────────────────────────────────
    YV = 5.55    # visual pathway centre row
    YA = 2.05    # audio  pathway centre row
    YM = (YV + YA) / 2    # midline (DIAM / TIL / head / output)

    # ── Dimensions ───────────────────────────────────────────────────────
    EW, EH   = 3.15, 2.55   # encoder width, height
    ZW, ZH   = 0.98, 0.48   # z-pill width, height
    HW, HH   = 1.30, 0.46   # h_t per-frame tap pill
    PW, PH   = 1.85, 0.98   # predictor block

    AW, AH   = 2.05, 2.55   # DIAM box (C_i, A)
    TW, TH   = 2.05, 1.85   # TIL box

    DW, DH   = 2.55, 3.10   # detection head
    OW, OH   = 1.00, 0.92   # output node

    ICON_W, ICON_H_VID = 1.95, 1.33
    ICON_H_MEL = 1.13

    # ── Key x-coordinates  (left to right) ────────────────────────────────
    X_ICON =  1.10
    X_ENC  =  4.30                        # encoder centre
    X_Z    = X_ENC + EW/2 + 0.42 + ZW/2
    X_PRED = X_Z   + ZW/2 + 0.95 + PW/2
    X_DIAM = X_PRED + PW/2 + 0.62 + AW/2
    X_HD   = X_DIAM + AW/2 + 0.62 + DW/2
    X_OUT  = X_HD  + DW/2 + 0.62 + OW/2

    PY_AV = YM + 0.78    # T_{a->v} centre  (upper predictor)
    PY_VA = YM - 0.78    # T_{v->a} centre  (lower predictor)

    # DIAM / TIL stacked at the X_DIAM column
    AY = YM + 1.00        # DIAM box centre
    TY = YM - 1.62        # TIL  box centre

    # ══════════════════════════════════════════════════════════════════════
    # INPUT ICONS
    # ══════════════════════════════════════════════════════════════════════
    film_strip(ax, X_ICON, YV, ICON_W, ICON_H_VID, VE, frames_rgb=frames_rgb)
    txt(ax, X_ICON, YV + ICON_H_VID/2 + 0.26, 'Video Frames',
        size=FS_SM, color='#1E293B', bold=True)
    txt(ax, X_ICON, YV - ICON_H_VID/2 - 0.25, r'$(B,\,T,\,C,\,H,\,W)$',
        size=FS_SM - 0.8, color='#64748B', italic=True)

    mel_icon(ax, X_ICON, YA, ICON_W, ICON_H_MEL, AE, mel_data=mel_data)
    txt(ax, X_ICON, YA + ICON_H_MEL/2 + 0.26, 'Mel Spectrogram',
        size=FS_SM, color='#1E293B', bold=True)
    txt(ax, X_ICON, YA - ICON_H_MEL/2 - 0.25, r'$(B,\,128,\,128)$',
        size=FS_SM - 0.8, color='#64748B', italic=True)

    # ══════════════════════════════════════════════════════════════════════
    # ENCODER BLOCKS
    # ══════════════════════════════════════════════════════════════════════
    encoder_block(ax, X_ENC, YV, EW, EH,
                  'Visual Encoder', VE,
                  [('ResNet50 (ImageNet pretrained)', VE),
                   (r'Linear proj. $\rightarrow$ LayerNorm (512-d)', '#3B82F6'),
                   ('Temporal self-attention (8 heads)', '#60A5FA')],
                  VE, VF)

    encoder_block(ax, X_ENC, YA, EW, EH,
                  'Audio Encoder', AE,
                  [('ResNet18 (1-ch, ImageNet init)', AE),
                   (r'Linear proj. $\rightarrow$ LayerNorm (512-d)', '#F97316'),
                   (r'Audio embed. $z_a \!\in\! \mathbb{R}^{512}$', '#FB923C')],
                  AE, AF)

    hline(ax, X_ICON + ICON_W/2, X_ENC - EW/2, YV, ARROW_VIS)
    hline(ax, X_ICON + ICON_W/2, X_ENC - EW/2, YA, ARROW_AUD)

    # ══════════════════════════════════════════════════════════════════════
    # EMBEDDING PILLS  z_v, z_a  (+ per-frame tap h_t)
    # ══════════════════════════════════════════════════════════════════════
    box(ax, X_Z, YV, ZW, ZH, VF, VE, lw=1.6, rpad=0.02)
    txt(ax, X_Z, YV, r'$z_v$', size=FS_MATH, color=VE, bold=True)

    box(ax, X_Z, YA, ZW, ZH, AF, AE, lw=1.6, rpad=0.02)
    txt(ax, X_Z, YA, r'$z_a$', size=FS_MATH, color=AE, bold=True)

    hline(ax, X_ENC + EW/2, X_Z - ZW/2, YV, ARROW_VIS)
    hline(ax, X_ENC + EW/2, X_Z - ZW/2, YA, ARROW_AUD)

    # h_t: per-frame visual features, tapped alongside z_v, feeds TIL
    HY = YV - 0.95
    box(ax, X_Z, HY, HW, HH, 'white', VE, lw=1.3, rpad=0.02, dashed=True)
    txt(ax, X_Z, HY, r'$h_t \in \mathbb{R}^{T \times 512}$',
        size=FS_SM - 0.8, color=VE, italic=True)
    arr(ax, X_ENC + EW/2 - 0.35, YV - EH/2 + 0.05, X_Z, HY + HH/2 + 0.02,
        VE, lw=LW_SKIP, dash=True, cs='arc3,rad=-0.15', shrinkA=2, shrinkB=3)

    # ══════════════════════════════════════════════════════════════════════
    # CROSS-MODAL PREDICTOR BLOCKS
    # ══════════════════════════════════════════════════════════════════════
    box(ax, X_PRED, PY_AV, PW, PH, PF, PE, lw=LW_BOX)
    txt(ax, X_PRED, PY_AV + 0.20, r'$T_{a \to v}$',
        size=FS_MATH, color=PE, bold=True)
    txt(ax, X_PRED, PY_AV - 0.20,
        r'MLP: $512 \!\to\! 1024 \!\to\! 512$',
        size=FS_SM, color='#374151')

    box(ax, X_PRED, PY_VA, PW, PH, PF, PE, lw=LW_BOX)
    txt(ax, X_PRED, PY_VA + 0.20, r'$T_{v \to a}$',
        size=FS_MATH, color=PE, bold=True)
    txt(ax, X_PRED, PY_VA - 0.20,
        r'MLP: $512 \!\to\! 1024 \!\to\! 512$',
        size=FS_SM, color='#374151')

    # z_a -> T_{a->v}
    arr(ax, X_Z + ZW/2, YA, X_PRED - PW/2, PY_AV,
        ARROW_AUD, cs='angle,angleA=90,angleB=180,rad=5')
    # z_v -> T_{v->a}
    arr(ax, X_Z + ZW/2, YV, X_PRED - PW/2, PY_VA,
        ARROW_VIS, cs='angle,angleA=-90,angleB=180,rad=5')

    # ══════════════════════════════════════════════════════════════════════
    # DIAM BOX — C_i (magnitude) and A (signed asymmetry)
    # ══════════════════════════════════════════════════════════════════════
    box(ax, X_DIAM, AY, AW, AH, CF, CE, lw=LW_BOX + 0.3)
    txt(ax, X_DIAM, AY + AH/2 - 0.28, 'DIAM', size=FS_TTL, color=CE, bold=True)

    diam_rows = [
        r'$\varepsilon_f = \|\hat{z}_v - z_v\|$',
        r'$\varepsilon_b = \|\hat{z}_a - z_a\|$',
        r'$\mathcal{C}_i = \varepsilon_f + \varepsilon_b$',
        r'$\mathcal{A} = \varepsilon_f - \varepsilon_b$',
    ]
    n_ar = len(diam_rows)
    ar_h, ar_g = 0.42, 0.10
    ar_top = AY + AH/2 - 0.62 - ar_h/2
    for i, lab in enumerate(diam_rows):
        ry = ar_top - i * (ar_h + ar_g)
        ec = CE if i >= 2 else '#F87171'
        inner_box(ax, X_DIAM, ry, AW - 0.24, ar_h, ec)
        txt(ax, X_DIAM, ry, lab, size=FS_SM, color='#7F1D1D')

    arr(ax, X_PRED + PW/2, PY_AV, X_DIAM - AW/2, AY + 0.30, ARROW_PRED)
    arr(ax, X_PRED + PW/2, PY_VA, X_DIAM - AW/2, AY - 0.55, ARROW_PRED)

    # ══════════════════════════════════════════════════════════════════════
    # TIL BOX — Temporal Inconsistency Localization
    # ══════════════════════════════════════════════════════════════════════
    box(ax, X_DIAM, TY, TW, TH, TF, TE, lw=LW_BOX + 0.3)
    txt(ax, X_DIAM, TY + TH/2 - 0.28, 'TIL', size=FS_TTL, color=TE, bold=True)

    til_rows = [
        r'$e_t = \|\hat{z}_v - h_t\|$',
        r'$\alpha_t = \mathrm{softmax}(\mathbf{w}^\top\!\tanh(\mathbf{W}e_t))$',
        r'$\mathrm{TIL} = \sum_t \alpha_t\, e_t$',
    ]
    n_tr = len(til_rows)
    tr_h, tr_g = 0.42, 0.10
    tr_top = TY + TH/2 - 0.62 - tr_h/2
    til_font_sizes = [FS_SM - 0.6, FS_SM - 2.0, FS_SM - 0.6]
    for i, (lab, fsz) in enumerate(zip(til_rows, til_font_sizes)):
        ry = tr_top - i * (tr_h + tr_g)
        inner_box(ax, X_DIAM, ry, TW - 0.24, tr_h, TE)
        txt(ax, X_DIAM, ry, lab, size=fsz, color='#831843')

    # h_t -> TIL  (dashed, per-frame tap)
    arr(ax, X_Z, HY - HH/2, X_DIAM - AW/2, TY + 0.35,
        VE, lw=LW_SKIP, dash=True, cs='arc3,rad=-0.12', shrinkA=3, shrinkB=4)
    # T_{a->v} -> TIL  (shares the forward predictor with DIAM)
    arr(ax, X_PRED, PY_AV - PH/2, X_DIAM - AW/2, TY - 0.30,
        ARROW_PRED, lw=LW_SKIP, dash=True, cs='arc3,rad=0.18', shrinkA=3, shrinkB=4)

    # ══════════════════════════════════════════════════════════════════════
    # DETECTION HEAD
    # ══════════════════════════════════════════════════════════════════════
    box(ax, X_HD, YM, DW, DH, DF, DE, lw=LW_BOX + 0.3)
    txt(ax, X_HD, YM + DH/2 - 0.28, 'Detection Head',
        size=FS_TTL, color=DE, bold=True)

    det_rows = [
        (r'$[z_v;z_a;\tilde{\mathcal{C}}_i;\tilde{\mathcal{A}};\widetilde{\mathrm{TIL}}]\!\in\!\mathbb{R}^{1027}$', '#34D399'),
        (r'FC $1027 \!\to\! 256$ + ReLU', '#10B981'),
        (r'FC $256 \!\to\! 64$ + ReLU',   '#059669'),
        (r'FC $64 \!\to\! 1 \!\to\! \sigma$', '#047857'),
    ]
    n_dr  = len(det_rows)
    dr_h, dr_g = 0.44, 0.11
    dr_top_y = YM + DH/2 - 0.60 - dr_h/2

    for i, (lab, ec) in enumerate(det_rows):
        ry = dr_top_y - i * (dr_h + dr_g)
        inner_box(ax, X_HD, ry, DW - 0.30, dr_h, ec)
        txt(ax, X_HD, ry, lab, size=FS_SM - 0.5, color='#1F2937')
        if i < n_dr - 1:
            y_tail = ry - dr_h / 2
            y_head = ry - dr_h / 2 - dr_g + 0.01
            ax.annotate('', xy=(X_HD, y_head), xytext=(X_HD, y_tail),
                        arrowprops=dict(arrowstyle='->', color=ec, lw=0.85,
                                        shrinkA=0, shrinkB=0), zorder=5)

    # DIAM -> Detection Head
    arr(ax, X_DIAM + AW/2, AY, X_HD - DW/2, YM + DH/2 - 0.60,
        ARROW_CI, lw=1.7, cs='arc3,rad=0.0')
    # TIL -> Detection Head
    arr(ax, X_DIAM + TW/2, TY, X_HD - DW/2, YM - DH/2 + 0.65,
        ARROW_TIL, lw=1.7, cs='arc3,rad=0.0')

    # Dashed skip connections: z_v and z_a feed directly into Detection Head
    arr(ax, X_Z + ZW/2, YV,
        X_HD - DW/2, YM + 0.28,
        ARROW_SKIP, lw=LW_SKIP, dash=True,
        cs='arc3,rad=0.0', shrinkA=5, shrinkB=5)

    arr(ax, X_Z + ZW/2, YA,
        X_HD - DW/2, YM - 0.28,
        ARROW_SKIP, lw=LW_SKIP, dash=True,
        cs='arc3,rad=0.0', shrinkA=5, shrinkB=5)

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT NODE
    # ══════════════════════════════════════════════════════════════════════
    box(ax, X_OUT, YM, OW, OH, OF, OE, lw=1.6, rpad=0.03)
    txt(ax, X_OUT, YM + 0.19, r'$\hat{p}$', size=17, color=OE, bold=True)
    txt(ax, X_OUT, YM - 0.20, r'$P(\mathrm{fake})$',
        size=FS_SM, color='#475569')

    hline(ax, X_HD + DW/2, X_OUT - OW/2, YM, ARROW_DET, lw=1.7)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION LABELS  (stage annotations near top)
    # ══════════════════════════════════════════════════════════════════════
    stage_labels = [
        (X_ENC,  'Feature\nExtraction'),
        (X_Z,    'Embedding'),
        (X_PRED, 'Cross-Modal\nPrediction'),
        (X_DIAM, 'DIAM + TIL'),
        (X_HD,   'Classification'),
    ]
    for sx, slabel in stage_labels:
        ax.text(sx, FIG_H - 0.15, slabel,
                ha='center', va='top', fontsize=FS_SM - 0.3,
                color='#94A3B8', style='italic', zorder=5)

    # Faint vertical stage dividers (in the gap between columns)
    gap_xs = [
        (X_ENC + EW/2 + 0.10,  X_Z    - ZW/2 - 0.10),
        (X_Z   + ZW/2 + 0.10,  X_PRED - PW/2 - 0.10),
        (X_PRED + PW/2 + 0.10, X_DIAM - AW/2 - 0.10),
        (X_DIAM + AW/2 + 0.10, X_HD   - DW/2 - 0.10),
        (X_HD  + DW/2 + 0.10,  X_OUT  - OW/2 - 0.10),
    ]
    for x0, x1 in gap_xs:
        ax.axvline(x=(x0 + x1) / 2, ymin=0.06, ymax=0.93,
                   color='#E2E8F0', lw=0.8, zorder=2)

    # ══════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════
    legend_items = [
        (VF, VE, 'Visual pathway',            2.35),
        (AF, AE, 'Audio pathway',             2.35),
        (PF, PE, 'Cross-modal predictor',     2.95),
        (CF, CE, r'DIAM ($\mathcal{C}_i,\mathcal{A}$)', 2.35),
        (TF, TE, 'TIL (temporal attention)',  3.00),
        (DF, DE, 'Detection head',            2.35),
    ]
    lx = 0.55; ly = 0.20; bs = 0.20
    for fc, ec, lab, step in legend_items:
        ax.add_patch(FancyBboxPatch((lx, ly - bs/2), bs, bs,
                                    boxstyle='round,pad=0.01', lw=1.0,
                                    edgecolor=ec, facecolor=fc, zorder=6))
        txt(ax, lx + bs + 0.10, ly, lab,
            size=FS_SM, color='#374151', ha='left')
        lx += step

    ax.plot([lx, lx + 0.40], [ly, ly], '--',
            color=ARROW_SKIP, lw=1.1, zorder=6)
    ax.annotate('', xy=(lx + 0.40, ly), xytext=(lx + 0.34, ly),
                arrowprops=dict(arrowstyle='->', color=ARROW_SKIP, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=6)
    txt(ax, lx + 0.58, ly, r'Direct skip / per-frame tap ($z_v,z_a,h_t$)',
        size=FS_SM, color=ARROW_SKIP, ha='left')

    # ══════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════
    out = Path(args.output)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)

    out.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out.with_suffix('.pdf')
    fig.savefig(str(pdf_path), format='pdf', bbox_inches='tight',
                facecolor='white', dpi=300)
    print(f"PDF  -> {pdf_path}")

    png_path = out.with_suffix('.png')
    fig.savefig(str(png_path), format='png', bbox_inches='tight',
                facecolor='white', dpi=300)
    print(f"PNG  -> {png_path}")

    plt.close(fig)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir', default='figures/raw_samples')
    p.add_argument('--output',      default='figures/fig2_architecture')
    return p.parse_args()


if __name__ == '__main__':
    generate(parse_args())
