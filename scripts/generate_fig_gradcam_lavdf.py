"""
GradCAM++ figure on LAV-DF (cross-domain) — ISWA revision, Reviewer #2 pt.9
=============================================================================
Wasim Ahmad, BIT — September 2026

"Visualization results of GradCAM++ on the LAV-DF dataset should be
supplemented. Current visualizations are entirely sourced from FakeAVCeleb,
which prevents observation of shifts in encoder attention patterns under
cross-domain scenarios."

Produces a 2x2 (real top, fake bottom) frame|overlay figure in the same
style as the existing FakeAVCeleb GradCAM figure (generate_fig6_gradcam.py),
but computed directly here rather than reusing pre-baked overlay assets,
using the GradCAM++ implementation from eval_gradcam_mouth_stats.py.

Sample selection: representative, not hand-picked for effect — for each
class, scans LAV-DF test samples and picks the ones whose mouth-attention
mass is closest to that class's already-reported mean (real 0.125, fake
0.111, from results/gradcam_mouth/lavdf/gradcam_mouth_stats_lavdf.json),
so the figure illustrates the typical case, not an outlier.

Usage
-----
python generate_fig_gradcam_lavdf.py \\
    --checkpoint results/diam/best_model.pt \\
    --data_root data/crossdataset \\
    --output figures/fig8_gradcam_lavdf
"""
import sys, argparse, json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT / "AMDD" / "models"), str(ROOT / "AMDD" / "data"), str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_crossdataset_diam import CMPIDIAMModel
from eval_gradcam_mouth_stats import (
    gradcam_pp, get_mouth_mask, ManualForward, VAL_TRANSFORM,
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
})
COL_REAL = '#2E7D32'
COL_FAKE = '#C62828'


def compute_overlay(model, manual_fwd, act_holder, frames_np, mel_np, device, frame_idx=8):
    T_ = frames_np.shape[0]
    ft = torch.stack([VAL_TRANSFORM(frames_np[i]) for i in range(T_)], dim=0).unsqueeze(0).to(device)
    mel = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logits = manual_fwd(ft, mel)
    logits.sum().backward()
    act = act_holder['act']
    cams = gradcam_pp(act.detach(), act.grad.detach())
    cams = F.interpolate(cams.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    cam = cams[frame_idx].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    frame = frames_np[frame_idx]
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * frame.astype(np.float32) + 0.45 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
    with torch.no_grad():
        probs = torch.sigmoid(logits).item()
    return frame, overlay, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--output', default='figures/fig8_gradcam_lavdf')
    p.add_argument('--n_scan', type=int, default=40)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    manual_fwd = ManualForward(model)
    act_holder = {}
    def fwd_hook(module, inp, out):
        out.retain_grad()
        act_holder['act'] = out
    model.visual_encoder.backbone[7].register_forward_hook(fwd_hook)

    import face_alignment
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)

    meta = pd.read_csv(Path(args.data_root) / 'lavdf' / 'metadata.csv')
    meta = meta[meta['success'] == True]
    frames_dir = Path(args.data_root) / 'lavdf' / 'preprocessed' / 'frames'
    audio_dir = Path(args.data_root) / 'lavdf' / 'preprocessed' / 'audio'

    target_mass = {0: 0.1253, 1: 0.1107}  # from gradcam_mouth_stats_lavdf.json
    candidates = {0: [], 1: []}
    rng = np.random.RandomState(0)

    for label in (0, 1):
        sub = meta[meta.label == label].sample(n=min(args.n_scan, (meta.label==label).sum()), random_state=0)
        for _, row in sub.iterrows():
            uid = str(row['uid']).zfill(6)
            fpath, apath = frames_dir / f"{uid}.npy", audio_dir / f"{uid}.npy"
            if not fpath.exists() or not apath.exists():
                continue
            frames_np = np.load(str(fpath))
            mel = np.load(str(apath)).astype(np.float32)
            mn, mx = mel.min(), mel.max()
            if mx > mn:
                mel = 2.0 * (mel - mn) / (mx - mn) - 1.0

            frame_idx = 8
            frame, overlay, prob = compute_overlay(model, manual_fwd, act_holder, frames_np, mel, device, frame_idx)
            # mouth mass for this single frame, to pick representative examples
            ft = torch.stack([VAL_TRANSFORM(frames_np[i]) for i in range(frames_np.shape[0])], dim=0).unsqueeze(0).to(device)
            mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            logits = manual_fwd(ft, mel_t)
            logits.sum().backward()
            act = act_holder['act']
            cams = gradcam_pp(act.detach(), act.grad.detach())
            cams = F.interpolate(cams.unsqueeze(1), size=(224,224), mode='bilinear', align_corners=False).squeeze(1)
            cam_raw = cams[frame_idx].detach().cpu().numpy()
            s = cam_raw.sum()
            mass = float('nan')
            if s > 1e-8:
                cam_norm = cam_raw / s
                mask = get_mouth_mask(fa_model, frame)
                if mask is not None:
                    mass = float(cam_norm[mask].sum())
            if np.isnan(mass):
                continue
            candidates[label].append((abs(mass - target_mass[label]), uid, frame, overlay, prob, mass))

    samples = []
    for label in (0, 1):
        candidates[label].sort(key=lambda x: x[0])
        samples.extend([(label, c) for c in candidates[label][:2]])

    fig = plt.figure(figsize=(7.16, 4.3))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.10,
                               left=0.05, right=0.98, top=0.89, bottom=0.15)
    for i, (label, (_, uid, frame, overlay, prob, mass)) in enumerate(samples):
        row, col = i // 2, i % 2
        col_border = COL_REAL if label == 0 else COL_FAKE
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.04)
        ax_o = fig.add_subplot(inner[0]); ax_o.imshow(frame); ax_o.axis('off')
        ax_v = fig.add_subplot(inner[1]); ax_v.imshow(overlay); ax_v.axis('off')
        for ax in (ax_o, ax_v):
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(col_border); sp.set_linewidth(1.3)
        if row == 0:
            ax_o.set_title('Frame', fontsize=8, color='#37474F', pad=4)
            ax_v.set_title('GradCAM++ Overlay', fontsize=8, color='#37474F', pad=4)
        score_txt = f'$\\hat{{p}}$={prob:.3f}   mouth mass={mass:.3f}'
        pos = outer[row, col].get_position(fig)
        fig.text(pos.x0 + pos.width/2, pos.y0 - 0.030, score_txt, ha='center', va='top',
                  fontsize=7.2, color=col_border, transform=fig.transFigure)

    fig.text(0.005, 0.72, 'Real', ha='left', va='center', rotation=90, fontsize=9,
              fontweight='bold', color=COL_REAL, transform=fig.transFigure)
    fig.text(0.005, 0.30, 'Fake', ha='left', va='center', rotation=90, fontsize=9,
              fontweight='bold', color=COL_FAKE, transform=fig.transFigure)
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.525, 0.525], transform=fig.transFigure,
                               color='#BDBDBD', lw=0.8, ls='--'))
    fig.suptitle('GradCAM++ Analysis on LAV-DF (Cross-Domain) — CMPI Visual Encoder',
                  fontsize=9.5, fontweight='bold', y=0.97, color='#1A1A1A')
    fig.text(0.50, 0.005,
              'Representative examples (closest to each class mean mouth-attention mass) --\n'
              'the real/fake separation visible on FakeAVCeleb is largely absent here.',
              ha='center', va='bottom', fontsize=7.0, color='#424242', transform=fig.transFigure)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + '.pdf', format='pdf', dpi=300)
    fig.savefig(str(out) + '.png', format='png', dpi=300)
    print(f"Saved {out}.pdf / .png")
    print("Selected samples:", [(l, u, round(m,4)) for l,(_,u,_,_,_,m) in samples])


if __name__ == '__main__':
    main()
