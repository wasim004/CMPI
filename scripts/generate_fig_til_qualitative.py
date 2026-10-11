"""
TIL qualitative case visualization on partially-tampered LAV-DF clips
=========================================================================
Wasim Ahmad, BIT — September 2026

ISWA revision (ISWA-D-26-01461), Reviewer #2 pt.2: "add more qualitative
case visualizations for partially tampered samples."

For a few example LAV-DF fakes spanning different tamper fractions, plots
the TIL module's per-frame attention weight alpha_t (from the trained
TemporalAttentionScorer) against time, with the ground-truth manipulated
interval (reconstructed from LAV-DF's original metadata.json fake_periods,
same method as eval_til_partial_tamper.py) shaded, so the reader can see
directly whether TIL's attention concentrates on the actually-manipulated
frames or not.

Usage
-----
python generate_fig_til_qualitative.py \\
    --checkpoint results/diam/best_model.pt \\
    --data_root data/crossdataset \\
    --output figures/fig9_til_qualitative
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT / "AMDD" / "models"), str(ROOT / "AMDD" / "data"), str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_crossdataset_diam import CMPIDIAMModel, VAL_TRANSFORM

LAVDF_ROOT = Path("datasets/LAV-DF/LAV-DF")


def reconstruct_fake_entries():
    """uid(str) -> (duration, fake_periods, n_fakes) for the 4000-row LAV-DF
    cross-dataset test set, same reconstruction as eval_til_partial_tamper.py."""
    with open(str(LAVDF_ROOT / 'metadata.json')) as f:
        meta = json.load(f)
    entries = [e for e in meta if e['split'] == 'test']
    real_entries = [e for e in entries if e['n_fakes'] == 0][:2000]
    fake_entries = [e for e in entries if e['n_fakes'] > 0][:2000]

    out = {}
    uid_counter = 0
    for e in real_entries:
        out[str(uid_counter).zfill(6)] = (e['duration'], [], 0)
        uid_counter += 1
    for e in fake_entries:
        out[str(uid_counter).zfill(6)] = (e['duration'], e['fake_periods'], e['n_fakes'])
        uid_counter += 1
    return out


@torch.no_grad()
def get_til_alpha(model, frames_np, mel_np, device):
    ft = torch.stack([VAL_TRANSFORM(frames_np[i]) for i in range(frames_np.shape[0])], dim=0).unsqueeze(0).to(device)
    mel = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0).to(device)
    out = model(ft, mel)
    return out['til_alpha'].squeeze(0).cpu().numpy(), out['det_probs'].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--output', default='figures/fig9_til_qualitative')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    entries = reconstruct_fake_entries()
    meta = pd.read_csv(Path(args.data_root) / 'lavdf' / 'metadata.csv')
    meta = meta[(meta['success'] == True) & (meta['label'] == 1)]
    frames_dir = Path(args.data_root) / 'lavdf' / 'preprocessed' / 'frames'
    audio_dir = Path(args.data_root) / 'lavdf' / 'preprocessed' / 'audio'
    NUM_FRAMES = 16

    # Pick 3 examples spanning low / mid / high tamper fraction, each with a
    # single contiguous fake_period (cleanest to visualize), fake segment not
    # touching the very start/end of the clip (so shading is visible).
    picks = []
    for _, row in meta.sample(frac=1.0, random_state=0).iterrows():
        uid = str(row['uid']).zfill(6)
        if uid not in entries:
            continue
        dur, periods, n_fakes = entries[uid]
        if n_fakes != 1 or dur <= 0:
            continue
        start, end = periods[0]
        frac = (end - start) / dur
        if start < 0.3 or end > dur - 0.3:
            continue
        fpath, apath = frames_dir / f"{uid}.npy", audio_dir / f"{uid}.npy"
        if not fpath.exists() or not apath.exists():
            continue
        picks.append((frac, uid, dur, start, end))

    picks.sort(key=lambda x: x[0])
    n = len(picks)
    chosen = [picks[int(n * 0.15)], picks[int(n * 0.5)], picks[int(n * 0.85)]] if n >= 3 else picks[:3]

    fig, axes = plt.subplots(1, len(chosen), figsize=(9.0, 2.6), sharey=True)
    if len(chosen) == 1:
        axes = [axes]

    for ax, (frac, uid, dur, start, end) in zip(axes, chosen):
        frames_np = np.load(str(frames_dir / f"{uid}.npy"))
        mel = np.load(str(audio_dir / f"{uid}.npy")).astype(np.float32)
        mn, mx = mel.min(), mel.max()
        if mx > mn:
            mel = 2.0 * (mel - mn) / (mx - mn) - 1.0
        alpha, prob = get_til_alpha(model, frames_np, mel, device)

        frame_times = np.linspace(0, dur, NUM_FRAMES)
        ax.axvspan(start, end, color='#C62828', alpha=0.15, label='True fake segment' if ax is axes[0] else None)
        ax.bar(frame_times, alpha, width=dur / NUM_FRAMES * 0.8, color='#37474F')
        ax.set_title(f'tamper frac={frac:.2f}\n$\\hat{{p}}$={prob:.3f}', fontsize=8.5)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel(r'TIL attention $\alpha_t$', fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='upper right', fontsize=7, frameon=False)
    fig.suptitle('TIL Per-Frame Attention on Partially-Tampered LAV-DF Clips',
                 fontsize=9.5, fontweight='bold', y=1.03)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(str(out) + '.png', format='png', dpi=300, bbox_inches='tight')
    print(f"Saved {out}.pdf / .png")
    print("Chosen:", [(u, round(f,3), round(s,2), round(e,2)) for f,u,d,s,e in chosen])


if __name__ == '__main__':
    main()
