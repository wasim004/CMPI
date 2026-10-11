"""
Extract sample frames and mel spectrogram for CMPI paper figures.
Saves:
  - figures/sample_real_frames.npy     [16, H, W, C] uint8
  - figures/sample_fake_frames.npy     [16, H, W, C] uint8
  - figures/sample_real_mel.npy        [128, 128] float32
  - figures/sample_fake_mel.npy        [128, 128] float32
  - figures/sample_real_frames_*.png   individual frame PNGs
  - figures/sample_fake_frames_*.png   individual frame PNGs
  - figures/real_mel.png               mel spectrogram image
  - figures/fake_mel.png               mel spectrogram image

Usage
-----
python extract_figure_samples.py \
    --metadata_csv     AMDD/data/metadata.csv \
    --preprocessed_dir AMDD/data/preprocessed \
    --output_dir       figures/raw_samples
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata_csv',
                   default='AMDD/data/metadata.csv')
    p.add_argument('--preprocessed_dir',
                   default='AMDD/data/preprocessed')
    p.add_argument('--output_dir', default='figures/raw_samples')
    p.add_argument('--n_frames_to_show', type=int, default=6,
                   help='How many frames to show in the film strip (4-8)')
    p.add_argument('--real_uid', default=None,
                   help='Specific UID for real sample (leave None for auto-select)')
    p.add_argument('--fake_uid', default=None,
                   help='Specific UID for fake sample (leave None for auto-select)')
    return p.parse_args()


def load_sample(uid, preprocessed_dir):
    uid_str     = str(uid).zfill(6)
    frames_path = Path(preprocessed_dir) / 'frames' / f"{uid_str}.npy"
    audio_path  = Path(preprocessed_dir) / 'audio'  / f"{uid_str}.npy"
    frames = np.load(str(frames_path))   # [16, H, W, C] uint8
    mel    = np.load(str(audio_path))    # [128, 128] float32
    return frames, mel


def save_individual_frames(frames, label, output_dir, n_show=6):
    """Save individual frame PNGs for use in figures."""
    total   = frames.shape[0]
    indices = np.linspace(0, total - 1, n_show, dtype=int)
    paths   = []
    for i, idx in enumerate(indices):
        frame = frames[idx]   # [H, W, C] uint8
        p     = output_dir / f"{label}_frame_{i:02d}.png"
        plt.imsave(str(p), frame)
        paths.append(p)
        print(f"  Saved: {p.name}")
    return paths, indices


def save_mel_image(mel, label, output_dir):
    """
    Save mel spectrogram as a clean image suitable for paper figures.
    Uses the standard librosa-style colormap (magma).
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    img = ax.imshow(mel, aspect='auto', origin='lower',
                    cmap='magma', interpolation='nearest')
    ax.set_xlabel('Time bins', fontsize=10)
    ax.set_ylabel('Mel bins', fontsize=10)
    ax.set_title(f'Mel Spectrogram ({label})', fontsize=11)
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = output_dir / f"{label}_mel.png"
    fig.savefig(str(p), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p.name}")
    return p


def save_filmstrip(frames, label, output_dir, n_show=6):
    """
    Save a horizontal film strip of n_show evenly spaced frames.
    This is the 'reel' style for the teaser figure.
    """
    total   = frames.shape[0]
    indices = np.linspace(0, total - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2.2))
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.imshow(frames[idx])
        ax.axis('off')
        # Subtle frame number label
        ax.set_title(f't={idx}', fontsize=7, color='gray', pad=2)

    # Add film-strip border effect
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')

    title_color = '#4CAF50' if label == 'real' else '#F44336'
    title_text  = 'Real (Authentic)' if label == 'real' else 'Fake (Manipulated)'
    fig.suptitle(title_text, fontsize=11, color=title_color,
                 fontweight='bold', y=1.02)

    fig.tight_layout(pad=0.3)
    p = output_dir / f"{label}_filmstrip.png"
    fig.savefig(str(p), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {p.name}")
    return p


def save_combined_preview(real_frames, fake_frames, real_mel, fake_mel,
                          output_dir, n_show=5):
    """
    Combined preview: [real filmstrip | real mel] / [fake filmstrip | fake mel]
    This is for quick inspection — not the final figure.
    """
    total_r  = real_frames.shape[0]
    total_f  = fake_frames.shape[0]
    r_idx    = np.linspace(0, total_r - 1, n_show, dtype=int)
    f_idx    = np.linspace(0, total_f - 1, n_show, dtype=int)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, n_show + 1,
                            width_ratios=[1]*n_show + [1.2],
                            hspace=0.4, wspace=0.1)

    # Real frames
    for i, idx in enumerate(r_idx):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(real_frames[idx])
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Real', fontsize=11, color='#4CAF50',
                          fontweight='bold', rotation=0, labelpad=40)

    # Real mel
    ax_rm = fig.add_subplot(gs[0, n_show])
    ax_rm.imshow(real_mel, aspect='auto', origin='lower',
                 cmap='magma', interpolation='nearest')
    ax_rm.set_title('Mel (real)', fontsize=9)
    ax_rm.axis('off')

    # Fake frames
    for i, idx in enumerate(f_idx):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(fake_frames[idx])
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Fake', fontsize=11, color='#F44336',
                          fontweight='bold', rotation=0, labelpad=40)

    # Fake mel
    ax_fm = fig.add_subplot(gs[1, n_show])
    ax_fm.imshow(fake_mel, aspect='auto', origin='lower',
                 cmap='magma', interpolation='nearest')
    ax_fm.set_title('Mel (fake)', fontsize=9)
    ax_fm.axis('off')

    fig.suptitle('Sample Inspection — Real vs Fake\n'
                 '(Select the clearest pair for paper figures)',
                 fontsize=12, y=1.01)

    p = output_dir / 'combined_preview.png'
    fig.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p.name}  ← inspect this first")
    return p


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)
    df = df[df['split'] == 'test'].reset_index(drop=True)
    df = df[df['success'] == True].reset_index(drop=True)

    prep_dir = Path(args.preprocessed_dir)

    # ── Select real sample ────────────────────────────────────────────────
    if args.real_uid:
        real_row = df[df['uid'] == int(args.real_uid)].iloc[0]
    else:
        # Pick a real sample from the middle of the dataset
        # (avoid first/last which may be edge cases)
        real_df  = df[df['label'] == 0].reset_index(drop=True)
        real_row = real_df.iloc[len(real_df) // 2]

    # ── Select fake sample ────────────────────────────────────────────────
    if args.fake_uid:
        fake_row = df[df['uid'] == int(args.fake_uid)].iloc[0]
    else:
        # Pick a fake sample — prefer Faceswap (generator_id=1) for clarity
        fake_df  = df[(df['label'] == 1) & (df['generator_id'] == 1)].reset_index(drop=True)
        if len(fake_df) == 0:
            fake_df = df[df['label'] == 1].reset_index(drop=True)
        fake_row = fake_df.iloc[len(fake_df) // 2]

    print(f"\n[Selected samples]")
    print(f"  Real UID: {real_row['uid']}  generator_id={real_row['generator_id']}")
    print(f"  Fake UID: {fake_row['uid']}  generator_id={fake_row['generator_id']}")

    # ── Load ──────────────────────────────────────────────────────────────
    real_frames, real_mel = load_sample(real_row['uid'], prep_dir)
    fake_frames, fake_mel = load_sample(fake_row['uid'], prep_dir)

    print(f"\n  real_frames shape: {real_frames.shape}")
    print(f"  fake_frames shape: {fake_frames.shape}")
    print(f"  real_mel shape:    {real_mel.shape}  range [{real_mel.min():.2f}, {real_mel.max():.2f}]")
    print(f"  fake_mel shape:    {fake_mel.shape}  range [{fake_mel.min():.2f}, {fake_mel.max():.2f}]")

    # ── Save raw arrays ───────────────────────────────────────────────────
    np.save(str(output_dir / 'real_frames.npy'), real_frames)
    np.save(str(output_dir / 'fake_frames.npy'), fake_frames)
    np.save(str(output_dir / 'real_mel.npy'),    real_mel)
    np.save(str(output_dir / 'fake_mel.npy'),    fake_mel)

    # Save selected UIDs for reference
    with open(str(output_dir / 'selected_uids.txt'), 'w') as f:
        f.write(f"real_uid={real_row['uid']}\n")
        f.write(f"fake_uid={fake_row['uid']}\n")

    # ── Generate outputs ──────────────────────────────────────────────────
    print(f"\n[Generating outputs]")

    # Individual frames
    save_individual_frames(real_frames, 'real', output_dir, args.n_frames_to_show)
    save_individual_frames(fake_frames, 'fake', output_dir, args.n_frames_to_show)

    # Mel spectrograms
    save_mel_image(real_mel, 'real', output_dir)
    save_mel_image(fake_mel, 'fake', output_dir)

    # Film strips
    save_filmstrip(real_frames, 'real', output_dir, args.n_frames_to_show)
    save_filmstrip(fake_frames, 'fake', output_dir, args.n_frames_to_show)

    # Combined preview for inspection
    save_combined_preview(real_frames, fake_frames, real_mel, fake_mel,
                          output_dir, n_show=args.n_frames_to_show)

    print(f"\n[Done] All outputs in: {output_dir}")
    print(f"\nNext step:")
    print(f"  1. Open combined_preview.png and inspect the real/fake pair")
    print(f"  2. If faces are clear and representative, confirm to proceed")
    print(f"  3. If not, re-run with --real_uid and --fake_uid to pick better samples")
    print(f"\n  List available test UIDs:")
    print(f"  Real UIDs: {list(df[df['label']==0]['uid'].values[:10])}")
    print(f"  Fake UIDs: {list(df[df['label']==1]['uid'].values[:10])}")


if __name__ == '__main__':
    main()