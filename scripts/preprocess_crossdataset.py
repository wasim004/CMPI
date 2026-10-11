"""
Cross-Dataset Preprocessing for CMPI Evaluation
==================================================
Wasim Ahmad, BIT

Converts raw videos from DF-TIMIT, LAV-DF, and DFDM into the same
.npy format used by AMDD/CMPI:
  - frames/XXXXXX.npy  : [T, H, W, C] uint8, T=16 face crops at 224x224
  - audio/XXXXXX.npy   : [128, 128] float32 mel spectrogram

Uses the same face_alignment pipeline as AMDD preprocess_videos.py
to ensure visual embeddings are in the same distribution.

Usage
-----
python preprocess_crossdataset.py --dataset all \
    --output_root data/crossdataset

# Quick test:
python preprocess_crossdataset.py --dataset all \
    --output_root data/crossdataset \
    --max_samples 50

# Single dataset:
python preprocess_crossdataset.py --dataset dftimit \
    --output_root data/crossdataset
"""

import os
import cv2
import json
import argparse
import subprocess
import tempfile
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Dataset root paths ────────────────────────────────────────────────────
DFTIMIT_ROOT = Path("datasets/DeepfakeTIMIT/DF-TIMIT")
LAVDF_ROOT   = Path("datasets/LAV-DF/LAV-DF")
DFDM_ROOT    = Path("datasets/DFDM/Full Dataset")
CELEBDF_ROOT = Path("datasets/CelebDF2")

# ── Preprocessing config — MUST match AMDD exactly ───────────────────────
NUM_FRAMES     = 16
FRAME_SIZE     = 224
NUM_MEL_BINS   = 128
MEL_TIME_BINS  = 128
SAMPLE_RATE    = 16000
AUDIO_DURATION = 4.0
HOP_LENGTH     = 512
N_FFT          = 1024

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=NUM_MEL_BINS,
    f_min=0,
    f_max=8000,
)

# ── Face detector — same as AMDD preprocess_videos.py ────────────────────
print("Loading face detector...")
try:
    import face_alignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cpu',
        flip_input=False
    )
    USE_FACE_ALIGNMENT = True
    print("face_alignment loaded — landmark-based face crop active")
except ImportError:
    fa = None
    USE_FACE_ALIGNMENT = False
    print("WARNING: face_alignment not available — using centre crop fallback")
    print("Install: pip install face-alignment --break-system-packages")


# ══════════════════════════════════════════════════════════════════════════
# Core utilities
# ══════════════════════════════════════════════════════════════════════════

def crop_face_from_frame(frame_rgb, frame_size=FRAME_SIZE):
    h, w = frame_rgb.shape[:2]
    if USE_FACE_ALIGNMENT:
        try:
            preds = fa.get_landmarks(frame_rgb)
            if preds is not None and len(preds) > 0:
                lm    = preds[0]
                x_min = int(max(0, lm[:, 0].min() - 20))
                x_max = int(min(w, lm[:, 0].max() + 20))
                y_min = int(max(0, lm[:, 1].min() - 30))
                y_max = int(min(h, lm[:, 1].max() + 20))
                face  = frame_rgb[y_min:y_max, x_min:x_max]
                if face.size > 0:
                    return cv2.resize(face, (frame_size, frame_size),
                                      interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
    # Centre crop fallback
    crop = min(h, w)
    y0   = (h - crop) // 2
    x0   = (w - crop) // 2
    return cv2.resize(frame_rgb[y0:y0+crop, x0:x0+crop],
                      (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)


def extract_face_frames(video_path, n_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frames.append(frames[-1].copy() if frames
                          else np.zeros((frame_size, frame_size, 3), np.uint8))
            continue
        frames.append(crop_face_from_frame(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_size))
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1].copy() if frames
                      else np.zeros((frame_size, frame_size, 3), np.uint8))
    return np.stack(frames[:n_frames], axis=0).astype(np.uint8)


def extract_audio_from_video(video_path, target_sr=SAMPLE_RATE):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path),
             '-ar', str(target_sr), '-ac', '1', '-loglevel', 'error', tmp_path],
            capture_output=True, timeout=30)
        if r.returncode != 0:
            return None
        waveform, sr = torchaudio.load(tmp_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def load_audio_wav(wav_path, target_sr=SAMPLE_RATE):
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform
    except Exception:
        return None


def audio_to_mel(waveform):
    target_samples = int(AUDIO_DURATION * SAMPLE_RATE)
    if waveform.shape[1] >= target_samples:
        waveform = waveform[:, :target_samples]
    else:
        waveform = torch.nn.functional.pad(
            waveform, (0, target_samples - waveform.shape[1]))
    mel = mel_transform(waveform).squeeze(0)
    if mel.shape[1] != MEL_TIME_BINS:
        mel = torch.tensor(cv2.resize(
            mel.numpy(), (MEL_TIME_BINS, NUM_MEL_BINS),
            interpolation=cv2.INTER_LINEAR))
    return torch.log(mel + 1e-9).numpy().astype(np.float32)


def save_sample(uid, frames, mel, frames_dir, audio_dir):
    np.save(str(frames_dir / f"{uid}.npy"), frames)
    np.save(str(audio_dir  / f"{uid}.npy"), mel)


# ══════════════════════════════════════════════════════════════════════════
# Dataset builders
# ══════════════════════════════════════════════════════════════════════════

def build_dftimit(output_dir, max_samples=None):
    """
    DF-TIMIT:
      Fake/higher_quality/<subject>/<n>-video-fram1.avi + <n>.wav
      Original/<subject>/audio/<n>.wav + video/<n>.avi

    Visual-only deepfake — audio is original speaker's voice.
    Expected: T_va (visual->audio) error high for fakes, T_av lower.
    """
    frames_dir = output_dir / 'preprocessed' / 'frames'
    audio_dir  = output_dir / 'preprocessed' / 'audio'
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    records     = []
    uid_counter = 0
    fake_root   = DFTIMIT_ROOT / 'Fake' / 'higher_quality'
    real_root   = DFTIMIT_ROOT / 'Original'
    n_fake_cap  = (max_samples // 2) if max_samples else 9999

    # Fake
    print("\n[DF-TIMIT] Fake samples...")
    for subject in tqdm(sorted(d for d in fake_root.iterdir() if d.is_dir()),
                        desc="  subjects"):
        for avi in sorted(subject.glob('*-video-fram1.avi')):
            stem = avi.name.replace('-video-fram1.avi', '')
            wav  = subject / f"{stem}.wav"
            if not wav.exists():
                continue
            frames = extract_face_frames(avi)
            if frames is None:
                continue
            waveform = load_audio_wav(wav)
            if waveform is None:
                continue
            uid = str(uid_counter).zfill(6)
            save_sample(uid, frames, audio_to_mel(waveform), frames_dir, audio_dir)
            records.append({'uid': uid, 'label': 1, 'generator_id': 1,
                            'split': 'test', 'success': True,
                            'source': 'dftimit_fake',
                            'modify_video': True, 'modify_audio': False})
            uid_counter += 1
            if uid_counter >= n_fake_cap:
                break
        if uid_counter >= n_fake_cap:
            break

    n_fake = uid_counter
    print(f"  Extracted {n_fake} fake samples")

    # Real — Original/<subject>/video/<clip_name>/<numbered_jpegs> + audio/<clip>.wav
    # The video folder contains clip subdirectories (sa1, sa2, si649, ...).
    # Each clip dir contains numbered extensionless JPEG files (001, 002, ...).
    # Audio lives in audio/<clip_stem>.wav alongside the video/<clip_stem>/ dir.
    print("[DF-TIMIT] Real samples (JPEG frame sequences)...")
    n_real = 0
    for subject in tqdm(sorted(d for d in real_root.iterdir() if d.is_dir()),
                        desc="  subjects"):
        vdir = subject / 'video'
        adir = subject / 'audio'
        if not vdir.exists() or not adir.exists():
            continue
        audio_stems = {f.stem for f in adir.glob('*.wav')}

        # Each entry in vdir is a clip subdirectory (not a video file)
        for clip_dir in sorted(d for d in vdir.iterdir() if d.is_dir()):
            clip_stem = clip_dir.name          # e.g. 'sa1', 'si649'
            if clip_stem not in audio_stems:
                continue
            wav = adir / f"{clip_stem}.wav"
            if not wav.exists():
                continue

            # Read numbered JPEG frames (extensionless files, sorted numerically)
            frame_files = sorted(
                [f for f in clip_dir.iterdir() if f.is_file()],
                key=lambda f: int(f.name) if f.name.isdigit() else 0
            )
            if len(frame_files) < 2:
                continue

            # Sample NUM_FRAMES evenly from the clip
            indices    = np.linspace(0, len(frame_files) - 1, NUM_FRAMES, dtype=int)
            raw_frames = []
            for idx in indices:
                img = cv2.imread(str(frame_files[idx]))
                if img is None:
                    # Try reading as JPEG explicitly
                    with open(str(frame_files[idx]), 'rb') as fh:
                        buf = np.frombuffer(fh.read(), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raw_frames.append(raw_frames[-1].copy() if raw_frames
                                      else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8))
                    continue
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                raw_frames.append(crop_face_from_frame(frame_rgb, FRAME_SIZE))

            while len(raw_frames) < NUM_FRAMES:
                raw_frames.append(raw_frames[-1].copy() if raw_frames
                                  else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8))
            frames = np.stack(raw_frames[:NUM_FRAMES], axis=0).astype(np.uint8)

            waveform = load_audio_wav(wav)
            if waveform is None:
                continue

            uid = str(uid_counter).zfill(6)
            save_sample(uid, frames, audio_to_mel(waveform), frames_dir, audio_dir)
            records.append({'uid': uid, 'label': 0, 'generator_id': 0,
                            'split': 'test', 'success': True,
                            'source': 'dftimit_real',
                            'modify_video': False, 'modify_audio': False})
            uid_counter += 1
            n_real      += 1
            if n_real >= n_fake:
                break
        if n_real >= n_fake:
            break

    df = pd.DataFrame(records)
    df.to_csv(str(output_dir / 'metadata.csv'), index=False)
    print(f"  Final — Real: {(df.label==0).sum()}  Fake: {(df.label==1).sum()}")
    return df


def build_lavdf(output_dir, max_samples=None, split='test'):
    frames_dir = output_dir / 'preprocessed' / 'frames'
    audio_dir  = output_dir / 'preprocessed' / 'audio'
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    with open(str(LAVDF_ROOT / 'metadata.json')) as f:
        meta = json.load(f)

    entries      = [e for e in meta if e['split'] == split]
    real_entries = [e for e in entries if e['n_fakes'] == 0]
    fake_entries = [e for e in entries if e['n_fakes'] >  0]
    cap          = 2000 if max_samples is None else max_samples // 2
    real_entries = real_entries[:cap]
    fake_entries = fake_entries[:cap]

    print(f"\n[LAV-DF] {len(real_entries)} real + {len(fake_entries)} fake...")
    records = []
    uid_counter = 0

    for label, entry_list in [(0, real_entries), (1, fake_entries)]:
        for entry in tqdm(entry_list, desc=f"  label={label}"):
            video_path = LAVDF_ROOT / entry['file']
            if not video_path.exists():
                continue
            frames = extract_face_frames(video_path)
            if frames is None:
                continue
            waveform = extract_audio_from_video(video_path)
            if waveform is None:
                continue
            uid = str(uid_counter).zfill(6)
            save_sample(uid, frames, audio_to_mel(waveform), frames_dir, audio_dir)
            records.append({'uid': uid, 'label': label, 'generator_id': label,
                            'split': 'test', 'success': True, 'source': 'lavdf',
                            'modify_video': entry.get('modify_video', False),
                            'modify_audio': entry.get('modify_audio', False)})
            uid_counter += 1

    df = pd.DataFrame(records)
    df.to_csv(str(output_dir / 'metadata.csv'), index=False)
    print(f"  Final — Real: {(df.label==0).sum()}  Fake: {(df.label==1).sum()}")
    return df


def build_dfdm(output_dir, max_samples=None, quality='DFDM_cfr10'):
    frames_dir = output_dir / 'preprocessed' / 'frames'
    audio_dir  = output_dir / 'preprocessed' / 'audio'
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    video_dir   = DFDM_ROOT / 'video_data' / quality / 'test'
    all_vids    = sorted(video_dir.glob('*.mp4'))
    real_videos = [v for v in all_vids if '1original' in v.name]
    fake_videos = [v for v in all_vids if '1original' not in v.name]
    cap         = len(real_videos) if max_samples is None else max_samples // 2
    real_videos = real_videos[:cap]
    fake_videos = fake_videos[:cap]

    print(f"\n[DFDM] {len(real_videos)} real + {len(fake_videos)} fake ({quality})...")
    records = []
    uid_counter = 0

    for label, vlist in [(0, real_videos), (1, fake_videos)]:
        for vp in tqdm(vlist, desc=f"  label={label}"):
            frames = extract_face_frames(vp)
            if frames is None:
                continue
            waveform = extract_audio_from_video(vp)
            if waveform is None:
                waveform = torch.zeros(1, int(AUDIO_DURATION * SAMPLE_RATE))
            uid = str(uid_counter).zfill(6)
            save_sample(uid, frames, audio_to_mel(waveform), frames_dir, audio_dir)
            parts = vp.stem.split('_')
            records.append({'uid': uid, 'label': label, 'generator_id': label,
                            'split': 'test', 'success': True, 'source': 'dfdm',
                            'modify_video': label == 1, 'modify_audio': False,
                            'model_code': parts[-1] if parts else 'unknown'})
            uid_counter += 1

    df = pd.DataFrame(records)
    df.to_csv(str(output_dir / 'metadata.csv'), index=False)
    print(f"  Final — Real: {(df.label==0).sum()}  Fake: {(df.label==1).sum()}")
    return df


def build_celebdf(output_dir, max_samples=None):
    """Uses the official Celeb-DF-v2 test protocol (List_of_testing_videos.txt,
    518 videos) rather than an arbitrary sample — a standard, directly citable
    benchmark split. Note the official list's label convention is inverted vs.
    CMPI's own (0=fake, 1=real in the list; flipped here to CMPI's 1=fake)."""
    frames_dir = output_dir / 'preprocessed' / 'frames'
    audio_dir  = output_dir / 'preprocessed' / 'audio'
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    list_path = CELEBDF_ROOT / 'List_of_testing_videos.txt'
    entries = []
    with open(str(list_path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            official_label, rel_path = line.split(' ', 1)
            label = 1 - int(official_label)  # flip: CMPI convention 1=fake
            entries.append((label, rel_path))

    if max_samples is not None:
        reals = [e for e in entries if e[0] == 0][:max_samples // 2]
        fakes = [e for e in entries if e[0] == 1][:max_samples // 2]
        entries = reals + fakes

    print(f"\n[Celeb-DF-v2] {sum(1 for e in entries if e[0]==0)} real + "
          f"{sum(1 for e in entries if e[0]==1)} fake (official test protocol)...")
    records = []
    uid_counter = 0

    for label, rel_path in tqdm(entries, desc="  celebdf"):
        video_path = CELEBDF_ROOT / rel_path
        if not video_path.exists():
            continue
        frames = extract_face_frames(video_path)
        if frames is None:
            continue
        waveform = extract_audio_from_video(video_path)
        if waveform is None:
            waveform = torch.zeros(1, int(AUDIO_DURATION * SAMPLE_RATE))
        uid = str(uid_counter).zfill(6)
        save_sample(uid, frames, audio_to_mel(waveform), frames_dir, audio_dir)
        records.append({'uid': uid, 'label': label, 'generator_id': label,
                        'split': 'test', 'success': True, 'source': 'celebdf',
                        'modify_video': label == 1, 'modify_audio': False,
                        'orig_path': rel_path})
        uid_counter += 1

    df = pd.DataFrame(records)
    df.to_csv(str(output_dir / 'metadata.csv'), index=False)
    print(f"  Final — Real: {(df.label==0).sum()}  Fake: {(df.label==1).sum()}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all',
                   choices=['all', 'dftimit', 'lavdf', 'dfdm', 'celebdf'])
    p.add_argument('--output_root',
                   default='data/crossdataset')
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--quality', default='DFDM_cfr10',
                   choices=['DFDM_cfr10', 'DFDM_cfr23', 'DFDM_cfr0'])
    return p.parse_args()


def main():
    args     = parse_args()
    root     = Path(args.output_root)
    datasets = (['dftimit', 'lavdf', 'dfdm'] if args.dataset == 'all'
                else [args.dataset])
    for ds in datasets:
        out = root / ds
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n  Preprocessing: {ds.upper()}\n{'='*60}")
        if ds == 'dftimit':
            build_dftimit(out, args.max_samples)
        elif ds == 'lavdf':
            build_lavdf(out, args.max_samples)
        elif ds == 'dfdm':
            build_dfdm(out, args.max_samples, args.quality)
        elif ds == 'celebdf':
            build_celebdf(out, args.max_samples)
    print(f"\n[Done] All preprocessed data in: {root}")


if __name__ == '__main__':
    main()