"""FakeAVCeleb dataset over preprocessed frame / mel-spectrogram arrays.

Expected layout:
    <preprocessed_dir>/frames/<uid>.npy   uint8 [T, H, W, 3]
    <preprocessed_dir>/audio/<uid>.npy    float [128, 128]
and a metadata CSV with columns: uid, split, success, label, generator_id.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import TRAIN_TRANSFORMS, EVAL_TRANSFORMS, normalize_mel


class FakeAVCelebDataset(Dataset):
    def __init__(self, metadata_csv, split, preprocessed_dir, transform=None, verbose=True):
        df = pd.read_csv(metadata_csv)
        df = df[(df['split'] == split) & (df['success'] == True)]
        self.df = df.reset_index(drop=True)
        self.split = split
        self.frames_dir = Path(preprocessed_dir) / 'frames'
        self.audio_dir = Path(preprocessed_dir) / 'audio'
        self.transform = transform or (TRAIN_TRANSFORMS if split == 'train' else EVAL_TRANSFORMS)
        if verbose:
            print(f"[{split}] {len(self.df)} samples "
                  f"(real={int((self.df.label == 0).sum())}, fake={int((self.df.label == 1).sum())})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['uid']).zfill(6)
        frames = np.load(str(self.frames_dir / f"{uid}.npy"))
        frames = torch.stack([self.transform(f) for f in frames], dim=0)   # [T, 3, H, W]
        mel = normalize_mel(np.load(str(self.audio_dir / f"{uid}.npy")))
        return {
            'frames': frames,
            'audio': torch.from_numpy(mel),
            'label': torch.tensor(int(row['label']), dtype=torch.long),
            'generator_id': torch.tensor(int(row['generator_id']), dtype=torch.long),
            'uid': uid,
        }
