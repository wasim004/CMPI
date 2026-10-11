"""Zero-shot evaluation datasets (LAV-DF, DFDM, Celeb-DF) produced by
scripts/preprocess_crossdataset.py.

Layout per dataset:  <root>/<name>/metadata.csv  and  <root>/<name>/preprocessed/{frames,audio}/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import EVAL_TRANSFORMS, normalize_mel


class CrossDomainDataset(Dataset):
    def __init__(self, metadata_csv, preprocessed_dir, verbose=True):
        df = pd.read_csv(metadata_csv)
        self.df = df[df['success'] == True].reset_index(drop=True)
        self.frames_dir = Path(preprocessed_dir) / 'frames'
        self.audio_dir = Path(preprocessed_dir) / 'audio'
        if verbose:
            print(f"  {len(self.df)} samples (real={int((self.df.label == 0).sum())}, "
                  f"fake={int((self.df.label == 1).sum())})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['uid']).zfill(6)
        frames = np.load(str(self.frames_dir / f"{uid}.npy"))
        frames = torch.stack([EVAL_TRANSFORMS(f) for f in frames], dim=0)
        mel_path = self.audio_dir / f"{uid}.npy"
        mel = normalize_mel(np.load(str(mel_path))) if mel_path.exists() \
            else np.zeros((128, 128), dtype=np.float32)
        item = {
            'frames': frames,
            'audio': torch.from_numpy(mel),
            'label': torch.tensor(int(row['label']), dtype=torch.long),
            'uid': uid,
        }
        # LAV-DF: which modality was manipulated (used for DIAM attribution)
        for col in ('modify_video', 'modify_audio'):
            if col in row:
                item[col] = bool(row[col])
        return item
