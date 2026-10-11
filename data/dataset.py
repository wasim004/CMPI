import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import torchvision.transforms as transforms

# ── Config ────────────────────────────────────────────────────────────────
PREPROCESSED_DIR = "data/preprocessed"

# ── Domain mapping (for domain-adversarial training) ────────────────────────
# Derived from the `source` column prefix rather than stored in metadata.csv,
# so no migration of existing rows is needed.
def source_to_domain(source):
    if source.startswith('favceleb'):
        return 0
    if source.startswith('dftimit'):
        return 1
    if source.startswith('lavdf'):
        return 2
    if source.startswith('dfdm'):
        return 3
    raise ValueError(f"Unknown source prefix for domain mapping: {source!r}")

# ── Augmentation transforms ───────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class DeepfakeDataset(Dataset):
    def __init__(self, metadata_csv, split='train',
                 preprocessed_dir=PREPROCESSED_DIR):
        self.df = pd.read_csv(metadata_csv)

        # Filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        # Filter to successfully preprocessed records only
        self.df = self.df[self.df['success'] == True].reset_index(drop=True)

        self.split = split
        self.frames_dir = Path(preprocessed_dir) / 'frames'
        self.audio_dir  = Path(preprocessed_dir) / 'audio'
        self.transform  = TRAIN_TRANSFORMS if split == 'train' else VAL_TRANSFORMS

        print(f"[{split}] Loaded {len(self.df)} samples")
        print(f"  Real: {(self.df.label == 0).sum()}, "
              f"Fake: {(self.df.label == 1).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #uid = row['uid']
        uid = str(row['uid']).zfill(6)

        # ── Load frames ───────────────────────────────────────────────────
        frames_path = self.frames_dir / f"{uid}.npy"
        frames = np.load(str(frames_path))  # [T, H, W, C] uint8

        # Apply transforms to each frame
        frame_tensors = []
        for i in range(frames.shape[0]):
            frame = frames[i]  # [H, W, C]
            frame_tensors.append(self.transform(frame))
        frames_tensor = torch.stack(frame_tensors, dim=0)  # [T, C, H, W]

        # ── Load audio ────────────────────────────────────────────────────
        audio_path = self.audio_dir / f"{uid}.npy"
        mel = np.load(str(audio_path)).astype(np.float32)  # [128, 128]

        # Normalize mel spectrogram to [-1, 1]
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max > mel_min:
            mel = 2.0 * (mel - mel_min) / (mel_max - mel_min) - 1.0
        mel_tensor = torch.tensor(mel, dtype=torch.float32)  # [128, 128]

        # ── Labels ────────────────────────────────────────────────────────
        label        = torch.tensor(row['label'], dtype=torch.long)
        generator_id = torch.tensor(row['generator_id'], dtype=torch.long)
        domain_id    = torch.tensor(source_to_domain(row['source']), dtype=torch.long)

        return {
            'frames':       frames_tensor,    # [T, C, H, W]
            'audio':        mel_tensor,        # [128, 128]
            'label':        label,             # scalar
            'generator_id': generator_id,      # scalar, -1 = unattributed (masked from attr/PAAL loss)
            'domain_id':    domain_id,         # scalar, 0=favceleb 1=dftimit 2=lavdf
            'uid':          uid,
        }


def get_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler that balances:
    1. Real vs Fake (binary)
    2. Generator classes (attribution)
    Combined weight per sample.
    """
    df = dataset.df

    # Per-generator weights
    generator_counts = df['generator_id'].value_counts().to_dict()
    n_total = len(df)
    n_classes = len(generator_counts)

    weights = []
    for _, row in df.iterrows():
        gid   = row['generator_id']
        w     = n_total / (n_classes * generator_counts[gid])
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def get_dataloaders(metadata_csv, batch_size=16, num_workers=4,
                    preprocessed_dir=PREPROCESSED_DIR):
    """Returns train, val, test dataloaders."""

    train_dataset = DeepfakeDataset(metadata_csv, split='train',
                                    preprocessed_dir=preprocessed_dir)
    val_dataset   = DeepfakeDataset(metadata_csv, split='val',
                                    preprocessed_dir=preprocessed_dir)
    test_dataset  = DeepfakeDataset(metadata_csv, split='test',
                                    preprocessed_dir=preprocessed_dir)

    # Weighted sampler for training
    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    METADATA_CSV = "data/metadata.csv"

    train_loader, val_loader, test_loader = get_dataloaders(
        METADATA_CSV, batch_size=4, num_workers=2
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch frames shape:  {batch['frames'].shape}")
    print(f"Batch audio shape:   {batch['audio'].shape}")
    print(f"Batch labels:        {batch['label']}")
    print(f"Batch generator_ids: {batch['generator_id']}")