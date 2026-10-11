"""Frame augmentations and mel-spectrogram normalisation."""

import numpy as np
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORMS = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def normalize_mel(mel):
    """Min-max scale a mel spectrogram to [-1, 1]."""
    mel = mel.astype(np.float32)
    lo, hi = mel.min(), mel.max()
    if hi > lo:
        mel = 2.0 * (mel - lo) / (hi - lo) - 1.0
    return mel
