"""Training sampler balancing generator classes (real counts as its own class)."""

import torch
from torch.utils.data import WeightedRandomSampler


def generator_balanced_sampler(dataset):
    df = dataset.df
    counts = df['generator_id'].value_counts().to_dict()
    n, k = len(df), len(counts)
    weights = torch.tensor([n / (k * counts[g]) for g in df['generator_id']],
                           dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
