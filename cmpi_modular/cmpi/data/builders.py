"""DataLoader factories driven by the config dict."""

from pathlib import Path

from torch.utils.data import DataLoader

from .fakeavceleb import FakeAVCelebDataset
from .crossdataset import CrossDomainDataset
from .samplers import generator_balanced_sampler


def build_fakeavceleb_loaders(cfg):
    d, t = cfg['data'], cfg['train']
    kw = dict(batch_size=t['batch_size'], num_workers=d.get('num_workers', 4), pin_memory=True)
    train_ds = FakeAVCelebDataset(d['metadata_csv'], 'train', d['preprocessed_dir'])
    val_ds = FakeAVCelebDataset(d['metadata_csv'], 'val', d['preprocessed_dir'])
    test_ds = FakeAVCelebDataset(d['metadata_csv'], 'test', d['preprocessed_dir'])
    return {
        'train': DataLoader(train_ds, sampler=generator_balanced_sampler(train_ds),
                            drop_last=True, **kw),
        'val': DataLoader(val_ds, shuffle=False, **kw),
        'test': DataLoader(test_ds, shuffle=False, **kw),
    }


def build_crossdomain_loader(root, name, batch_size=8, num_workers=4):
    ds_dir = Path(root) / name
    meta, prep = ds_dir / 'metadata.csv', ds_dir / 'preprocessed'
    if not meta.exists() or not prep.exists():
        return None
    ds = CrossDomainDataset(str(meta), str(prep))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
