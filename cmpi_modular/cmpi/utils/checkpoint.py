import shutil
from pathlib import Path

import torch


def save_checkpoint(state, out_dir, is_best):
    last = Path(out_dir) / 'last_checkpoint.pt'
    torch.save(state, str(last))
    if is_best:
        shutil.copyfile(str(last), str(Path(out_dir) / 'best_model.pt'))


def load_checkpoint(model, path, device, strict=True):
    """Load a checkpoint from this package or from scripts/train_cmpi_diam.py."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_checkpoint_state(ckpt.get('model_state', ckpt), strict=strict)
    return ckpt


def load_amdd_encoders(model, path, device):
    """Warm-start the visual/audio encoders from an AMDD checkpoint."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    for enc, prefix in [(model.visual_encoder, 'visual_encoder.'),
                        (model.audio_encoder, 'audio_encoder.')]:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            enc.load_state_dict(sub, strict=False)
            print(f"  {prefix}: {len(sub)} keys loaded")
        else:
            print(f"  {prefix}: not found, keeping ImageNet init")
