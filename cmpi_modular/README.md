# CMPI (modular implementation)

A modular rewrite of `scripts/train_cmpi_diam.py`: each component is its own file, there is a config-driven training pipeline, and a single evaluation entry point.
The rewrite is behaviourally identical to the original. Checkpoints trained with the original script load here directly, and on the paper's trained checkpoint both versions produce identical outputs.

```
cmpi_modular/
├── configs/cmpi_diam_til.yaml     all hyper-parameters (paper settings)
├── train.py                       train on FakeAVCeleb, then test with the best checkpoint
├── evaluate.py                    in-domain + zero-shot cross-domain evaluation
└── cmpi/
    ├── models/
    │   ├── visual_encoder.py      ResNet-50 + temporal self-attention  -> z_v, h_t
    │   ├── audio_encoder.py       ResNet-18 (1-channel mel)            -> z_a
    │   ├── predictor.py           cross-modal MLPs T_av, T_va
    │   ├── diam.py                eps_f, eps_b, C_i = eps_f + eps_b, A = eps_f - eps_b
    │   ├── til.py                 per-frame errors + temporal attention -> TIL score
    │   ├── score_norm.py          EMA standardisation of C_i, A, TIL
    │   ├── detection_head.py      MLP over [z_v; z_a; C_i; A; TIL]
    │   └── cmpi_model.py          full model + legacy-checkpoint conversion
    ├── data/
    │   ├── transforms.py          frame augmentation, mel normalisation
    │   ├── fakeavceleb.py         FakeAVCeleb dataset
    │   ├── crossdataset.py        LAV-DF / DFDM / Celeb-DF dataset
    │   ├── samplers.py            generator-balanced sampler
    │   └── builders.py            DataLoader factories
    ├── engine/
    │   ├── trainer.py             AdamW (separate encoder/head LRs), warmup + cosine, AMP, checkpointing
    │   └── evaluator.py           inference + metrics
    ├── losses.py                  focal detection loss + real-only predictive loss
    └── utils/                     config, seeding, checkpoints, metrics, plotting
```

## Training

```bash
cd cmpi_modular
python train.py --config configs/cmpi_diam_til.yaml \
    data.metadata_csv=/path/to/metadata.csv \
    data.preprocessed_dir=/path/to/preprocessed \
    model.amdd_checkpoint=/path/to/amdd_encoder.pth   # optional; omit for ImageNet init
```

Any config value can be overridden on the command line as a dotted `key=value` pair (e.g. `train.epochs=10`, `train.batch_size=16`).
Resume an interrupted run with `--resume runs/cmpi_diam_til/last_checkpoint.pt`.

Outputs are written to `output_dir`: `best_model.pt`, `last_checkpoint.pt`, `history.json`, `training_curves.png` and `final_results.json`.

## Evaluation

```bash
# FakeAVCeleb test split + zero-shot LAV-DF, DFDM, Celeb-DF
python evaluate.py --checkpoint runs/cmpi_diam_til/best_model.pt \
    eval.crossdomain_root=/path/to/crossdataset
```

Use `--indomain` or `--crossdomain` to run only one of the two. The cross-domain data is produced by `../scripts/preprocess_crossdataset.py`.
On LAV-DF the report also breaks results down by manipulated modality (video-only, audio-only, both), together with the mean DIAM asymmetry `A`.

## Using the model directly

```python
import torch
from cmpi.models import CMPIModel
from cmpi.utils import load_checkpoint

model = CMPIModel(pretrained_backbones=False)
load_checkpoint(model, "best_model.pt", "cpu")
model.eval()

out = model(frames, mel)   # frames [B, T, 3, 224, 224], mel [B, 128, 128] in [-1, 1]
out["det_probs"]           # P(fake)
out["A"]                   # DIAM asymmetry: > 0 points to the visual stream, < 0 to the audio stream (in-domain)
out["til_alpha"]           # per-frame temporal attention
```
