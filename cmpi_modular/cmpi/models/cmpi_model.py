"""Full CMPI + DIAM + TIL detector."""

import re

import torch
import torch.nn as nn

from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder
from .predictor import CrossModalPredictor
from .diam import DIAM
from .til import TemporalInconsistencyLocalization
from .score_norm import ScoreNormalizer
from .detection_head import DetectionHead


class CMPIModel(nn.Module):
    """
    forward(frames [B,T,3,H,W], audio [B,128,128]) returns a dict with
    z_v, z_a, pred_v, pred_a, eps_f, eps_b, C_i, A, til, til_alpha, frame_err,
    det_logits, det_probs.
    """

    def __init__(self, embed_dim=512, hidden_dim=1024, dropout=0.1, num_heads=8,
                 til_hidden=32, pretrained_backbones=True):
        super().__init__()
        self.visual_encoder = VisualEncoder(embed_dim, num_heads, dropout,
                                            pretrained=pretrained_backbones)
        self.audio_encoder = AudioEncoder(embed_dim, dropout,
                                          pretrained=pretrained_backbones)
        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.diam = DIAM()
        self.til = TemporalInconsistencyLocalization(til_hidden)
        self.score_norm = ScoreNormalizer()
        self.detection_head = DetectionHead(embed_dim, n_scores=3, dropout=dropout)

    def encoder_parameters(self):
        return list(self.visual_encoder.parameters()) + list(self.audio_encoder.parameters())

    def head_parameters(self):
        return (list(self.T_av.parameters()) + list(self.T_va.parameters()) +
                list(self.til.parameters()) + list(self.detection_head.parameters()))

    def forward(self, frames, audio, update_stats=True):
        vis = self.visual_encoder(frames)
        z_v, h_t = vis['video_embedding'], vis['frame_features']
        z_a = self.audio_encoder(audio)['audio_embedding']

        pred_v = self.T_av(z_a)
        pred_a = self.T_va(z_v)

        out = {'z_v': z_v, 'z_a': z_a, 'pred_v': pred_v, 'pred_a': pred_a}
        out.update(self.diam(z_v, z_a, pred_v, pred_a))
        out.update(self.til(pred_v, h_t))

        if self.training and update_stats:
            self.score_norm.update(ci=out['C_i'].detach(), asym=out['A'].detach(),
                                   til=out['til'].detach())

        det_in = torch.cat([z_v, z_a,
                            self.score_norm(out['C_i'], 'ci'),
                            self.score_norm(out['A'], 'asym'),
                            self.score_norm(out['til'], 'til')], dim=1)
        out['det_logits'] = self.detection_head(det_in)
        out['det_probs'] = torch.sigmoid(out['det_logits'])
        return out

    # ── checkpoint compatibility with the original single-file script ──
    _LEGACY_RULES = [
        (r'^til_scorer\.', 'til.scorer.'),
        (r'^detection_head\.(\d)', r'detection_head.net.\1'),
        (r'^(ci|asym|til)_(mean|std)$', r'score_norm.\1_\2'),
        (r'^n_updates$', 'score_norm.n_updates'),
    ]

    @classmethod
    def convert_legacy_state_dict(cls, state):
        """Map keys from scripts/train_cmpi_diam.py checkpoints to this layout."""
        out = {}
        for k, v in state.items():
            for pat, rep in cls._LEGACY_RULES:
                if re.match(pat, k):
                    k = re.sub(pat, rep, k)
                    break
            out[k] = v
        return out

    def load_checkpoint_state(self, state, strict=True):
        if any(k.startswith('til_scorer.') for k in state):
            state = self.convert_legacy_state_dict(state)
        return self.load_state_dict(state, strict=strict)


def build_model(cfg):
    m = cfg.get('model', {})
    return CMPIModel(embed_dim=m.get('embed_dim', 512),
                     hidden_dim=m.get('hidden_dim', 1024),
                     dropout=m.get('dropout', 0.1),
                     num_heads=m.get('num_heads', 8),
                     til_hidden=m.get('til_hidden', 32),
                     pretrained_backbones=m.get('pretrained_backbones', True))
