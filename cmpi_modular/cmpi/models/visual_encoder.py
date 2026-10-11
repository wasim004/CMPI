"""Visual encoder: ResNet-50 per-frame backbone + temporal self-attention."""

import torch.nn as nn
import torchvision.models as tvm


class TemporalAttention(nn.Module):
    """Multi-head self-attention over the frame axis, followed by mean pooling."""

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x.mean(dim=1), x                      # [B, D], [B, T, D]


class VisualEncoder(nn.Module):
    """
    Input:  frames [B, T, 3, H, W]
    Output: {'video_embedding': [B, D], 'frame_features': [B, T, D]}
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(2048, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W)).flatten(1)   # [B*T, 2048]
        feats = self.projection(feats).view(B, T, -1)              # [B, T, D]
        video_emb, frame_feats = self.temporal_attn(feats)
        return {'video_embedding': video_emb, 'frame_features': frame_feats}
