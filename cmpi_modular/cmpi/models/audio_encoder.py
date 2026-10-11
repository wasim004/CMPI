"""Audio encoder: ResNet-18 adapted to single-channel mel spectrograms."""

import torch.nn as nn
import torchvision.models as tvm


class AudioEncoder(nn.Module):
    """
    Input:  mel spectrogram [B, 128, 128]
    Output: {'audio_embedding': [B, D]}
    """

    def __init__(self, embed_dim=512, dropout=0.1,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)

        # 1-channel stem initialised with the mean of the pretrained RGB filters
        old = resnet.conv1
        new = nn.Conv2d(1, 64, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=False)
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        resnet.conv1 = new

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        feats = self.backbone(x.unsqueeze(1)).flatten(1)            # [B, 512]
        return {'audio_embedding': self.projection(feats)}         # [B, D]
