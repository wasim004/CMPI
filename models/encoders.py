import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TemporalAttention(nn.Module):
    """Lightweight temporal attention for frame sequences."""
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        # Global temporal pooling
        video_emb = x.mean(dim=1)  # [B, D]
        return video_emb, x


class VisualEncoder(nn.Module):
    """
    ResNet50 backbone with temporal attention.
    Input:  [B, T, C, H, W]
    Output: video_embedding [B, D], frame_features [B, T, D]
    """
    def __init__(self, embed_dim=512, num_heads=8,
                 dropout=0.1, freeze_backbone=False):
        super().__init__()

        # Pretrained ResNet50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Process all frames through backbone
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)          # [B*T, 2048, 1, 1]
        features = features.flatten(1)       # [B*T, 2048]
        features = self.projection(features) # [B*T, embed_dim]
        features = features.view(B, T, -1)   # [B, T, embed_dim]

        # Temporal attention
        video_emb, frame_features = self.temporal_attn(features)

        return {
            'video_embedding': video_emb,      # [B, embed_dim]
            'frame_features':  frame_features, # [B, T, embed_dim]
        }


class AudioEncoder(nn.Module):
    """
    Pretrained ResNet18 adapted for mel spectrograms.
    Much stronger than a scratch CNN for audio forensics.
    Input:  [B, 128, 128] mel spectrogram
    Output: audio_embedding [B, embed_dim]
    """
    def __init__(self, embed_dim=512, dropout=0.1, freeze_backbone=False):
        super().__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv to accept 1-channel input (mel spectrogram)
        # Keep pretrained weights by averaging across RGB channels
        original_conv = resnet.conv1
        new_conv = nn.Conv2d(
            1, 64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # Initialize with mean of pretrained RGB weights
        new_conv.weight.data = original_conv.weight.data.mean(
            dim=1, keepdim=True
        )
        resnet.conv1 = new_conv

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.embed_dim = embed_dim

    def forward(self, x):
        # x: [B, 128, 128]
        x = x.unsqueeze(1)                   # [B, 1, 128, 128]
        features = self.backbone(x)           # [B, 512, 1, 1]
        features = features.flatten(1)        # [B, 512]
        audio_emb = self.projection(features) # [B, embed_dim]

        return {
            'audio_embedding': audio_emb,     # [B, embed_dim]
        }


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing encoders on {device}")

    # Visual encoder
    vis_enc = VisualEncoder(embed_dim=512).to(device)
    dummy_frames = torch.randn(2, 16, 3, 224, 224).to(device)
    vis_out = vis_enc(dummy_frames)
    print(f"Visual embedding: {vis_out['video_embedding'].shape}")
    print(f"Frame features:   {vis_out['frame_features'].shape}")

    # Audio encoder
    aud_enc = AudioEncoder(embed_dim=512).to(device)
    dummy_audio = torch.randn(2, 128, 128).to(device)
    aud_out = aud_enc(dummy_audio)
    print(f"Audio embedding:  {aud_out['audio_embedding'].shape}")

    # Parameter counts
    vis_params = sum(p.numel() for p in vis_enc.parameters()) / 1e6
    aud_params = sum(p.numel() for p in aud_enc.parameters()) / 1e6
    print(f"\nVisual encoder params: {vis_params:.1f}M")
    print(f"Audio encoder params:  {aud_params:.1f}M")