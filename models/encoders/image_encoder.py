"""
Image encoder using Vision Transformer (ViT) or ResNet.
"""

import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """Encode camera images into feature vectors."""

    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, output_dim: int = 256):
        super().__init__()
        self.backbone_name = backbone

        if 'resnet' in backbone:
            from torchvision import models
            if backbone == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                feat_dim = 512
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                feat_dim = 512
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                feat_dim = 2048
            else:
                self.backbone = models.resnet101(pretrained=pretrained)
                feat_dim = 2048
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif 'vit' in backbone:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        single_input = False
        if x.dim() == 4:
            single_input = True
            x = x.unsqueeze(1)

        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = self.projection(features)
        features = features.view(batch_size, seq_len, -1)
        if single_input:
            features = features.squeeze(1)
        return features
