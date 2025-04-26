"""
Vision Encoder: processes tumor slice images
Based on medical image pre-trained models
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    """
    Vision Encoder: processes tumor slice images
    Based on medical image pre-trained models
    """
    def __init__(self, output_dim=512, pretrained=True):
        super(VisionEncoder, self).__init__()
        # Using ViT as the base model, can be replaced with other vision models
        self.backbone = models.vit_b_16(pretrained=pretrained)
        
        # Replace the final classification layer
        self.backbone.heads = nn.Linear(self.backbone.hidden_dim, output_dim)
        
        # Tumor microenvironment recognition component
        self.tumor_env_module = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Tumor microenvironment features (assuming the input size has been adjusted to fit this module)
        tumor_env_features = self.tumor_env_module(x)
        
        # Fusion of features
        combined = torch.cat([backbone_features, tumor_env_features], dim=1)
        output = self.fusion(combined)
        
        return output, backbone_features, tumor_env_features