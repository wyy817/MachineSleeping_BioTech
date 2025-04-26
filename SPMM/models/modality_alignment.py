"""
Modality Alignment Module: Integrates spatial, text, and visual encoding information
"""

import torch
import torch.nn as nn


class ModalityAlignment(nn.Module):
    """
    Modality Alignment Module: Integrates three types of encoding information
    """
    def __init__(self, feature_dim=512, hidden_dim=256):
        super(ModalityAlignment, self).__init__()
        
        # Cross-attention module - Text to Vision
        self.text_to_vision_attn = nn.MultiheadAttention(feature_dim, 8)
        # Cross-attention module - Vision to Text
        self.vision_to_text_attn = nn.MultiheadAttention(feature_dim, 8)
        # Cross-attention module - Spatial to Text/Vision
        self.spatial_to_tv_attn = nn.MultiheadAttention(feature_dim, 8)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # # Contrastive learning projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, spatial_features, text_features, vision_features):
        # Text to vision attention
        text_vision_attn, _ = self.text_to_vision_attn(
            text_features.unsqueeze(0), 
            vision_features.unsqueeze(0), 
            vision_features.unsqueeze(0)
        )
        
        # Vision to text attention
        vision_text_attn, _ = self.vision_to_text_attn(
            vision_features.unsqueeze(0), 
            text_features.unsqueeze(0), 
            text_features.unsqueeze(0)
        )
        
        # Spatial to text/vision attention
        tv_combined = (text_vision_attn + vision_text_attn) / 2
        spatial_tv_attn, _ = self.spatial_to_tv_attn(
            spatial_features.unsqueeze(0),
            tv_combined,
            tv_combined
        )
        
        # Feature fusion
        combined_features = torch.cat([
            spatial_features, 
            text_vision_attn.squeeze(0), 
            vision_text_attn.squeeze(0)
        ], dim=1)
        
        aligned_features = self.fusion_layer(combined_features)
        projection = self.projection_head(aligned_features)
        
        return aligned_features, projection