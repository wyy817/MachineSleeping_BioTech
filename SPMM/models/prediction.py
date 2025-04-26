"""
Prediction module: for tumor/cancer slide prediction and interpretability analysis
"""

import torch
import torch.nn as nn


class PredictionModule(nn.Module):
    """
    Prediction module: for tumor/cancer slide prediction and interpretability analysis
    """
    def __init__(self, input_dim=256, num_classes=4):
        super(PredictionModule, self).__init__()
        
        # Tumor classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_classes)
        )
        
        # Survival prediction
        self.survival_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Attention weights (for interpretability)
        self.attention_weights = nn.Parameter(torch.ones(input_dim) / input_dim)
        
    def forward(self, features):
        # Apply attention weights
        weighted_features = features * self.attention_weights
        
        # Classification prediction
        classification_output = self.classifier(weighted_features)
        
        # Survival prediction
        survival_output = self.survival_predictor(weighted_features)
        
        return classification_output, survival_output, self.attention_weights