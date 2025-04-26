"""
MMSEG Adapter: For improving the domain generalization ability of models
"""

import torch
import torch.nn as nn


class MMSEGAdapter(nn.Module):
    """
    MMSEG Adapter: For improving the domain generalization ability of models
    """
    def __init__(self, input_dim=256, hidden_dim=128):
        super(MMSEGAdapter, self).__init__()
        
        # Adapter layers
        self.adapter_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Scaling factor for residual connection
        self.scale = nn.Parameter(torch.ones([]) * 0.1)
        
    def forward(self, x):
        # Residual connection
        return x + self.scale * self.adapter_layers(x)