"""
Spatial encoder module: processes molecular spatial position information
Based on graph neural networks to capture cell-molecule spatial relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SpatialEncoder(nn.Module):
    """
    Spatial encoder module: processes molecular spatial position information
    Based on graph neural networks to capture cell-molecule spatial relationships
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SpatialEncoder, self).__init__()
        self.gnn1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
        self.gnn2 = GATConv(hidden_dim * 4, hidden_dim, heads=2, dropout=dropout)
        self.gnn3 = GATConv(hidden_dim * 2, output_dim, heads=1, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Point cloud processing component
        self.pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, 1)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, x, edge_index, pos):
        # GNN processing
        x1 = self.gnn1(x, edge_index)
        x1 = F.relu(self.norm1(x1))
        x1 = self.dropout(x1)
        
        x2 = self.gnn2(x1, edge_index)
        x2 = F.relu(self.norm2(x2))
        x2 = self.dropout(x2)
        
        x3 = self.gnn3(x2, edge_index)
        
        # Point cloud processing
        pos_features = self.pointnet(pos.transpose(1, 0)).transpose(1, 0)
        
        # Fusion of GNN and point cloud features
        combined = torch.cat([x3, pos_features], dim=-1)
        output = self.fusion(combined)
        
        return output, x3, pos_features