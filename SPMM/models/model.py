"""
Overall Model Architecture: Spatial Transcriptomics Multimodal Medical Large Model
"""

import torch
import torch.nn as nn
from .spatial_encoder import SpatialEncoder
from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder
from .modality_alignment import ModalityAlignment
from .optimization import OptimizationModule
from .prediction import PredictionModule
from .mmseg_adapter import MMSEGAdapter


class SpatialMultimodalModel(nn.Module):
    """
    Overall Multimodal Medical Large Model
    """
    def __init__(
        self,
        spatial_input_dim=1024,
        text_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        vision_pretrained=True,
        feature_dim=512,
        hidden_dim=256,
        num_classes=4,
        dropout=0.1
    ):
        super(SpatialMultimodalModel, self).__init__()
        
        # Encoders
        self.spatial_encoder = SpatialEncoder(
            input_dim=spatial_input_dim, 
            hidden_dim=feature_dim, 
            output_dim=feature_dim, 
            dropout=dropout
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name, 
            output_dim=feature_dim
        )
        self.vision_encoder = VisionEncoder(
            output_dim=feature_dim, 
            pretrained=vision_pretrained
        )
        
        # Modality Alignment
        self.modality_alignment = ModalityAlignment(
            feature_dim=feature_dim, 
            hidden_dim=hidden_dim
        )
        
        # Optimization Module
        self.optimization_module = OptimizationModule(feature_dim=hidden_dim)
        
        # Prediction Module
        self.prediction_module = PredictionModule(
            input_dim=hidden_dim, 
            num_classes=num_classes
        )
        
        # MMSEG Adapter for Domain Generalization
        self.mmseg_adapter = MMSEGAdapter(input_dim=hidden_dim)
        
        # Uncertainty Quantification
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Softplus()  # Ensure uncertainty is positive
        )
        
    def forward(
        self, 
        spatial_x, 
        spatial_edge_index, 
        spatial_pos,
        text_input_ids, 
        text_attention_mask,
        vision_x,
        masked_indices=None,
        task=None
    ):
        # Feature Encoding
        spatial_features, spatial_raw, spatial_pos_features = self.spatial_encoder(
            spatial_x, spatial_edge_index, spatial_pos
        )
        
        text_features, text_task_features = self.text_encoder(
            text_input_ids, text_attention_mask, task
        )
        
        vision_features, vision_raw, vision_env_features = self.vision_encoder(vision_x)
        
        # Modality Alignment
        aligned_features, projections = self.modality_alignment(
            spatial_features, text_features, vision_features
        )
        
        # Apply MMSEG Adapter to improve domain generalization
        adapted_features = self.mmseg_adapter(aligned_features)
        
        # Optimization Module
        mask_pred, value_estimate = self.optimization_module(
            adapted_features, masked_indices
        )
        
        # Prediction
        classification_output, survival_output, attention_weights = self.prediction_module(
            adapted_features
        )
        
        # Uncertainty Estimation
        uncertainty = self.uncertainty_estimator(adapted_features)
        
        return {
            'classification': classification_output,
            'survival': survival_output,
            'attention_weights': attention_weights,
            'value_estimate': value_estimate,
            'uncertainty': uncertainty,
            'features': adapted_features,
            'projections': projections
        }