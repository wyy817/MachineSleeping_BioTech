"""
Configuration file: Store Model and Training Parameters
"""

# Model Configuration
MODEL_CONFIG = {
    # Spatial Encoder Configuration
    'spatial_encoder': {
        'input_dim': 1024,
        'hidden_dim': 512,
        'output_dim': 512,
        'dropout': 0.1,
    },
    
    # Text Encoder Configuration
    'text_encoder': {
        'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'output_dim': 512,
    },
    
    # Vision Encoder Configuration
    'vision_encoder': {
        'output_dim': 512,
        'pretrained': True,
    },
    
    # Modality Alignment Configuration
    'modality_alignment': {
        'feature_dim': 512,
        'hidden_dim': 256,
    },
    
    # Prediction Module Configuration
    'prediction': {
        'input_dim': 256,
        'num_classes': 4,  # Number of Tumor Classification Categories
    },
    
    # MMSEG Adapter Configuration
    'mmseg_adapter': {
        'input_dim': 256,
        'hidden_dim': 128,
    },
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'lr_scheduler_factor': 0.1,
    'lr_scheduler_patience': 5,
}

# Data Configuration
DATA_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
}

# Path Configuration
PATH_CONFIG = {
    'data_dir': './data',
    'model_dir': './saved_models',
    'log_dir': './logs',
    'visualization_dir': './visualizations',
}