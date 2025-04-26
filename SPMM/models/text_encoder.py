"""
Text Encoder module: processes spatial transcriptome knowledge, pathology reports and molecular information
Based on large biomedical language models
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Text Encoder module: processes spatial transcriptome knowledge, pathology reports and molecular information
    Based on large biomedical language models
    """
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", output_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Additional layers for spatial transcriptome specific tasks
        self.spatial_task_layers = nn.ModuleDict({
            'clustering': nn.Linear(output_dim, output_dim),
            'cell_typing': nn.Linear(output_dim, output_dim),
            'deconvolution': nn.Linear(output_dim, output_dim)
        })
        
    def forward(self, input_ids, attention_mask, task=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.out_proj(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        
        # If a specific task is specified, apply the corresponding task layer
        if task and task in self.spatial_task_layers:
            task_output = self.spatial_task_layers[task](pooled_output)
            return task_output, pooled_output
            
        return pooled_output, None