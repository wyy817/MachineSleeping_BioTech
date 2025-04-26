"""
Dataset module: Defines spatial transcriptomics multimodal datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms


class SpatialMultimodalDataset(Dataset):
    """Spatial transcriptomics multimodal dataset"""
    
    def __init__(self, spatial_data, image_data, text_data, labels=None, mode='train'):
        """
        Initialize the dataset
        
        Args:
            spatial_data: Spatial transcriptomics data, dictionary format
            image_data: Image data, dictionary format
            text_data: Text data, dictionary format
            labels: Sample labels
            mode: Dataset mode, 'train', 'val', or 'test'
        """
        self.spatial_data = spatial_data
        self.image_data = image_data
        self.text_data = text_data
        self.labels = labels
        self.mode = mode
        
        # Verify that the number of samples is consistent across all modalities
        if spatial_data['expression'].shape[0] != image_data['features'].shape[0]:
            # Handle missing image data
            self._align_data_sizes()
    
    def _align_data_sizes(self):
        """Align data sizes across different modalities"""
        spatial_size = self.spatial_data['expression'].shape[0]
        image_size = self.image_data['features'].shape[0]
        
        # Get valid image indices
        valid_indices = self.image_data['valid_indices']
        
        # Create zero tensors for missing image features
        if image_size < spatial_size:
            zero_features = torch.zeros(
                (spatial_size - image_size, self.image_data['features'].shape[1]),
                dtype=self.image_data['features'].dtype
            )
            self.image_data['features'] = torch.cat([self.image_data['features'], zero_features], dim=0)
            
            # Update valid index information
            if valid_indices is not None:
                self.image_data['valid_mask'] = torch.zeros(spatial_size, dtype=torch.bool)
                self.image_data['valid_mask'][valid_indices] = True
            else:
                self.image_data['valid_mask'] = torch.cat([
                    torch.ones(image_size, dtype=torch.bool),
                    torch.zeros(spatial_size - image_size, dtype=torch.bool)
                ])
        
        print(f"Aligned data sizes: spatial={spatial_size}, image={self.image_data['features'].shape[0]}")
    
    def __len__(self):
        """Return dataset size"""
        return self.spatial_data['expression'].shape[0]
    
    def __getitem__(self, idx):
        """
        Get data sample
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Sample data dictionary
        """
        # Get spatial transcriptomics data
        spatial_expr = self.spatial_data['expression'][idx]
        spatial_coords = self.spatial_data['coordinates'][idx]
        
        # Get image features
        image_features = self.image_data['features'][idx]
        image_valid = self.image_data.get('valid_mask', torch.ones(1, dtype=torch.bool))[idx]
        
        # Get text data
        text_input_ids = self.text_data['input_ids']
        text_attention_mask = self.text_data['attention_mask']
        
        # Build sample dictionary
        sample = {
            'spatial_expr': spatial_expr,
            'spatial_coords': spatial_coords,
            'image_features': image_features,
            'image_valid': image_valid,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
        }
        
        # If labels exist, add to sample
        if self.labels is not None:
            sample['label'] = self.labels[idx]
        
        return sample


class PatchwiseDataset(Dataset):
    """Patch-based dataset"""
    
    def __init__(self, patch_paths, coords, expression_data=None, labels=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            patch_paths: List of image patch paths
            coords: Corresponding spatial coordinates
            expression_data: Gene expression data
            labels: Labels
            transform: Image transformation function
        """
        self.patch_paths = patch_paths
        self.coords = coords
        self.expression_data = expression_data
        self.labels = labels
        
        # Default transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Return dataset size"""
        return len(self.patch_paths)
    
    def __getitem__(self, idx):
        """Get data sample"""
        # Load image
        image_path = self.patch_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        
        # Build sample dictionary
        sample = {
            'image': image,
            'coords': self.coords[idx] if self.coords is not None else None,
        }
        
        # Add gene expression data (if available)
        if self.expression_data is not None:
            sample['expression'] = self.expression_data[idx]
        
        # Add labels (if available)
        if self.labels is not None:
            sample['label'] = self.labels[idx]
        
        return sample


class MultiSlideDataset(Dataset):
    """Multi-slide dataset for processing slides from multiple samples"""
    
    def __init__(self, sample_data_dict, mode='train', transform=None):
        """
        Initialize the dataset
        
        Args:
            sample_data_dict: Dictionary containing data for multiple samples, each with spatial transcriptomics and image data
            mode: Dataset mode 'train', 'val', or 'test'
            transform: Image transformation function
        """
        self.sample_data_dict = sample_data_dict
        self.mode = mode
        self.transform = transform
        
        # Create sample index mapping
        self.sample_ids = list(sample_data_dict.keys())
        self.index_mapping = []
        
        for sample_id in self.sample_ids:
            sample_data = sample_data_dict[sample_id]
            num_spots = len(sample_data['expression'])
            for i in range(num_spots):
                self.index_mapping.append((sample_id, i))
    
    def __len__(self):
        """Return dataset size"""
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        """Get data sample"""
        sample_id, spot_idx = self.index_mapping[idx]
        sample_data = self.sample_data_dict[sample_id]
        
        # Get expression data
        expression = torch.tensor(sample_data['expression'][spot_idx], dtype=torch.float32)
        
        # Get coordinates
        coords = torch.tensor(sample_data['coordinates'][spot_idx], dtype=torch.float32)
        
        # Get image patch (if available)
        image = None
        if 'images' in sample_data and spot_idx < len(sample_data['images']):
            image = sample_data['images'][spot_idx]
            if self.transform:
                image = self.transform(image)
        
        # Get labels (if available)
        label = None
        if 'labels' in sample_data:
            if isinstance(sample_data['labels'], list):
                label = sample_data['labels'][spot_idx]
            else:
                label = sample_data['labels']
        
        # Build sample dictionary
        sample = {
            'sample_id': sample_id,
            'spot_idx': spot_idx,
            'expression': expression,
            'coords': coords,
            'image': image,
            'label': label
        }
        
        return sample


class GraphSpatialTemporalDataset(Dataset):
    """Graph-structured spatial-temporal dataset for processing spatial transcriptomics data that changes over time"""
    
    def __init__(self, time_series_data, graph_data, window_size=3, stride=1):
        """
        Initialize the dataset
        
        Args:
            time_series_data: Time series data, format {time_point: spatial_data}
            graph_data: Graph structure data, including nodes and edges information
            window_size: Time window size
            stride: Sliding window step size
        """
        self.time_series_data = time_series_data
        self.graph_data = graph_data
        self.window_size = window_size
        self.stride = stride
        
        # Get all time points
        self.time_points = sorted(list(time_series_data.keys()))
        
        # Create time windows
        self.windows = []
        for i in range(0, len(self.time_points) - window_size + 1, stride):
            self.windows.append(self.time_points[i:i+window_size])
    
    def __len__(self):
        """Return dataset size"""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get data sample"""
        window = self.windows[idx]
        
        # Collect time series data within the window
        window_data = []
        for time_point in window:
            window_data.append(self.time_series_data[time_point])
        
        # Get graph structure
        edge_index = self.graph_data['edge_index']
        
        # Build sample dictionary
        sample = {
            'time_window': window,
            'window_data': window_data,
            'edge_index': edge_index
        }
        
        return sample


class SpatialOmicsSlideDataset(Dataset):
    """Dataset integrating spatial transcriptomics and whole slide images"""
    
    def __init__(self, 
                spatial_data, 
                wsi_image, 
                clinical_text=None, 
                labels=None, 
                patch_size=256, 
                transform=None):
        """
        Initialize the dataset
        
        Args:
            spatial_data: Spatial transcriptomics data, including expression matrix and coordinates
            wsi_image: Whole slide image
            clinical_text: Clinical text data
            labels: Labels
            patch_size: Image patch size
            transform: Image transformation function
        """
        self.spatial_data = spatial_data
        self.wsi_image = wsi_image
        self.clinical_text = clinical_text
        self.labels = labels
        self.patch_size = patch_size
        
        # Default transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Extract image patches
        self.patches = self._extract_patches()
    
    def _extract_patches(self):
        """Extract image patches from WSI image corresponding to spatial coordinates"""
        patches = []
        half_size = self.patch_size // 2
        coords = self.spatial_data['coordinates']
        
        if self.wsi_image is not None:
            height, width = self.wsi_image.shape[:2]
            
            for x, y in coords:
                # Convert to integer coordinates
                x, y = int(x), int(y)
                
                # Check boundaries
                if (x - half_size >= 0 and x + half_size < width and 
                    y - half_size >= 0 and y + half_size < height):
                    
                    # Extract image patch
                    patch = self.wsi_image[y - half_size:y + half_size, x - half_size:x + half_size]
                    patch = Image.fromarray(patch)
                    
                    # Apply transformation
                    if self.transform:
                        patch = self.transform(patch)
                    
                    patches.append(patch)
                else:
                    # If out of boundaries, add None
                    patches.append(None)
        else:
            # If no WSI image, add None
            patches = [None] * len(coords)
        
        return patches
    
    def __len__(self):
        """Return dataset size"""
        return len(self.spatial_data['expression'])
    
    def __getitem__(self, idx):
        """Get data sample"""
        # Get spatial transcriptomics data
        expression = torch.tensor(self.spatial_data['expression'][idx], dtype=torch.float32)
        coordinates = torch.tensor(self.spatial_data['coordinates'][idx], dtype=torch.float32)
        
        # Get image patch
        patch = self.patches[idx]
        
        # Build sample dictionary
        sample = {
            'expression': expression,
            'coordinates': coordinates,
            'patch': patch
        }
        
        # Add clinical text (if available)
        if self.clinical_text is not None:
            sample['clinical_text'] = self.clinical_text
        
        # Add labels (if available)
        if self.labels is not None:
            sample['label'] = self.labels[idx] if isinstance(self.labels, list) else self.labels
        
        return sample


def collate_spatial_multimodal_batch(batch):
    """
    Custom batch processing function for spatial transcriptomics multimodal data
    
    Args:
        batch: Batch data list
        
    Returns:
        collated_batch: Integrated batch data
    """
    # Extract fields from the batch
    spatial_expr = [item['spatial_expr'] for item in batch]
    spatial_coords = [item['spatial_coords'] for item in batch]
    image_features = [item['image_features'] for item in batch]
    image_valid = [item['image_valid'] for item in batch]
    text_input_ids = [item['text_input_ids'] for item in batch]
    text_attention_mask = [item['text_attention_mask'] for item in batch]
    
    # Integrate batch data
    collated_batch = {
        'spatial_expr': torch.stack(spatial_expr),
        'spatial_coords': torch.stack(spatial_coords),
        'image_features': torch.stack(image_features),
        'image_valid': torch.stack(image_valid),
        'text_input_ids': torch.stack(text_input_ids),
        'text_attention_mask': torch.stack(text_attention_mask),
    }
    
    # If labels exist, also integrate labels
    if 'label' in batch[0]:
        labels = [item['label'] for item in batch]
        collated_batch['labels'] = torch.tensor(labels)
    
    return collated_batch


def collate_graph_batch(batch):
    """
    Custom batch processing function for graph structure data
    
    Args:
        batch: Batch data list
        
    Returns:
        collated_batch: Integrated batch data
    """
    from torch_geometric.data import Batch
    
    # Convert samples to PyTorch Geometric Data objects
    data_list = []
    for item in batch:
        # Extract node features and edge indices
        x = item['node_features']
        edge_index = item['edge_index']
        
        # Create Data object
        data = torch.geometric.data.Data(x=x, edge_index=edge_index)
        
        # Add additional attributes
        for key, value in item.items():
            if key not in ['node_features', 'edge_index']:
                data[key] = value
        
        data_list.append(data)
    
    # Use Batch.from_data_list to merge multiple graphs
    collated_batch = Batch.from_data_list(data_list)
    
    return collated_batch


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=None):
    """
    Create data loader
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_fn: Batch processing function
        
    Returns:
        dataloader: Data loader
    """
    # Choose appropriate collate_fn for dataset type
    if collate_fn is None:
        if isinstance(dataset, SpatialMultimodalDataset):
            collate_fn = collate_spatial_multimodal_batch
        elif 'Graph' in dataset.__class__.__name__:
            try:
                import torch_geometric
                collate_fn = collate_graph_batch
            except ImportError:
                print("Warning: torch_geometric not found, using default collate_fn")
                collate_fn = torch.utils.data.dataloader.default_collate
        else:
            collate_fn = torch.utils.data.dataloader.default_collate
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader