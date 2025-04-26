"""
Attention Visualization Module: Visualize the attention weights and feature importance of models
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


class AttentionVisualizer:
    """Visualization of attention and feature importance"""
    
    def __init__(self, output_dir='./visualizations'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_attention_weights(self, attention_weights, feature_names, output_path=None):
        """
        Visualize attention weights
        
        Args:
            attention_weights: Attention weights tensor
            feature_names: List of feature names
            output_path: Output file path
            
        Returns:
            sorted_indices: Feature indices sorted by weight
            sorted_weights: Sorted weight values
        """
        plt.figure(figsize=(12, 8))
        
        # Ensure input is a NumPy array
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.cpu().detach().numpy()
            
        # Sort for better visualization
        sorted_indices = np.argsort(attention_weights)[::-1]
        sorted_weights = attention_weights[sorted_indices]
        
        # Ensure feature names match weights length
        if len(feature_names) < len(attention_weights):
            feature_names = list(feature_names) + [f"Feature_{i}" for i in range(len(feature_names), len(attention_weights))]
        elif len(feature_names) > len(attention_weights):
            feature_names = feature_names[:len(attention_weights)]
            
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Only display top 30 features with highest weights
        top_k = min(30, len(sorted_weights))
        plt.barh(range(top_k), sorted_weights[:top_k], color='skyblue')
        plt.yticks(range(top_k), sorted_names[:top_k])
        plt.xlabel('Attention Weight')
        plt.title('Feature Importance based on Attention Weights')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return sorted_indices, sorted_weights
    
    def visualize_attention_heatmap(self, attention_matrix, row_labels=None, col_labels=None, output_path=None):
        """
        Visualize attention heatmap
        
        Args:
            attention_matrix: Attention matrix
            row_labels: Row labels
            col_labels: Column labels
            output_path: Output file path
        """
        plt.figure(figsize=(12, 10))
        
        # Ensure input is a NumPy array
        if torch.is_tensor(attention_matrix):
            attention_matrix = attention_matrix.cpu().detach().numpy()
        
        # Create heatmap
        sns.heatmap(attention_matrix, annot=False, cmap='viridis', 
                   xticklabels=col_labels if col_labels else False,
                   yticklabels=row_labels if row_labels else False)
        
        plt.title('Attention Matrix Heatmap')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_multihead_attention(self, multihead_attention, label_mapping=None, output_path=None):
        """
        Visualize multi-head attention
        
        Args:
            multihead_attention: Multi-head attention tensor with shape [n_heads, seq_len, seq_len]
            label_mapping: Mapping from positions to labels
            output_path: Output file path
        """
        if torch.is_tensor(multihead_attention):
            multihead_attention = multihead_attention.cpu().detach().numpy()
        
        n_heads = multihead_attention.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
        
        for i in range(n_heads):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            im = ax.imshow(multihead_attention[i], cmap='viridis')
            ax.set_title(f'Head {i+1}')
            
            # Set labels
            if label_mapping:
                seq_len = multihead_attention.shape[1]
                if len(label_mapping) >= seq_len:
                    positions = list(range(seq_len))
                    labels = [label_mapping[p] for p in positions]
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels)
        
        plt.tight_layout()
        fig.colorbar(im, ax=axes.ravel().tolist())
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_feature_embeddings(self, features, labels=None, method='tsne', output_path=None):
        """
        Visualize feature embeddings using dimensionality reduction techniques
        
        Args:
            features: Feature matrix
            labels: Label array (optional)
            method: Dimensionality reduction method ('tsne' or 'pca')
            output_path: Output file path
        """
        plt.figure(figsize=(10, 8))
        
        # Ensure input is a NumPy array
        if torch.is_tensor(features):
            features = features.cpu().detach().numpy()
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # pca
            reducer = PCA(n_components=2)
        
        # Reduce to 2D
        reduced_features = reducer.fit_transform(features)
        
        # Plot scatter
        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            
            # Get unique labels
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                           color=colors(i), label=f'Class {label}', alpha=0.7)
            
            plt.legend()
        else:
            plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
        
        plt.title(f'Feature Embeddings ({method.upper()})')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return reduced_features
    
    def visualize_feature_correlations(self, features, feature_names=None, output_path=None):
        """
        Visualize correlations between features
        
        Args:
            features: Feature matrix
            feature_names: Feature names
            output_path: Output file path
        """
        plt.figure(figsize=(12, 10))
        
        # Ensure input is a NumPy array
        if torch.is_tensor(features):
            features = features.cpu().detach().numpy()
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features, rowvar=False)
        
        # Create heatmap
        if feature_names and len(feature_names) == features.shape[1]:
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                       xticklabels=feature_names, yticklabels=feature_names)
        else:
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_uncertainty(self, uncertainty, class_names=None, output_path=None):
        """
        Visualize prediction uncertainty
        
        Args:
            uncertainty: Uncertainty tensor with shape [batch_size, n_classes]
            class_names: Class names
            output_path: Output file path
            
        Returns:
            uncertainty_np: Uncertainty data in NumPy format
        """
        plt.figure(figsize=(10, 6))
        
        # Ensure input is a NumPy array
        if torch.is_tensor(uncertainty):
            uncertainty = uncertainty.cpu().detach().numpy()
            
        # If batch data, take the mean
        if len(uncertainty.shape) > 1:
            uncertainty = np.mean(uncertainty, axis=0)
        
        # Set class names
        if class_names is None or len(class_names) != len(uncertainty):
            class_names = [f"Class {i}" for i in range(len(uncertainty))]
        
        # Create bar chart
        plt.bar(range(len(uncertainty)), uncertainty, color='salmon')
        plt.xticks(range(len(uncertainty)), class_names, rotation=45, ha='right')
        plt.ylabel('Uncertainty (Variance)')
        plt.title('Prediction Uncertainty by Class')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return uncertainty
    
    def visualize_gradients(self, gradients, feature_names=None, top_k=20, output_path=None):
        """
        Visualize feature gradients (Grad-CAM like method)
        
        Args:
            gradients: Feature gradients
            feature_names: Feature names
            top_k: Number of top features to display
            output_path: Output file path
        """
        plt.figure(figsize=(12, 8))
        
        # Ensure input is a NumPy array
        if torch.is_tensor(gradients):
            gradients = gradients.cpu().detach().numpy()
        
        # Take absolute value of gradients
        grad_magnitude = np.abs(gradients)
        
        # Sort
        sorted_indices = np.argsort(grad_magnitude)[::-1]
        sorted_grads = grad_magnitude[sorted_indices]
        
        # Set feature names
        if feature_names is None or len(feature_names) != len(gradients):
            feature_names = [f"Feature {i}" for i in range(len(gradients))]
        
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Display top k features
        k = min(top_k, len(sorted_grads))
        plt.barh(range(k), sorted_grads[:k], color='lightgreen')
        plt.yticks(range(k), sorted_names[:k])
        plt.xlabel('Gradient Magnitude')
        plt.title('Feature Importance based on Gradient Magnitude')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()