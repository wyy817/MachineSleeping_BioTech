"""
Multimodal Visualization Module: Visualizing multimodal feature integration and correlation
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.manifold import TSNE
import os


class MultimodalVisualizer:
    """Multimodal Feature Visualization"""
    
    def __init__(self, output_dir='./visualizations/multimodal'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_multimodal_integration(self, spatial_features, text_features, vision_features, output_path=None):
        """
        Visualize multimodal feature integration
        
        Args:
            spatial_features: Spatial features
            text_features: Text features
            vision_features: Vision features
            output_path: Output file path
            
        Returns:
            embedding: t-SNE embedding
            labels: Modality labels
        """
        # Ensure data is on CPU
        spatial_feats = spatial_features.cpu().detach().numpy() if torch.is_tensor(spatial_features) else spatial_features
        text_feats = text_features.cpu().detach().numpy() if torch.is_tensor(text_features) else text_features
        vision_feats = vision_features.cpu().detach().numpy() if torch.is_tensor(vision_features) else vision_features
        
        # Adjust dimensions if needed
        if len(text_feats.shape) == 3:  # [batch_size, seq_len, hidden_size]
            text_feats = text_feats[:, 0, :]  # Take the representation of the CLS token
        
        # Combine features for t-SNE dimensionality reduction
        combined_features = np.vstack([spatial_feats, text_feats, vision_feats])
        
        # Create labels to distinguish different modalities
        labels = np.array(['Spatial'] * len(spatial_feats) + 
                         ['Text'] * len(text_feats) + 
                         ['Vision'] * len(vision_feats))
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding = tsne.fit_transform(combined_features)
        
        # Assign different colors to each modality
        colors = {'Spatial': 'blue', 'Text': 'green', 'Vision': 'red'}
        
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c=colors[label], label=label, alpha=0.7)
            
        plt.legend()
        plt.title('t-SNE Projection of Multimodal Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return embedding, labels
    
    def visualize_cross_modal_attention(self, attention_weights, modality_names, output_path=None):
        """
        Visualize cross-modal attention weights
        
        Args:
            attention_weights: Attention weight matrix, shape [num_modalities, num_modalities]
            modality_names: List of modality names
            output_path: Output file path
        """
        # Ensure data is on CPU
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.cpu().detach().numpy()
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(attention_weights, annot=True, fmt='.2f', cmap='YlGnBu',
                  xticklabels=modality_names, yticklabels=modality_names)
        
        plt.title('Cross-Modal Attention Weights')
        plt.xlabel('Target Modality')
        plt.ylabel('Source Modality')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_modality_contributions(self, contributions, class_names, modality_names, output_path=None):
        """
        Visualize different modalities' contributions to classification tasks
        
        Args:
            contributions: Contribution weights, shape [num_classes, num_modalities]
            class_names: Class names
            modality_names: Modality names
            output_path: Output file path
        """
        # Ensure data is on CPU
        if torch.is_tensor(contributions):
            contributions = contributions.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(contributions, annot=True, fmt='.2f', cmap='YlGnBu',
                  xticklabels=modality_names, yticklabels=class_names)
        
        plt.title('Modality Contributions to Classification')
        plt.xlabel('Modality')
        plt.ylabel('Class')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_feature_alignment(self, spatial_features, text_features, vision_features, 
                                  labels=None, output_path=None):
        """
        Visualize feature alignment
        
        Args:
            spatial_features: Spatial features
            text_features: Text features
            vision_features: Vision features
            labels: Sample labels
            output_path: Output file path
        """
        # Ensure data is on CPU
        spatial_feats = spatial_features.cpu().detach().numpy() if torch.is_tensor(spatial_features) else spatial_features
        text_feats = text_features.cpu().detach().numpy() if torch.is_tensor(text_features) else text_features
        vision_feats = vision_features.cpu().detach().numpy() if torch.is_tensor(vision_features) else vision_features
        
        if labels is not None and torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # Create t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        
        # Perform separate dimensionality reduction for each modality
        spatial_2d = tsne.fit_transform(spatial_feats)
        text_2d = tsne.fit_transform(text_feats)
        vision_2d = tsne.fit_transform(vision_feats)
        
        # Create a 3x2 grid for visualization
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Define color mapping
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            # First column: t-SNE for each modality separately
            for i, (data_2d, title) in enumerate(zip([spatial_2d, text_2d, vision_2d], 
                                                    ['Spatial Features', 'Text Features', 'Vision Features'])):
                ax = axes[i, 0]
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(data_2d[mask, 0], data_2d[mask, 1], color=colors(j), 
                              label=f'Class {label}', alpha=0.7)
                ax.set_title(title)
                if i == 0:  # Only show legend in the first subplot
                    ax.legend()
            
            # Second column: Comparisons between modalities (paired)
            pairs = [(spatial_2d, text_2d, 'Spatial vs Text'), 
                   (spatial_2d, vision_2d, 'Spatial vs Vision'), 
                   (text_2d, vision_2d, 'Text vs Vision')]
            
            for i, (data1, data2, title) in enumerate(pairs):
                ax = axes[i, 1]
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    # Draw points for the first modality
                    ax.scatter(data1[mask, 0], data1[mask, 1], color=colors(j), 
                              marker='o', label=f'Class {label} (Mod1)', alpha=0.7)
                    # Draw points for the second modality
                    ax.scatter(data2[mask, 0], data2[mask, 1], color=colors(j), 
                              marker='x', label=f'Class {label} (Mod2)', alpha=0.7)
                    
                    # Draw connecting lines (if number of samples is not too large)
                    if sum(mask) < 50:
                        for idx in np.where(mask)[0]:
                            ax.plot([data1[idx, 0], data2[idx, 0]], 
                                  [data1[idx, 1], data2[idx, 1]], 
                                  color=colors(j), alpha=0.3)
                
                ax.set_title(title)
        else:
            # If no labels, use fixed colors
            # First column: t-SNE for each modality separately
            for i, (data_2d, title, color) in enumerate(zip(
                [spatial_2d, text_2d, vision_2d], 
                ['Spatial Features', 'Text Features', 'Vision Features'],
                ['blue', 'green', 'red'])):
                
                ax = axes[i, 0]
                ax.scatter(data_2d[:, 0], data_2d[:, 1], color=color, alpha=0.7)
                ax.set_title(title)
            
            # Second column: Comparisons between modalities (paired)
            pairs = [(spatial_2d, text_2d, 'Spatial vs Text', 'blue', 'green'), 
                   (spatial_2d, vision_2d, 'Spatial vs Vision', 'blue', 'red'), 
                   (text_2d, vision_2d, 'Text vs Vision', 'green', 'red')]
            
            for i, (data1, data2, title, color1, color2) in enumerate(pairs):
                ax = axes[i, 1]
                # Draw points for the first modality
                ax.scatter(data1[:, 0], data1[:, 1], color=color1, 
                          marker='o', label=f'Modality 1', alpha=0.7)
                # Draw points for the second modality
                ax.scatter(data2[:, 0], data2[:, 1], color=color2, 
                          marker='x', label=f'Modality 2', alpha=0.7)
                
                # Draw connecting lines (if number of samples is not too large)
                if len(data1) < 50:
                    for idx in range(len(data1)):
                        ax.plot([data1[idx, 0], data2[idx, 0]], 
                              [data1[idx, 1], data2[idx, 1]], 
                              color='gray', alpha=0.3)
                
                ax.set_title(title)
                ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_integrated_analysis(self, predictions, attention_weights, uncertainty, 
                                    feature_names, class_names, output_path=None):
        """
        Comprehensive visualization of prediction results, attention weights, and uncertainty
        
        Args:
            predictions: Prediction results
            attention_weights: Attention weights
            uncertainty: Uncertainty
            feature_names: Feature names
            class_names: Class names
            output_path: Output file path
            
        Returns:
            results_dict: Dictionary containing results
        """
        # Ensure data is on CPU
        preds = predictions.cpu().detach().numpy() if torch.is_tensor(predictions) else predictions
        attn = attention_weights.cpu().detach().numpy() if torch.is_tensor(attention_weights) else attention_weights
        uncert = uncertainty.cpu().detach().numpy() if torch.is_tensor(uncertainty) else uncertainty
        
        plt.figure(figsize=(15, 12))
        
        # 1. Prediction result distribution
        plt.subplot(2, 2, 1)
        plt.bar(range(len(class_names)), preds, color='skyblue')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylabel('Prediction Score')
        plt.title('Prediction Distribution by Class')
        
        # 2. Prediction uncertainty
        plt.subplot(2, 2, 2)
        plt.bar(range(len(class_names)), uncert, color='salmon')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylabel('Uncertainty (Variance)')
        plt.title('Prediction Uncertainty by Class')
        
        # 3. Feature importance (based on attention weights)
        plt.subplot(2, 1, 2)
        # Sort for better visualization
        sorted_indices = np.argsort(attn)[-20:]  # Only show top 20 most important features
        sorted_weights = attn[sorted_indices]
        
        # Ensure feature names are consistent with weight length
        if len(feature_names) < len(attn):
            feature_names = list(feature_names) + [f"Feature_{i}" for i in range(len(feature_names), len(attn))]
        
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        plt.barh(range(len(sorted_indices)), sorted_weights, color='lightgreen')
        plt.yticks(range(len(sorted_indices)), sorted_names)
        plt.xlabel('Attention Weight')
        plt.title('Top 20 Features by Importance')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        # Return integrated results
        results_dict = {
            'predictions': preds,
            'attention_weights': attn,
            'uncertainty': uncert,
            'top_features': sorted_indices
        }
        
        return results_dict
    
    def visualize_multimodal_predictions(self, spatial_pred, text_pred, vision_pred, combined_pred,
                                       true_labels=None, class_names=None, output_path=None):
        """
        Compare performance of different modalities and integrated predictions
        
        Args:
            spatial_pred: Spatial modality predictions
            text_pred: Text modality predictions
            vision_pred: Vision modality predictions
            combined_pred: Integrated predictions
            true_labels: True labels
            class_names: Class names
            output_path: Output file path
        """
        # Ensure data is on CPU
        spatial_p = spatial_pred.cpu().detach().numpy() if torch.is_tensor(spatial_pred) else spatial_pred
        text_p = text_pred.cpu().detach().numpy() if torch.is_tensor(text_pred) else text_pred
        vision_p = vision_pred.cpu().detach().numpy() if torch.is_tensor(vision_pred) else vision_pred
        combined_p = combined_pred.cpu().detach().numpy() if torch.is_tensor(combined_pred) else combined_pred
        
        if true_labels is not None and torch.is_tensor(true_labels):
            true_labels = true_labels.cpu().numpy()
        
        # If probability distribution, convert to class
        if len(spatial_p.shape) > 1:
            spatial_class = np.argmax(spatial_p, axis=1)
            text_class = np.argmax(text_p, axis=1)
            vision_class = np.argmax(vision_p, axis=1)
            combined_class = np.argmax(combined_p, axis=1)
        else:
            spatial_class = spatial_p
            text_class = text_p
            vision_class = vision_p
            combined_class = combined_p
        
        # Calculate accuracy for each modality
        if true_labels is not None:
            spatial_acc = np.mean(spatial_class == true_labels)
            text_acc = np.mean(text_class == true_labels)
            vision_acc = np.mean(vision_class == true_labels)
            combined_acc = np.mean(combined_class == true_labels)
            
            # Create accuracy bar chart
            plt.figure(figsize=(10, 6))
            accuracies = [spatial_acc, text_acc, vision_acc, combined_acc]
            modalities = ['Spatial', 'Text', 'Vision', 'Combined']
            
            plt.bar(modalities, accuracies, color=['blue', 'green', 'red', 'purple'])
            plt.ylim(0, 1.1)
            plt.ylabel('Accuracy')
            plt.title('Prediction Accuracy by Modality')
            
            # Display accuracy values above the bars
            for i, acc in enumerate(accuracies):
                plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()
            
            return accuracies
        
        # If no true labels, draw confusion matrix
        else:
            plt.figure(figsize=(15, 10))
            
            # Prepare class names
            if class_names is None:
                n_classes = max(
                    np.max(spatial_class), 
                    np.max(text_class), 
                    np.max(vision_class), 
                    np.max(combined_class)
                ) + 1
                class_names = [f'Class {i}' for i in range(n_classes)]
            
            # 1. Spatial vs Text
            plt.subplot(2, 3, 1)
            matrix = np.zeros((len(class_names), len(class_names)))
            for s, t in zip(spatial_class, text_class):
                matrix[s, t] += 1
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Spatial vs Text Predictions')
            plt.xlabel('Text Predictions')
            plt.ylabel('Spatial Predictions')
            
            # 2. Spatial vs Vision
            plt.subplot(2, 3, 2)
            matrix = np.zeros((len(class_names), len(class_names)))
            for s, v in zip(spatial_class, vision_class):
                matrix[s, v] += 1
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Spatial vs Vision Predictions')
            plt.xlabel('Vision Predictions')
            plt.ylabel('Spatial Predictions')
            
            # 3. Text vs Vision
            plt.subplot(2, 3, 3)
            matrix = np.zeros((len(class_names), len(class_names)))
            for t, v in zip(text_class, vision_class):
                matrix[t, v] += 1
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Text vs Vision Predictions')
            plt.xlabel('Vision Predictions')
            plt.ylabel('Text Predictions')
            
            # 4. Spatial vs Combined
            plt.subplot(2, 3, 4)
            matrix = np.zeros((len(class_names), len(class_names)))
            for s, c in zip(spatial_class, combined_class):
                matrix[s, c] += 1
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Spatial vs Combined Predictions')
            plt.xlabel('Combined Predictions')
            plt.ylabel('Spatial Predictions')
            
            # 5. Text vs Combined
            plt.subplot(2, 3, 5)
            matrix = np.zeros((len(class_names), len(class_names)))
            for t, c in zip(text_class, combined_class):
                matrix[t, c] += 1
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Text vs Combined Predictions')
            plt.xlabel('Combined Predictions')
            plt.ylabel('Text Predictions')
            
            # 6. Vision vs Combined
            plt.subplot(2, 3, 6)
            matrix = np.zeros((len(class_names), len(class_names)))
            for v, c in zip(vision_class, combined_class):
                matrix[v, c] += 1
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Vision vs Combined Predictions')
            plt.xlabel('Combined Predictions')
            plt.ylabel('Vision Predictions')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()