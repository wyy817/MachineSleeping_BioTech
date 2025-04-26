"""
Evaluation Metrics Module: Calculate various evaluation metrics
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import classification_report, accuracy_score, f1_score


def calculate_auc(y_true, y_pred):
    """
    Calculate AUC (Area Under ROC Curve)
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        auc: AUC value
    """
    try:
        # Binary classification
        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_pred)
        # Multi-class classification
        else:
            # Ensure y_pred contains probabilities for each class (N, num_classes)
            if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                raise ValueError("Multi-class problems require probabilities for each class")
            
            # Convert true labels to one-hot encoding
            y_true_onehot = np.zeros((len(y_true), y_pred.shape[1]))
            for i, label in enumerate(y_true):
                y_true_onehot[i, label] = 1
            
            # Calculate multi-class AUC (OVR)
            return roc_auc_score(y_true_onehot, y_pred, multi_class='ovr')
    except:
        return np.nan


def calculate_aupr(y_true, y_pred):
    """
    Calculate AUPR (Area Under Precision-Recall Curve)
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        aupr: AUPR value
    """
    try:
        # Binary classification
        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            return auc(recall, precision)
        # Multi-class classification
        else:
            # Ensure y_pred contains probabilities for each class (N, num_classes)
            if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                raise ValueError("Multi-class problems require probabilities for each class")
            
            # Convert true labels to one-hot encoding
            y_true_onehot = np.zeros((len(y_true), y_pred.shape[1]))
            for i, label in enumerate(y_true):
                y_true_onehot[i, label] = 1
            
            # Calculate AUPR for each class
            n_classes = y_pred.shape[1]
            aupr_list = []
            
            for i in range(n_classes):
                if np.sum(y_true_onehot[:, i]) > 0:  # Ensure there are positive samples
                    precision, recall, _ = precision_recall_curve(
                        y_true_onehot[:, i], y_pred[:, i])
                    aupr_list.append(auc(recall, precision))
            
            # Return average AUPR
            return np.mean(aupr_list)
    except:
        return np.nan


def calculate_ci_index(predicted_risk, survival_time, event_indicator):
    """
    Calculate Concordance Index (CI), used for survival analysis
    
    Args:
        predicted_risk: Predicted risk scores
        survival_time: Survival times
        event_indicator: Event indicators (1 means event occurred, 0 means censored)
        
    Returns:
        ci: Concordance Index
    """
    try:
        from lifelines.utils import concordance_index
        return concordance_index(survival_time, -predicted_risk, event_indicator)
    except:
        return np.nan


def calculate_purity(labels, pred_clusters):
    """
    Calculate clustering purity
    
    Args:
        labels: True labels
        pred_clusters: Predicted cluster labels
        
    Returns:
        purity: Clustering purity
    """
    # Create contingency matrix
    contingency = np.zeros((np.max(pred_clusters) + 1, np.max(labels) + 1))
    
    for i in range(len(labels)):
        contingency[pred_clusters[i], labels[i]] += 1
    
    # Calculate purity
    purity = np.sum(np.max(contingency, axis=1)) / len(labels)
    
    return purity


def calculate_silhouette(features, pred_clusters):
    """
    Calculate silhouette coefficient, used for evaluating clustering quality
    
    Args:
        features: Feature matrix
        pred_clusters: Predicted cluster labels
        
    Returns:
        silhouette: Silhouette coefficient
    """
    try:
        from sklearn.metrics import silhouette_score
        return silhouette_score(features, pred_clusters)
    except:
        return np.nan


def calculate_spatial_correlation(values, coordinates):
    """
    Calculate spatial autocorrelation coefficient (Moran's I)
    
    Args:
        values: Measured values
        coordinates: Spatial coordinates
        
    Returns:
        moran_i: Moran's I coefficient
    """
    try:
        from pysal.explore import esda
        from pysal.lib import weights
        
        # Create spatial weight matrix
        knn = weights.KNN.from_array(coordinates, k=5)
        
        # Calculate Moran's I
        moran = esda.Moran(values, knn)
        
        return moran.I
    except:
        return np.nan


def calculate_spatial_autocorrelation(values, coordinates, max_distance=100.0):
    """
    Calculate spatial autocorrelation function (similar to variogram)
    
    Args:
        values: Measured values
        coordinates: Spatial coordinates
        max_distance: Maximum distance
        
    Returns:
        distances: Distance intervals
        correlations: Corresponding correlation coefficients
    """
    # Calculate distances between all point pairs
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Define distance intervals
    n_bins = 10
    distance_bins = np.linspace(0, max_distance, n_bins + 1)
    
    # Calculate autocorrelation coefficient for each distance interval
    correlations = []
    distances = []
    
    for i in range(n_bins):
        lower = distance_bins[i]
        upper = distance_bins[i+1]
        
        # Find all point pairs within this distance interval
        mask = (dist_matrix > lower) & (dist_matrix <= upper)
        
        if np.sum(mask) > 0:
            # Calculate correlation coefficient for values of pairs in this interval
            value_pairs = []
            
            for j in range(n):
                for k in range(j+1, n):
                    if mask[j, k]:
                        value_pairs.append((values[j], values[k]))
            
            if len(value_pairs) > 1:
                value_pairs = np.array(value_pairs)
                correlation = np.corrcoef(value_pairs[:, 0], value_pairs[:, 1])[0, 1]
                
                distances.append((lower + upper) / 2)
                correlations.append(correlation)
    
    return np.array(distances), np.array(correlations)


def calculate_domain_discrepancy(source_features, target_features, method='mmd', **kwargs):
    """
    Calculate domain discrepancy measure
    
    Args:
        source_features: Source domain features
        target_features: Target domain features
        method: Method, 'mmd' or 'coral'
        
    Returns:
        discrepancy: Domain discrepancy measure
    """
    if method == 'mmd':
        kernel_type = kwargs.get('kernel_type', 'rbf')
        
        if kernel_type == 'linear':
            # Linear MMD
            source_mean = np.mean(source_features, axis=0)
            target_mean = np.mean(target_features, axis=0)
            return np.sum((source_mean - target_mean) ** 2)
        
        elif kernel_type == 'rbf':
            # RBF kernel MMD
            from sklearn.metrics.pairwise import rbf_kernel
            
            # Choose appropriate gamma
            gamma = kwargs.get('gamma', 1.0)
            
            # Calculate kernel matrices
            XX = rbf_kernel(source_features, source_features, gamma=gamma)
            YY = rbf_kernel(target_features, target_features, gamma=gamma)
            XY = rbf_kernel(source_features, target_features, gamma=gamma)
            
            # Calculate MMD
            mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
            
            return mmd
    
    elif method == 'coral':
        # CORAL: CORrelation ALignment
        # Calculate covariance matrices
        source_cov = np.cov(source_features, rowvar=False)
        target_cov = np.cov(target_features, rowvar=False)
        
        # Calculate Frobenius norm
        coral_dist = np.sum((source_cov - target_cov) ** 2)
        
        return coral_dist
    
    else:
        raise ValueError(f"Unsupported method: {method}")


def calculate_multimodal_metrics(predictions, true_labels, mod_contributions=None):
    """
    Calculate comprehensive evaluation metrics for multimodal models
    
    Args:
        predictions: Predicted values, shape=(n_samples, n_classes)
        true_labels: True labels, shape=(n_samples,)
        mod_contributions: Modality contribution dictionary, {modality_name: contribution_value}
        
    Returns:
        metrics: Evaluation metrics dictionary
    """
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, np.argmax(predictions, axis=1)),
        'f1_score': f1_score(true_labels, np.argmax(predictions, axis=1), average='weighted'),
        'auc': calculate_auc(true_labels, predictions),
        'aupr': calculate_aupr(true_labels, predictions)
    }
    
    # Detailed classification report
    metrics['classification_report'] = classification_report(
        true_labels, np.argmax(predictions, axis=1), output_dict=True
    )
    
    # Modality contributions
    if mod_contributions is not None:
        metrics['modality_contributions'] = mod_contributions
        
        # Calculate modality synergistic gain
        # Assuming maximum single modality contribution
        if len(mod_contributions) > 1:
            max_single_mod = max(mod_contributions.values())
            total_perf = metrics['accuracy']  # Using accuracy to measure performance
            
            # Synergistic gain = total performance - maximum single modality performance
            metrics['synergistic_gain'] = total_perf - max_single_mod
    
    return metrics