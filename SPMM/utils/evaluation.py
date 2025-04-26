"""
Evaluation Module: Evaluate Model Performance
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def evaluate_model(predictions, true_labels, class_names=None):
    """
    Evaluate Model Performance
    
    Args:
        predictions: Prediction results, shape (n_samples, n_classes) or (n_samples,)
        true_labels: True labels, shape (n_samples,)
        class_names: List of class names
        
    Returns:
        results: Dictionary containing various evaluation metrics
    """
    # If predictions are probability distributions, convert to class labels
    if len(predictions.shape) > 1:
        pred_probs = predictions
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions
        # Construct one-hot encoded probability distribution
        n_classes = len(np.unique(true_labels))
        pred_probs = np.zeros((len(predictions), n_classes))
        pred_probs[np.arange(len(predictions)), pred_labels] = 1
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Calculate precision, recall and F1 score
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Generate classification report
    if class_names:
        report = classification_report(true_labels, pred_labels, target_names=class_names)
    else:
        report = classification_report(true_labels, pred_labels)
    
    # Calculate ROC AUC (only applicable for binary or multiclass)
    try:
        if len(np.unique(true_labels)) == 2:  # Binary classification
            roc_auc = roc_auc_score(true_labels, pred_probs[:, 1])
        else:  # Multi-class
            roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    except:
        roc_auc = None
    
    # Integrate results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc
    }
    
    return results


def evaluate_survival_prediction(predicted_survival, true_survival, censoring=None):
    """
    Evaluate survival prediction performance
    
    Args:
        predicted_survival: Predicted survival time or risk score
        true_survival: True survival time
        censoring: Censoring indicator (1 means event occurred, 0 means censored)
        
    Returns:
        results: Dictionary containing various evaluation metrics
    """
    from lifelines.utils import concordance_index
    
    # Calculate concordance index (C-index)
    if censoring is not None:
        c_index = concordance_index(true_survival, -predicted_survival, censoring)
    else:
        # If no censoring information, assume all samples observed the event
        c_index = concordance_index(true_survival, -predicted_survival, np.ones_like(true_survival))
    
    # Calculate mean squared error (MSE)
    mse = np.mean((predicted_survival - true_survival) ** 2)
    
    # Calculate mean absolute error (MAE)
    mae = np.mean(np.abs(predicted_survival - true_survival))
    
    # Integrate results
    results = {
        'c_index': c_index,
        'mse': mse,
        'mae': mae
    }
    
    return results


def evaluate_spatial_prediction(predicted_values, true_values, spatial_coords):
    """
    Evaluate Spatial Prediction Performance
    
    Args:
        predicted_values: Predicted values
        true_values: True values
        spatial_coords: Spatial coordinates
        
    Returns:
        results: Dictionary containing various evaluation metrics
    """
    # Calculate global mean squared error (MSE)
    global_mse = np.mean((predicted_values - true_values) ** 2)
    
    # Calculate global mean absolute error (MAE)
    global_mae = np.mean(np.abs(predicted_values - true_values))
    
    # Calculate spatial autocorrelation (Moran's I)
    # This requires more complex spatial statistics libraries, simplified here
    spatial_correlation = np.nan
    
    try:
        from pysal.explore import esda
        from pysal.lib import weights
        
        # Create spatial weight matrix
        knn = weights.KNN.from_array(spatial_coords, k=5)
        
        # Calculate Moran's I
        moran = esda.Moran(true_values, knn)
        spatial_correlation_true = moran.I
        
        moran = esda.Moran(predicted_values, knn)
        spatial_correlation_pred = moran.I
        
        # Spatial autocorrelation difference between true and predicted values
        spatial_correlation_diff = abs(spatial_correlation_true - spatial_correlation_pred)
        
        spatial_correlation = spatial_correlation_pred
    except:
        pass
    
    # Calculate spatial error distribution
    # Uniformity of error distribution across space
    error = predicted_values - true_values
    
    # Divide spatial coordinates into grid
    try:
        from sklearn.cluster import KMeans
        
        # Assume spatial coordinates are two-dimensional
        n_regions = min(10, len(spatial_coords))
        kmeans = KMeans(n_clusters=n_regions, random_state=0).fit(spatial_coords)
        regions = kmeans.labels_
        
        # Calculate average error for each region
        region_errors = [np.mean(error[regions == i]**2) for i in range(n_regions)]
        
        # Inter-region error variance (lower indicates more uniform error distribution)
        spatial_error_variance = np.var(region_errors)
    except:
        spatial_error_variance = np.nan
    
    # Integrate results
    results = {
        'global_mse': global_mse,
        'global_mae': global_mae,
        'spatial_correlation': spatial_correlation,
        'spatial_error_variance': spatial_error_variance
    }
    
    return results


def evaluate_multimodal_contribution(model, data_batch, modalities=['spatial', 'text', 'vision']):
    """
    Evaluate the contribution of different modalities to model predictions
    
    Args:
        model: Model instance
        data_batch: Data batch
        modalities: List of modalities
        
    Returns:
        contributions: Dictionary of modality contributions
    """
    import torch
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Get complete predictions
    with torch.no_grad():
        full_outputs = model(
            data_batch['spatial_expr'].to(device),
            data_batch['edge_index'].to(device) if 'edge_index' in data_batch else None,
            data_batch['spatial_coords'].to(device),
            data_batch['text_input_ids'].to(device),
            data_batch['text_attention_mask'].to(device),
            data_batch['image_features'].to(device)
        )
    
    full_prediction = full_outputs['classification']
    
    # Perform ablation experiment for each modality
    contributions = {}
    
    for modality in modalities:
        # Create modality ablation data
        ablated_batch = data_batch.copy()
        
        if modality == 'spatial':
            # Replace spatial features with zeros
            ablated_batch['spatial_expr'] = torch.zeros_like(data_batch['spatial_expr'])
        elif modality == 'text':
            # Replace text features with zeros
            ablated_batch['text_input_ids'] = torch.zeros_like(data_batch['text_input_ids'])
            ablated_batch['text_attention_mask'] = torch.zeros_like(data_batch['text_attention_mask'])
        elif modality == 'vision':
            # Replace vision features with zeros
            ablated_batch['image_features'] = torch.zeros_like(data_batch['image_features'])
        
        # Get predictions after ablation
        with torch.no_grad():
            ablated_outputs = model(
                ablated_batch['spatial_expr'].to(device),
                ablated_batch['edge_index'].to(device) if 'edge_index' in ablated_batch else None,
                ablated_batch['spatial_coords'].to(device),
                ablated_batch['text_input_ids'].to(device),
                ablated_batch['text_attention_mask'].to(device),
                ablated_batch['image_features'].to(device)
            )
        
        ablated_prediction = ablated_outputs['classification']
        
        # Calculate ablation impact (prediction change)
        prediction_change = torch.norm(full_prediction - ablated_prediction, dim=1)
        contribution = prediction_change.mean().item()
        
        contributions[modality] = contribution
    
    # Normalize contributions
    total_contribution = sum(contributions.values())
    if total_contribution > 0:
        for modality in contributions:
            contributions[modality] /= total_contribution
    
    return contributions