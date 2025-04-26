"""
Loss Function Module: Defines various loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedMultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss function, automatically learns task weights
    
    Based on paper:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    def __init__(self, num_tasks=2):
        super(WeightedMultiTaskLoss, self).__init__()
        # Initialize task uncertainty parameters (log sigma^2)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        """
        Calculate weighted multi-task loss
        
        Args:
            losses: List of losses for each task
            
        Returns:
            loss: Total loss
            weights: Weights for each task
        """
        # Calculate task weights 1/(2*sigma^2)
        weights = torch.exp(-self.log_vars)
        
        # Calculate weighted loss weight*loss + log(sigma^2)
        weighted_losses = []
        for i, loss in enumerate(losses):
            weighted_losses.append(weights[i] * loss + 0.5 * self.log_vars[i])
        
        # Total loss
        loss = sum(weighted_losses)
        
        return loss, weights


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function
    
     Based on paper:
    "A Simple Framework for Contrastive Learning of Visual Representations"
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features_1, features_2, batch_size=None):
        """
        Calculate contrastive loss
        
        Args:
            features_1: First set of features
            features_2: Second set of features
            batch_size: Batch size (if None, use first dimension of features_1)
            
        Returns:
            loss: Contrastive loss
        """
        if batch_size is None:
            batch_size = features_1.shape[0]
        
        # Feature normalization
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        
        # Concatenate features from different views
        features = torch.cat([features_1, features_2], dim=0)
        
        # Calculate similarity matrix
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature
        
        # Create labels: indices of positive pairs
        labels = torch.arange(batch_size, device=features_1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, device=features_1.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Calculate InfoNCE loss
        positives = torch.cat([
            similarity_matrix[labels[i]][labels[i+batch_size] - batch_size].unsqueeze(0)
            for i in range(batch_size)
        ] + [
            similarity_matrix[labels[i+batch_size]][labels[i]].unsqueeze(0)
            for i in range(batch_size)
        ], dim=0)
        
        # Compare positive sample similarity with all similarities
        loss = -torch.mean(positives - torch.logsumexp(similarity_matrix, dim=1))
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Based on paper:
    "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calculate Focal Loss
        
        Args:
            inputs: Predictions, shape=(N, C)
            targets: Target values, shape=(N,)
            
        Returns:
            loss: Focal Loss
        """
        # Convert to probability distribution
        inputs = F.softmax(inputs, dim=1)
        
        # Get probability for each sample's corresponding class
        batch_size = inputs.size(0)
        probs = inputs[torch.arange(batch_size), targets]
        
        # Focal Loss formula: -(1-p)^gamma * log(p)
        focal_weight = torch.pow(1 - probs, self.gamma)
        loss = -focal_weight * torch.log(probs + 1e-8)  # Add small value to avoid log(0)
        
        # If class weights are provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha[targets]
            else:
                alpha = torch.tensor([self.alpha] * batch_size, device=inputs.device)
            loss = alpha * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SpatialRegularizationLoss(nn.Module):
    """
    Spatial regularization loss, encourages spatially close samples to have similar features
    """
    def __init__(self, weight=1.0, sigma=1.0):
        super(SpatialRegularizationLoss, self).__init__()
        self.weight = weight
        self.sigma = sigma
    
    def forward(self, features, coords):
        """
        Calculate spatial regularization loss
        
        Args:
            features: Feature matrix, shape=(N, D)
            coords: Spatial coordinates, shape=(N, 2)
            
        Returns:
            loss: Spatial regularization loss
        """
        # Calculate spatial distance matrix
        n = coords.size(0)
        x = coords.unsqueeze(1)  # (N, 1, 2)
        y = coords.unsqueeze(0)  # (1, N, 2)
        spatial_dist = torch.sqrt(torch.sum((x - y) ** 2, dim=2) + 1e-8)  # (N, N)
        
        # Calculate feature distance matrix
        features_norm = F.normalize(features, dim=1)
        feature_dist = 1 - torch.mm(features_norm, features_norm.t())  # (N, N)
        
        # Weights based on spatial distance (Gaussian kernel)
        weight_matrix = torch.exp(-spatial_dist ** 2 / (2 * self.sigma ** 2))
        
        # Mask diagonal (self)
        mask = torch.eye(n, device=features.device, dtype=torch.bool)
        weight_matrix = weight_matrix[~mask].view(n, n-1)
        feature_dist = feature_dist[~mask].view(n, n-1)
        
        # Weighted feature distance
        loss = torch.mean(weight_matrix * feature_dist)
        
        return self.weight * loss


class GraphSmoothingLoss(nn.Module):
    """
    Graph smoothing loss, encourages connected nodes in a graph to have similar features
    """
    def __init__(self, weight=1.0):
        super(GraphSmoothingLoss, self).__init__()
        self.weight = weight
    
    def forward(self, features, edge_index):
        """
        Calculate graph smoothing loss
        
        Args:
            features: Node features, shape=(N, D)
            edge_index: Edge indices, shape=(2, E)
            
        Returns:
            loss: Graph smoothing loss
        """
        # Extract source and target nodes
        src = edge_index[0]
        dst = edge_index[1]
        
        # Calculate distance between source and target node features
        src_features = features[src]
        dst_features = features[dst]
        
        # Use L2 distance
        feature_dist = torch.norm(src_features - dst_features, dim=1)
        
        # Take mean
        loss = torch.mean(feature_dist)
        
        return self.weight * loss


class DomainAdversarialLoss(nn.Module):
    """
    Domain adversarial loss, used for domain adaptation/generalization
    
    Based on paper:
    "Domain-Adversarial Training of Neural Networks"
    """
    def __init__(self, domain_classifier):
        super(DomainAdversarialLoss, self).__init__()
        self.domain_classifier = domain_classifier
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, features, domain_labels, lambda_param=1.0):
        """
        Calculate domain adversarial loss
        
        Args:
            features: Features, shape=(N, D)
            domain_labels: Domain labels, shape=(N,)
            lambda_param: Gradient reversal parameter
            
        Returns:
            loss: Domain adversarial loss
        """
        # Gradient reversal layer
        reversed_features = GradientReversal.apply(features, lambda_param)
        
        # Forward pass through domain classifier
        domain_preds = self.domain_classifier(reversed_features)
        
        # Calculate domain classification loss
        loss = self.criterion(domain_preds, domain_labels.float())
        
        return loss


class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer
    
    Used for domain adversarial training, forward pass remains unchanged, 
    backward pass reverses gradients
    """
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None


class CoxPHLoss(nn.Module):
    """
    Calculate Cox proportional hazards model loss
    """
    def __init__(self):
        super(CoxPHLoss, self).__init__()
    
    def forward(self, risk_scores, survival_time, event_indicator):
        """
        Calculate Cox proportional hazards model loss
        
        Args:
            risk_scores: Risk scores, shape=(N,)
            survival_time: Survival time, shape=(N,)
            event_indicator: Event indicator (1 for event occurrence, 0 for censoring), shape=(N,)
            
        Returns:
            loss: Cox proportional hazards model loss
        """
        # Sort by survival time
        idx = torch.argsort(survival_time, descending=True)
        risk_scores = risk_scores[idx]
        event_indicator = event_indicator[idx]
        
        # Calculate risk set
        n_samples = risk_scores.size(0)
        log_risk = torch.log(torch.cumsum(torch.exp(risk_scores), dim=0))
        
        # Calculate negative log partial likelihood
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * event_indicator
        
        # Calculate average negative log likelihood
        loss = -torch.sum(censored_likelihood) / torch.sum(event_indicator)
        
        return loss


class HierarchicalClassificationLoss(nn.Module):
    """
    Hierarchical classification loss, considers hierarchical relationships between classes
    """
    def __init__(self, hierarchy_matrix, alpha=1.0):
        """
        Initialize hierarchical classification loss
        
        Args:
            hierarchy_matrix: Class hierarchy relationship matrix, shape=(n_classes, n_classes)
                              hierarchy_matrix[i, j] = 1 indicates class i is a parent of class j
            alpha: Hierarchical loss weight
        """
        super(HierarchicalClassificationLoss, self).__init__()
        self.hierarchy_matrix = hierarchy_matrix
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Calculate hierarchical classification loss
        
        Args:
            logits: Predicted log probabilities, shape=(N, n_classes)
            targets: Target classes, shape=(N,)
            
        Returns:
            loss: Hierarchical classification loss
        """
        # Standard cross entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Calculate predicted probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate hierarchical loss
        batch_size = probs.size(0)
        device = probs.device
        hierarchy_matrix = self.hierarchy_matrix.to(device)
        
        # Calculate hierarchical loss for each sample
        h_loss = 0
        for i in range(batch_size):
            target = targets[i]
            
            # Get parent classes of the target class
            parents = torch.nonzero(hierarchy_matrix[:, target]).squeeze(-1)
            
            if parents.dim() > 0 and parents.size(0) > 0:
                # Calculate sum of parent class predicted probabilities
                parent_probs = probs[i, parents].sum()
                
                # Target class probability should be less than or equal to the sum of parent probabilities
                h_loss += F.relu(probs[i, target] - parent_probs)
            
            # Get child classes of the target class
            children = torch.nonzero(hierarchy_matrix[target, :]).squeeze(-1)
            
            if children.dim() > 0 and children.size(0) > 0:
                # Calculate sum of child class predicted probabilities
                children_probs = probs[i, children].sum()
                
                # Target class probability should be greater than or equal to the sum of child probabilities
                h_loss += F.relu(children_probs - probs[i, target])
        
        # Normalize hierarchical loss
        h_loss = h_loss / batch_size
        
        # Combined loss
        loss = ce_loss + self.alpha * h_loss
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss, used for metric learning
    
    Based on paper:
    "FaceNet: A Unified Embedding for Face Recognition and Clustering"
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss
        
        Args:
            anchor: Anchor features, shape=(N, D)
            positive: Positive sample features, shape=(N, D)
            negative: Negative sample features, shape=(N, D)
            
        Returns:
            loss: Triplet loss
        """
        # Calculate distance between positive pairs
        pos_dist = torch.norm(anchor - positive, dim=1)
        
        # Calculate distance between negative pairs
        neg_dist = torch.norm(anchor - negative, dim=1)
        
        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss, used for domain adaptation均值差异损失，用于域适应
    
    Based on paper:
    "Learning Transferable Features with Deep Adaptation Networks"
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None):
        """
        Calculate Gaussian kernel
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        """
        Calculate MMD loss
        
        Args:
            source: Source domain features, shape=(N_s, D)
            target: Target domain features, shape=(N_t, D)
            
        Returns:
            loss: MMD loss
        """
        batch_size = source.size(0)
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num
        )
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        
        return loss