# loss_functions_fixed.py - Comprehensive loss functions with numerical stability

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings


class FocalLoss(nn.Module):
    """
    Focal Loss with numerically stable implementation
    
    Args:
        gamma: Focusing parameter for down-weighting easy samples
        alpha: Optional class-weights for imbalanced datasets
        reduction: Reduction method ('mean', 'sum', 'none')
        eps: Small constant for numerical stability
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Forward pass with error handling
        
        Args:
            inputs: Predicted logits (B, C) or (B, T, C)
            targets: Target classes (B) or (B, T)
            
        Returns:
            loss: Focal loss value
        """
        try:
            # Handle different input shapes
            if inputs.dim() == 3:
                # Reshape for sequence data
                B, T, C = inputs.shape
                inputs = inputs.reshape(-1, C)
                targets = targets.reshape(-1)
            
            # Get class probabilities
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)
            
            # Extract probs for target class
            if targets.dim() == 1:
                # One-hot encode if needed
                target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            else:
                # Already one-hot encoded
                target_probs = (probs * targets).sum(dim=1)
            
            # Ensure numerical stability
            target_probs = torch.clamp(target_probs, min=self.eps, max=1.0-self.eps)
            
            # Focal loss formula
            focal_weight = (1 - target_probs) ** self.gamma
            focal_loss = -focal_weight * torch.log(target_probs)
            
            # Apply alpha weighting if provided
            if self.alpha is not None:
                alpha = self.alpha.to(inputs.device)
                if targets.dim() == 1:
                    alpha_weight = alpha.gather(0, targets)
                else:
                    alpha_weight = (alpha.unsqueeze(0) * targets).sum(1)
                
                focal_loss = alpha_weight * focal_loss
            
            # Apply reduction
            if self.reduction == 'none':
                return focal_loss
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:  # mean
                return focal_loss.mean()
                
        except Exception as e:
            warnings.warn(f"Error in focal loss: {e}. Using cross entropy instead.")
            # Fallback to cross entropy
            return F.cross_entropy(inputs, targets)


class TemporalConsistencyLoss(nn.Module):
    """
    Loss function that penalizes unrealistic temporal fluctuations in sentiment
    
    Args:
        lambda_smooth: Smoothness coefficient
        reduction: Reduction method
    """
    def __init__(self, lambda_smooth=0.01, reduction='mean'):
        super(TemporalConsistencyLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.reduction = reduction

    def forward(self, predictions):
        """
        Calculate temporal consistency loss
        
        Args:
            predictions: Predicted sentiments (B, T)
            
        Returns:
            loss: Temporal consistency loss
        """
        try:
            # Ensure predictions has the right dimensions
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)
                
            batch_size, seq_len = predictions.shape
            
            if seq_len < 3:
                # Not enough timesteps for second derivatives
                return torch.tensor(0.0, device=predictions.device)
            
            # Calculate second derivatives (∂²y/∂t²)
            first_diff = predictions[:, 1:] - predictions[:, :-1]
            second_diff = first_diff[:, 1:] - first_diff[:, :-1]
            
            # Square the second derivatives
            squared_diff = second_diff ** 2
            
            # Apply scaling factor
            loss = self.lambda_smooth * squared_diff
            
            # Apply reduction
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # mean
                return loss.mean()
                
        except Exception as e:
            warnings.warn(f"Error in temporal consistency loss: {e}")
            return torch.tensor(0.0, device=predictions.device if hasattr(predictions, 'device') else 'cpu')


class TrendDirectionLoss(nn.Module):
    """
    Loss function that penalizes incorrect prediction of trend directions
    
    Args:
        margin: Margin for trend direction matching
        reduction: Reduction method
    """
    def __init__(self, margin=0.1, reduction='mean'):
        super(TrendDirectionLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Calculate trend direction loss
        
        Args:
            predictions: Predicted sentiments (B, T)
            targets: Target sentiments (B, T)
            
        Returns:
            loss: Trend direction loss
        """
        try:
            # Ensure predictions has the right dimensions
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
                
            batch_size, seq_len = predictions.shape
            
            if seq_len < 2:
                # Not enough timesteps for trend analysis
                return torch.tensor(0.0, device=predictions.device)
            
            # Calculate deltas for predictions and targets
            pred_deltas = predictions[:, 1:] - predictions[:, :-1]
            target_deltas = targets[:, 1:] - targets[:, :-1]
            
            # Get signs of deltas
            pred_signs = torch.sign(pred_deltas)
            target_signs = torch.sign(target_deltas)
            
            # Calculate whether direction matches
            direction_match = pred_signs * target_signs
            
            # Hinge loss formulation: penalize when directions don't match
            trend_loss = torch.max(
                torch.zeros_like(direction_match),
                self.margin - direction_match
            )
            
            # Apply reduction
            if self.reduction == 'none':
                return trend_loss
            elif self.reduction == 'sum':
                return trend_loss.sum()
            else:  # mean
                return trend_loss.mean()
                
        except Exception as e:
            warnings.warn(f"Error in trend direction loss: {e}")
            return torch.tensor(0.0, device=predictions.device if hasattr(predictions, 'device') else 'cpu')


class SentimentLoss(nn.Module):
    """
    Combined loss function for sentiment forecasting
    
    Args:
        alpha: Weight for classification loss
        beta: Weight for regression loss
        gamma: Weight for temporal consistency loss
        delta: Weight for trend direction loss
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super(SentimentLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        self.focal_loss = FocalLoss(gamma=2.0)
        self.huber_loss = nn.SmoothL1Loss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.trend_loss = TrendDirectionLoss()

    def forward(self, predictions, targets):
        """
        Calculate total loss with error handling
        
        Args:
            predictions: Model predictions (B, T, C) or (B, C)
            targets: Target values (B, T) or (B)
            
        Returns:
            loss: Total loss
            components: Dict of loss components
        """
        try:
            # Ensure targets are long for classification
            if targets.dtype != torch.long:
                targets_cls = targets.long()
            else:
                targets_cls = targets
            
            # Compute classification loss
            cls_loss = self.focal_loss(predictions, targets_cls)
            
            # Convert predictions to sentiment scores
            if predictions.dim() > 2:
                # (B, T, C) -> (B*T, C)
                pred_flat = predictions.reshape(-1, predictions.size(-1))
                target_flat = targets.reshape(-1)
                
                # Get sentiment scores (0-1)
                pred_probs = F.softmax(pred_flat, dim=-1)
                weights = torch.linspace(0, 1, pred_probs.size(-1), device=pred_probs.device)
                pred_scores = (pred_probs * weights).sum(dim=-1)
                
                # Reshape back
                pred_scores = pred_scores.reshape(predictions.size(0), -1)
                target_scores = targets.float() / (predictions.size(-1) - 1)
                
                # Compute regression loss
                reg_loss = self.huber_loss(pred_scores, target_scores)
                
                # Compute temporal loss components if possible
                temp_loss = self.temporal_loss(pred_scores)
                trend_loss = self.trend_loss(pred_scores, target_scores)
            else:
                # (B, C) case - no temporal losses
                pred_probs = F.softmax(predictions, dim=-1)
                weights = torch.linspace(0, 1, pred_probs.size(-1), device=pred_probs.device)
                pred_scores = (pred_probs * weights).sum(dim=-1)
                target_scores = targets.float() / (predictions.size(-1) - 1)
                
                reg_loss = self.huber_loss(pred_scores, target_scores)
                temp_loss = torch.tensor(0.0, device=predictions.device)
                trend_loss = torch.tensor(0.0, device=predictions.device)
            
            # Total loss
            total_loss = (
                self.alpha * cls_loss +
                self.beta * reg_loss +
                self.gamma * temp_loss +
                self.delta * trend_loss
            )
            
            # Return loss and components
            return total_loss, {
                'classification': cls_loss.item(),
                'regression': reg_loss.item(),
                'temporal': temp_loss.item(),
                'trend': trend_loss.item(),
                'total': total_loss.item()
            }
            
        except Exception as e:
            warnings.warn(f"Error in sentiment loss calculation: {e}")
            # Fallback to basic loss
            if predictions.dim() > 2:
                pred_flat = predictions.reshape(-1, predictions.size(-1))
                target_flat = targets.reshape(-1).long()
                basic_loss = F.cross_entropy(pred_flat, target_flat)
            else:
                basic_loss = F.cross_entropy(predictions, targets.long())
            
            return basic_loss, {'total': basic_loss.item()}


# Additional utility loss functions
class HuberFocalLoss(nn.Module):
    """
    Combined Huber and Focal loss for robust regression and classification
    """
    def __init__(self, delta=1.0, gamma=2.0, reduction='mean'):
        super(HuberFocalLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=delta)
        self.focal = FocalLoss(gamma=gamma, reduction='none')
        self.reduction = reduction

    def forward(self, pred_logits, targets):
        """
        Forward pass combining regression and classification losses
        
        Args:
            pred_logits: Predicted logits
            targets: Target values (classification) or scores (regression)
        
        Returns:
            Combined loss
        """
        # Focal loss for classification
        focal_loss = self.focal(pred_logits, targets.long())
        
        # Huber loss for regression
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_scores = torch.sum(pred_probs * torch.arange(pred_probs.size(-1), 
                                                         device=pred_logits.device).float(), dim=-1)
        target_scores = targets.float() / (pred_logits.size(-1) - 1)
        huber_loss = self.huber(pred_scores, target_scores)
        
        # Combine losses
        combined_loss = focal_loss + huber_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return combined_loss
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:  # mean
            return combined_loss.mean()