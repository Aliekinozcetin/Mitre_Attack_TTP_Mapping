"""
Model definition module for CTI-BERT TTP tagging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This loss down-weights easy examples (high confidence correct predictions)
    and focuses on hard examples (misclassified or low confidence).
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for class imbalance (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (before sigmoid) [batch_size, num_labels]
            targets: Binary ground truth [batch_size, num_labels]
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # For positive class (target=1): p_t = p
        # For negative class (target=0): p_t = 1 - p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BERTForMultiLabelClassification(nn.Module):
    """
    BERT-based model for multi-label classification.
    Supports both BCEWithLogitsLoss and Focal Loss for training.
    """
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, 
                 use_focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pretrained BERT model
            num_labels: Number of labels for classification
            dropout: Dropout probability
            use_focal_loss: Whether to use Focal Loss instead of BCE
            focal_alpha: Alpha parameter for Focal Loss (class balance weight)
            focal_gamma: Gamma parameter for Focal Loss (focusing parameter)
        """
        super(BERTForMultiLabelClassification, self).__init__()
        
        self.num_labels = num_labels
        self.use_focal_loss = use_focal_loss
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()
        
        # Loss function
        if use_focal_loss:
            self.loss_fct = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()
            print("Using BCEWithLogitsLoss")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing loss and logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


def load_model(model_name: str, num_labels: int, device: str = 'cuda', 
               use_focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> BERTForMultiLabelClassification:
    """
    Load and initialize a model.
    
    Args:
        model_name: Name of the pretrained model
        num_labels: Number of labels
        device: Device to load model on
        use_focal_loss: Whether to use Focal Loss instead of BCE
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
        
    Returns:
        Initialized model
    """
    print(f"Loading model: {model_name}")
    print(f"Number of labels: {num_labels}")
    
    model = BERTForMultiLabelClassification(
        model_name, num_labels, 
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU")
    else:
        print(f"Model loaded on CPU")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model loading
    print("Testing BERT-base-uncased:")
    model_bert = load_model("bert-base-uncased", num_labels=100, device='cpu')
