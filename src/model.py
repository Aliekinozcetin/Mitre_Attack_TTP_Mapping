"""
Model definition module for CTI-BERT TTP tagging.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BERTForMultiLabelClassification(nn.Module):
    """
    BERT-based model for multi-label classification.
    Uses BCEWithLogitsLoss for training.
    """
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pretrained BERT model
            num_labels: Number of labels for classification
            dropout: Dropout probability
        """
        super(BERTForMultiLabelClassification, self).__init__()
        
        self.num_labels = num_labels
        
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
            # Binary Cross Entropy with Logits Loss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


def load_model(model_name: str, num_labels: int, device: str = 'cuda') -> BERTForMultiLabelClassification:
    """
    Load and initialize a model.
    
    Args:
        model_name: Name of the pretrained model
        num_labels: Number of labels
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    print(f"Loading model: {model_name}")
    print(f"Number of labels: {num_labels}")
    
    model = BERTForMultiLabelClassification(model_name, num_labels)
    
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
