"""
Simplified AttentionXML implementation for CTI-BERT.

Based on: "AttentionXML: Label Tree-based Attention-Aware Deep Model for 
High-Performance Extreme Multi-Label Text Classification" (NeurIPS 2019)
https://arxiv.org/abs/1811.01727

Key idea: Multi-label attention mechanism - each label gets its own attention
weights over the input sequence, allowing the model to focus on different parts
of the text for different labels.

This is a simplified version adapted for 499 labels (no label tree needed).

IMPROVEMENTS (v2):
- Fixed attention initialization with proper scaling
- Added LayerNorm for stability
- Combined CLS pooling with attention for better feature extraction
- Added residual connection
- Improved classifier with batch normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math


class MultiLabelAttention(nn.Module):
    """
    Improved Multi-label attention mechanism with scaled dot-product attention.
    Each label has its own attention weights to focus on relevant text parts.
    """
    
    def __init__(self, hidden_size, num_labels):
        super(MultiLabelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.scale = math.sqrt(hidden_size)
        
        # Label query projection - better than raw parameters
        self.label_queries = nn.Linear(hidden_size, num_labels, bias=False)
        
        # Key and Value projections for better representation
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.xavier_uniform_(self.label_queries.weight, gain=0.1)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        
    def forward(self, sequence_output, attention_mask):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            attended_output: [batch_size, num_labels, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.size()
        
        # Project keys and values
        keys = self.key_proj(sequence_output)  # [batch_size, seq_len, hidden_size]
        values = self.value_proj(sequence_output)  # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores using label queries
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_labels]
        attention_scores = self.label_queries(keys)
        
        # Transpose to [batch_size, num_labels, seq_len]
        attention_scores = attention_scores.transpose(1, 2)
        
        # Scale scores for stability
        attention_scores = attention_scores / self.scale
        
        # Mask padding tokens with large negative value
        mask = attention_mask.unsqueeze(1).float()  # [batch_size, 1, seq_len]
        attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        
        # Softmax to get attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, num_labels, seq_len]
        
        # Apply attention to values: [batch_size, num_labels, hidden_size]
        attended_output = torch.bmm(attention_probs, values)
        
        # Apply layer normalization
        attended_output = self.layer_norm(attended_output)
        
        return attended_output


class AttentionXMLModel(nn.Module):
    """
    Improved AttentionXML model using CTI-BERT backbone.
    
    Architecture:
    1. CTI-BERT encoder
    2. Multi-label attention (each label attends to different parts of text)
    3. CLS token pooling combined with attention
    4. Label-specific classification with proper normalization
    """
    
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(AttentionXMLModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_labels = num_labels
        
        # Freeze first few layers to prevent overfitting
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Multi-label attention
        self.attention = MultiLabelAttention(self.hidden_size, num_labels)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # CLS projection for combining with attention
        self.cls_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Improved classifier with batch normalization
        # Each label gets its own output from the attended representation
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Initialize classifier properly
        self._init_classifier()
        
    def _init_classifier(self):
        """Initialize classifier weights properly."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_labels]
        """
        # Encode with CTI-BERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get CLS token representation
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        cls_projected = self.cls_projection(cls_output)  # [batch_size, hidden_size]
        
        # Multi-label attention
        attended_output = self.attention(sequence_output, attention_mask)
        # [batch_size, num_labels, hidden_size]
        attended_output = self.attention_dropout(attended_output)
        
        batch_size = input_ids.size(0)
        
        # Expand CLS for each label: [batch_size, num_labels, hidden_size]
        cls_expanded = cls_projected.unsqueeze(1).expand(-1, self.num_labels, -1)
        
        # Concatenate CLS with attention output
        # [batch_size, num_labels, hidden_size * 2]
        combined = torch.cat([attended_output, cls_expanded], dim=-1)
        
        # Reshape for classifier: [batch_size * num_labels, hidden_size * 2]
        combined_flat = combined.view(-1, self.hidden_size * 2)
        
        # Apply classifier: [batch_size * num_labels, 1]
        logits_flat = self.classifier(combined_flat)
        
        # Reshape back: [batch_size, num_labels]
        logits = logits_flat.view(batch_size, self.num_labels)
        
        return logits


def load_attention_xml_model(model_name, num_labels, device, dropout=0.1):
    """
    Load AttentionXML model with CTI-BERT backbone.
    
    Args:
        model_name: HuggingFace model name (e.g., 'ibm-research/CTI-BERT')
        num_labels: Number of output labels
        device: 'cuda' or 'cpu'
        dropout: Dropout rate
    
    Returns:
        model: AttentionXML model
    """
    model = AttentionXMLModel(model_name, num_labels, dropout=dropout)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AttentionXML Model loaded:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model
