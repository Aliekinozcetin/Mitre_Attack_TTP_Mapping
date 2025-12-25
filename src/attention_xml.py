"""
Simplified AttentionXML implementation for CTI-BERT.

Based on: "AttentionXML: Label Tree-based Attention-Aware Deep Model for 
High-Performance Extreme Multi-Label Text Classification" (NeurIPS 2019)
https://arxiv.org/abs/1811.01727

Key idea: Multi-label attention mechanism - each label gets its own attention
weights over the input sequence, allowing the model to focus on different parts
of the text for different labels.

This is a simplified version adapted for 499 labels (no label tree needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MultiLabelAttention(nn.Module):
    """
    Multi-label attention mechanism.
    Each label has its own attention weights to focus on relevant text parts.
    """
    
    def __init__(self, hidden_size, num_labels):
        super(MultiLabelAttention, self).__init__()
        # Each label has its own attention query vector
        self.attention_weights = nn.Parameter(torch.randn(num_labels, hidden_size))
        nn.init.xavier_uniform_(self.attention_weights)
        
    def forward(self, sequence_output, attention_mask):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            attended_output: [batch_size, num_labels, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.size()
        num_labels = self.attention_weights.size(0)
        
        # Expand attention weights for batch
        # [num_labels, hidden_size] -> [batch_size, num_labels, hidden_size]
        attention_w = self.attention_weights.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute attention scores: [batch_size, num_labels, seq_len]
        attention_scores = torch.bmm(attention_w, sequence_output.transpose(1, 2))
        
        # Mask padding tokens
        # [batch_size, seq_len] -> [batch_size, 1, seq_len]
        mask = attention_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, num_labels, seq_len]
        
        # Apply attention: [batch_size, num_labels, hidden_size]
        attended_output = torch.bmm(attention_probs, sequence_output)
        
        return attended_output


class AttentionXMLModel(nn.Module):
    """
    Simplified AttentionXML model using CTI-BERT backbone.
    
    Architecture:
    1. CTI-BERT encoder
    2. Multi-label attention (each label attends to different parts of text)
    3. Label-specific classification layers
    """
    
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(AttentionXMLModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Multi-label attention
        self.attention = MultiLabelAttention(hidden_size, num_labels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Label-specific linear layers (shared across labels for efficiency)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
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
        
        # Multi-label attention
        attended_output = self.attention(sequence_output, attention_mask)
        # [batch_size, num_labels, hidden_size]
        
        # Dropout
        attended_output = self.dropout(attended_output)
        
        # Classify each label independently
        # Reshape: [batch_size * num_labels, hidden_size]
        batch_size, num_labels, hidden_size = attended_output.size()
        attended_flat = attended_output.view(-1, hidden_size)
        
        # Apply classifier: [batch_size * num_labels, 1]
        logits_flat = self.classifier(attended_flat)
        
        # Reshape back: [batch_size, num_labels]
        logits = logits_flat.view(batch_size, num_labels)
        
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
    return model
