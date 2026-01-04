"""
Simplified LightXML implementation for CTI-BERT.

Based on: "LightXML: Transformer with Dynamic Negative Sampling for 
High-Performance Extreme Multi-label Text Classification" (AAAI 2021)
https://arxiv.org/abs/2101.03305

Key ideas:
1. Two-stage approach: Label group ranking + Candidate refinement
2. Dynamic negative sampling during training
3. Label embedding space for efficient candidate generation

This is a simplified version adapted for 499 labels (no complex clustering).

IMPROVEMENTS (v2):
- Fixed label-to-group mapping consistency
- Ensured all positive labels are included in candidates
- Deeper text projection with LayerNorm
- L2 normalized embeddings for cosine similarity
- Label bias for class imbalance handling
- Learnable loss weighting between stages
- Proper weight initialization
- Frozen embedding layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
import math


class LabelGroupRanking(nn.Module):
    """
    First stage: Rank label groups to identify promising candidates.
    This reduces computational cost by filtering out unlikely labels.
    """
    
    def __init__(self, hidden_size, num_label_groups):
        super(LabelGroupRanking, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_label_groups)
        )
        self._init_weights()
        
    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, pooled_output):
        """
        Args:
            pooled_output: [batch_size, hidden_size]
        
        Returns:
            group_logits: [batch_size, num_label_groups]
        """
        return self.classifier(pooled_output)


class CandidateRanking(nn.Module):
    """
    Second stage: Rank candidate labels using label embeddings.
    Uses normalized dot product (cosine similarity) between text representation and label embeddings.
    """
    
    def __init__(self, hidden_size, num_labels, label_emb_dim=256):
        super(CandidateRanking, self).__init__()
        self.hidden_size = hidden_size
        self.label_emb_dim = label_emb_dim
        self.scale = math.sqrt(label_emb_dim)
        
        # Deeper text projection for better representation
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, label_emb_dim)
        )
        
        # Label embeddings (learnable) with better initialization
        self.label_embeddings = nn.Embedding(num_labels, label_emb_dim)
        nn.init.normal_(self.label_embeddings.weight, mean=0, std=0.02)
        
        # Bias for each label (helps with class imbalance)
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.text_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, pooled_output, candidate_indices=None):
        """
        Args:
            pooled_output: [batch_size, hidden_size]
            candidate_indices: [batch_size, num_candidates] - indices of candidate labels
        
        Returns:
            logits: [batch_size, num_labels] or [batch_size, num_candidates]
        """
        # Project text to label space: [batch_size, label_emb_dim]
        text_repr = self.text_projection(pooled_output)
        text_repr = F.normalize(text_repr, p=2, dim=-1)  # L2 normalize for cosine similarity
        
        if candidate_indices is not None:
            # Training mode: only score candidate labels
            batch_size, num_candidates = candidate_indices.size()
            
            # Get candidate label embeddings: [batch_size, num_candidates, label_emb_dim]
            candidate_embs = self.label_embeddings(candidate_indices)
            candidate_embs = F.normalize(candidate_embs, p=2, dim=-1)  # L2 normalize
            
            # Get candidate biases
            candidate_bias = self.label_bias[candidate_indices]  # [batch_size, num_candidates]
            
            # Compute scaled dot product scores: [batch_size, num_candidates]
            logits = torch.bmm(
                candidate_embs,  # [batch_size, num_candidates, label_emb_dim]
                text_repr.unsqueeze(2)  # [batch_size, label_emb_dim, 1]
            ).squeeze(2) * self.scale + candidate_bias
        else:
            # Inference mode: score all labels
            all_label_embs = F.normalize(self.label_embeddings.weight, p=2, dim=-1)
            # [batch_size, label_emb_dim] @ [num_labels, label_emb_dim]^T
            logits = torch.matmul(text_repr, all_label_embs.t()) * self.scale + self.label_bias
        
        return logits


class LightXMLModel(nn.Module):
    """
    Improved LightXML model using CTI-BERT backbone.
    
    Architecture:
    1. CTI-BERT encoder
    2. Label group ranking (coarse-grained classification)
    3. Candidate ranking with label embeddings (fine-grained)
    
    Training strategy:
    - Dynamic negative sampling: Sample hard negatives based on group predictions
    - Two-stage loss: Group classification + Candidate ranking
    - Ensure all positive labels are included in candidates
    """
    
    def __init__(self, model_name, num_labels, num_label_groups=50, 
                 label_emb_dim=256, dropout=0.1):
        super(LightXMLModel, self).__init__()
        
        self.num_labels = num_labels
        self.num_label_groups = num_label_groups
        self.labels_per_group = (num_labels + num_label_groups - 1) // num_label_groups
        
        # CTI-BERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Freeze embedding layer to prevent overfitting
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stage 1: Label group ranking
        self.group_ranker = LabelGroupRanking(hidden_size, num_label_groups)
        
        # Stage 2: Candidate ranking
        self.candidate_ranker = CandidateRanking(hidden_size, num_labels, label_emb_dim)
        
        # Create label-to-group and group-to-labels mappings
        self.register_buffer('label_to_group_map', self._create_label_to_group_map())
        self.group_to_labels = self._create_group_to_labels()
        
        # Learnable loss weighting
        self.log_loss_weight = nn.Parameter(torch.tensor(0.0))
        
    def _create_label_to_group_map(self):
        """Create tensor mapping each label to its group."""
        label_to_group = torch.zeros(self.num_labels, dtype=torch.long)
        for label_idx in range(self.num_labels):
            group_idx = min(label_idx // self.labels_per_group, self.num_label_groups - 1)
            label_to_group[label_idx] = group_idx
        return label_to_group
    
    def _create_group_to_labels(self):
        """Create dictionary mapping each group to its labels."""
        group_to_labels = {}
        for group_idx in range(self.num_label_groups):
            start_idx = group_idx * self.labels_per_group
            end_idx = min(start_idx + self.labels_per_group, self.num_labels)
            group_to_labels[group_idx] = list(range(start_idx, end_idx))
        return group_to_labels
    
    def get_candidates(self, group_logits, labels=None, num_candidates=150, topk_groups=5):
        """
        Select candidate labels based on top-k label groups.
        Ensures all positive labels are included.
        
        Args:
            group_logits: [batch_size, num_label_groups]
            labels: [batch_size, num_labels] - ground truth (to ensure positives are included)
            num_candidates: Total number of candidates per sample
            topk_groups: Number of top groups to select
        
        Returns:
            candidates: [batch_size, num_candidates] - candidate label indices
        """
        batch_size = group_logits.size(0)
        device = group_logits.device
        
        # Get top-k groups
        _, top_groups = torch.topk(group_logits, k=min(topk_groups, self.num_label_groups), dim=1)
        top_groups = top_groups.cpu().numpy()
        
        all_candidates = []
        for i in range(batch_size):
            candidates_set = set()
            
            # First, add all positive labels (critical for learning!)
            if labels is not None:
                positive_labels = torch.nonzero(labels[i]).squeeze(-1).cpu().numpy()
                if positive_labels.ndim == 0:
                    positive_labels = [int(positive_labels)]
                else:
                    positive_labels = positive_labels.tolist()
                candidates_set.update(positive_labels)
            
            # Then, add labels from top groups (hard negatives)
            for group_idx in top_groups[i]:
                if group_idx in self.group_to_labels:
                    candidates_set.update(self.group_to_labels[group_idx])
                if len(candidates_set) >= num_candidates:
                    break
            
            # Convert to list
            candidates = list(candidates_set)
            
            # If we need more candidates, add random ones
            if len(candidates) < num_candidates:
                remaining = num_candidates - len(candidates)
                available = set(range(self.num_labels)) - candidates_set
                if len(available) > 0:
                    random_labels = np.random.choice(list(available), 
                                                     min(remaining, len(available)), 
                                                     replace=False)
                    candidates.extend(random_labels.tolist())
            
            # Truncate and pad to exact size
            candidates = candidates[:num_candidates]
            while len(candidates) < num_candidates:
                candidates.append(0)  # Pad with label 0
            
            all_candidates.append(candidates)
        
        return torch.LongTensor(all_candidates).to(device)
    
    def forward(self, input_ids, attention_mask, labels=None, candidate_indices=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, num_labels] - ground truth (training only)
            candidate_indices: [batch_size, num_candidates] - pre-computed candidates
        
        Returns:
            If training:
                group_logits: [batch_size, num_label_groups]
                candidate_logits: [batch_size, num_candidates]
                loss: Combined loss
            If inference:
                logits: [batch_size, num_labels]
        """
        # Encode with CTI-BERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Stage 1: Label group ranking
        group_logits = self.group_ranker(pooled_output)
        
        if self.training and labels is not None:
            # Training mode: use dynamic negative sampling
            batch_size = labels.size(0)
            
            # Compute group labels using registered buffer
            group_labels = torch.zeros(batch_size, self.num_label_groups, 
                                      dtype=torch.float, device=labels.device)
            
            for i in range(batch_size):
                positive_labels = torch.nonzero(labels[i]).squeeze(-1)
                if positive_labels.numel() > 0:
                    positive_groups = self.label_to_group_map[positive_labels]
                    group_labels[i].scatter_(0, positive_groups, 1.0)
            
            # Generate candidates (positive + hard negatives)
            if candidate_indices is None:
                candidates = self.get_candidates(group_logits.detach(), labels=labels, 
                                                 num_candidates=150, topk_groups=5)
            else:
                candidates = candidate_indices
            
            # Stage 2: Candidate ranking
            candidate_logits = self.candidate_ranker(pooled_output, candidates)
            
            # Compute candidate labels efficiently
            batch_size, num_candidates = candidates.size()
            candidate_labels = torch.zeros(batch_size, num_candidates, 
                                          dtype=torch.float, device=labels.device)
            
            # Gather labels for candidate indices
            for i in range(batch_size):
                for j in range(num_candidates):
                    label_idx = candidates[i, j].item()
                    if label_idx < self.num_labels:
                        candidate_labels[i, j] = labels[i, label_idx]
            
            # Combined loss with learnable weighting
            loss_fn = nn.BCEWithLogitsLoss()
            group_loss = loss_fn(group_logits, group_labels)
            candidate_loss = loss_fn(candidate_logits, candidate_labels)
            
            # Learnable loss combination: sigmoid ensures weight is in (0, 1)
            loss_weight = torch.sigmoid(self.log_loss_weight)
            loss = loss_weight * group_loss + (1 - loss_weight) * candidate_loss
            
            return group_logits, candidate_logits, loss
        else:
            # Inference mode: score all labels
            logits = self.candidate_ranker(pooled_output, candidate_indices=None)
            return logits


def load_light_xml_model(model_name, num_labels, device, num_label_groups=50, 
                        label_emb_dim=256, dropout=0.1):
    """
    Load LightXML model with CTI-BERT backbone.
    
    Args:
        model_name: HuggingFace model name (e.g., 'ibm-research/CTI-BERT')
        num_labels: Number of output labels
        device: 'cuda' or 'cpu'
        num_label_groups: Number of label groups for coarse ranking
        label_emb_dim: Dimension of label embeddings (256 recommended)
        dropout: Dropout rate
    
    Returns:
        model: LightXML model
    """
    model = LightXMLModel(
        model_name, 
        num_labels, 
        num_label_groups=num_label_groups,
        label_emb_dim=label_emb_dim,
        dropout=dropout
    )
    model = model.to(device)
    return model
