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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np


class LabelGroupRanking(nn.Module):
    """
    First stage: Rank label groups to identify promising candidates.
    This reduces computational cost by filtering out unlikely labels.
    """
    
    def __init__(self, hidden_size, num_label_groups):
        super(LabelGroupRanking, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_label_groups)
        
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
    Uses dot product similarity between text representation and label embeddings.
    """
    
    def __init__(self, hidden_size, num_labels, label_emb_dim=128):
        super(CandidateRanking, self).__init__()
        # Project text representation to label embedding space
        self.text_projection = nn.Linear(hidden_size, label_emb_dim)
        
        # Label embeddings (learnable)
        self.label_embeddings = nn.Embedding(num_labels, label_emb_dim)
        nn.init.xavier_uniform_(self.label_embeddings.weight)
        
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
        
        if candidate_indices is not None:
            # Training mode: only score candidate labels
            batch_size, num_candidates = candidate_indices.size()
            
            # Get candidate label embeddings: [batch_size, num_candidates, label_emb_dim]
            candidate_embs = self.label_embeddings(candidate_indices)
            
            # Compute scores: [batch_size, num_candidates]
            logits = torch.bmm(
                candidate_embs,  # [batch_size, num_candidates, label_emb_dim]
                text_repr.unsqueeze(2)  # [batch_size, label_emb_dim, 1]
            ).squeeze(2)
        else:
            # Inference mode: score all labels
            # [batch_size, label_emb_dim] @ [num_labels, label_emb_dim]^T
            logits = torch.matmul(text_repr, self.label_embeddings.weight.t())
        
        return logits


class LightXMLModel(nn.Module):
    """
    Simplified LightXML model using CTI-BERT backbone.
    
    Architecture:
    1. CTI-BERT encoder
    2. Label group ranking (coarse-grained classification)
    3. Candidate ranking with label embeddings (fine-grained)
    
    Training strategy:
    - Dynamic negative sampling: Sample hard negatives based on group predictions
    - Two-stage loss: Group classification + Candidate ranking
    """
    
    def __init__(self, model_name, num_labels, num_label_groups=50, 
                 label_emb_dim=128, dropout=0.1):
        super(LightXMLModel, self).__init__()
        
        self.num_labels = num_labels
        self.num_label_groups = num_label_groups
        
        # CTI-BERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stage 1: Label group ranking
        self.group_ranker = LabelGroupRanking(hidden_size, num_label_groups)
        
        # Stage 2: Candidate ranking
        self.candidate_ranker = CandidateRanking(hidden_size, num_labels, label_emb_dim)
        
        # Label to group mapping (simple partitioning for now)
        self.label_to_group = self._create_label_groups(num_labels, num_label_groups)
        
    def _create_label_groups(self, num_labels, num_label_groups):
        """
        Create simple label grouping by partitioning label space.
        In full LightXML, this would use clustering.
        """
        labels_per_group = num_labels // num_label_groups + 1
        label_to_group = {}
        
        for label_idx in range(num_labels):
            group_idx = label_idx // labels_per_group
            if group_idx not in label_to_group:
                label_to_group[group_idx] = []
            label_to_group[group_idx].append(label_idx)
        
        return label_to_group
    
    def get_candidates(self, group_logits, topk=3):
        """
        Select candidate labels based on top-k label groups.
        
        Args:
            group_logits: [batch_size, num_label_groups]
            topk: Number of top groups to select
        
        Returns:
            candidates: [batch_size, num_candidates] - candidate label indices
        """
        batch_size = group_logits.size(0)
        
        # Get top-k groups
        _, top_groups = torch.topk(group_logits, k=min(topk, self.num_label_groups), dim=1)
        top_groups = top_groups.cpu().numpy()
        
        # Collect candidate labels from top groups
        all_candidates = []
        for i in range(batch_size):
            candidates = []
            for group_idx in top_groups[i]:
                if group_idx in self.label_to_group:
                    candidates.extend(self.label_to_group[group_idx])
            
            # Pad if needed
            if len(candidates) < 100:
                # Add random labels to reach 100 candidates
                remaining = 100 - len(candidates)
                random_labels = np.random.choice(self.num_labels, remaining, replace=False)
                candidates.extend(random_labels.tolist())
            
            all_candidates.append(candidates[:100])  # Limit to 100 candidates
        
        return torch.LongTensor(all_candidates).to(group_logits.device)
    
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
            
            # Compute group labels from labels
            batch_size = labels.size(0)
            group_labels = torch.zeros(batch_size, self.num_label_groups, 
                                      dtype=torch.float, device=labels.device)
            
            for i in range(batch_size):
                positive_labels = torch.nonzero(labels[i]).squeeze(1).cpu().numpy()
                for label_idx in positive_labels:
                    group_idx = label_idx // (self.num_labels // self.num_label_groups + 1)
                    if group_idx < self.num_label_groups:
                        group_labels[i, group_idx] = 1.0
            
            # Generate candidates (positive + hard negatives)
            if candidate_indices is None:
                # Get top-k groups for hard negative mining
                candidates = self.get_candidates(group_logits.detach(), topk=5)
            else:
                candidates = candidate_indices
            
            # Stage 2: Candidate ranking
            candidate_logits = self.candidate_ranker(pooled_output, candidates)
            
            # Compute candidate labels
            batch_size, num_candidates = candidates.size()
            candidate_labels = torch.zeros(batch_size, num_candidates, 
                                          dtype=torch.float, device=labels.device)
            
            for i in range(batch_size):
                for j, label_idx in enumerate(candidates[i].cpu().numpy()):
                    if label_idx < self.num_labels:
                        candidate_labels[i, j] = labels[i, label_idx]
            
            # Combined loss
            loss_fn = nn.BCEWithLogitsLoss()
            group_loss = loss_fn(group_logits, group_labels)
            candidate_loss = loss_fn(candidate_logits, candidate_labels)
            loss = group_loss + candidate_loss
            
            return group_logits, candidate_logits, loss
        else:
            # Inference mode: score all labels
            logits = self.candidate_ranker(pooled_output, candidate_indices=None)
            return logits


def load_light_xml_model(model_name, num_labels, device, num_label_groups=50, 
                        label_emb_dim=128, dropout=0.1):
    """
    Load LightXML model with CTI-BERT backbone.
    
    Args:
        model_name: HuggingFace model name (e.g., 'ibm-research/CTI-BERT')
        num_labels: Number of output labels
        device: 'cuda' or 'cpu'
        num_label_groups: Number of label groups for coarse ranking
        label_emb_dim: Dimension of label embeddings
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
