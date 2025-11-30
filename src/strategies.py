"""
Different training strategies for handling class imbalance.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np


def compute_pos_weights(train_dataset, num_labels: int, min_weight: float = 1.0, max_weight: float = 500.0):
    """
    Compute positive class weights for each label based on frequency.
    
    Weight formula: pos_weight[i] = num_negatives / num_positives
    This gives higher weight to rare labels.
    
    Args:
        train_dataset: Training dataset
        num_labels: Number of labels
        min_weight: Minimum weight (for very frequent labels)
        max_weight: Maximum weight (for very rare labels)
        
    Returns:
        Tensor of positive weights for each label
    """
    print("\n📊 Computing positive class weights...")
    
    # Count positive samples for each label
    all_labels = torch.stack([train_dataset[i]['labels'] for i in range(len(train_dataset))])
    pos_counts = all_labels.sum(dim=0).float()
    neg_counts = len(train_dataset) - pos_counts
    
    # pos_weight = num_negatives / num_positives
    # Handle divide by zero for labels with no positive samples
    pos_weights = torch.zeros(num_labels)
    for i in range(num_labels):
        if pos_counts[i] > 0:
            weight = neg_counts[i] / pos_counts[i]
            # Clip weights to reasonable range
            pos_weights[i] = torch.clamp(torch.tensor(weight), min=min_weight, max=max_weight)
        else:
            pos_weights[i] = max_weight
    
    print(f"   Min weight: {pos_weights.min():.2f}")
    print(f"   Max weight: {pos_weights.max():.2f}")
    print(f"   Mean weight: {pos_weights.mean():.2f}")
    print(f"   Median weight: {pos_weights.median():.2f}")
    
    return pos_weights


def create_oversampled_dataloader(dataset, batch_size: int, num_labels: int, 
                                  oversample_factor: int = 10):
    """
    Create a dataloader with oversampling for rare labels.
    
    Samples with rare labels are repeated more frequently.
    
    Args:
        dataset: Dataset to oversample
        batch_size: Batch size
        num_labels: Number of labels
        oversample_factor: How many times to repeat samples with rare labels
        
    Returns:
        DataLoader with weighted sampling
    """
    print(f"\n📊 Creating oversampled dataloader (factor={oversample_factor})...")
    
    # Compute label frequencies
    all_labels = torch.stack([dataset[i]['labels'] for i in range(len(dataset))])
    label_counts = all_labels.sum(dim=0)
    
    # Compute sample weights based on rarest label
    sample_weights = []
    for i in range(len(dataset)):
        labels = dataset[i]['labels']
        label_indices = torch.where(labels == 1)[0]
        
        if len(label_indices) == 0:
            # No labels - give default weight
            sample_weights.append(1.0)
        else:
            # Weight based on rarest label in this sample
            rarest_count = label_counts[label_indices].min().item()
            # Inverse frequency weighting
            weight = 1.0 / (rarest_count + 1)  # +1 to avoid division by zero
            sample_weights.append(weight)
    
    sample_weights = torch.tensor(sample_weights)
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(dataset)
    
    print(f"   Sample weight range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset) * oversample_factor,
        replacement=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print(f"   Effective dataset size: {len(dataset) * oversample_factor}")
    
    return dataloader


def filter_top_k_labels(dataset, label_list: list, k: int = 100):
    """
    Create a new dataset with only the top-K most frequent labels.
    
    This reduces the problem complexity and can help verify if the model
    can learn anything at all.
    
    Args:
        dataset: Original dataset
        label_list: List of all label names
        k: Number of top labels to keep
        
    Returns:
        Tuple of (filtered_dataset, filtered_label_list, label_mapping)
    """
    print(f"\n📊 Filtering to top {k} most frequent labels...")
    
    # Count label frequencies
    all_labels = torch.stack([dataset[i]['labels'] for i in range(len(dataset))])
    label_counts = all_labels.sum(dim=0)
    
    # Get top-k labels
    top_k_counts, top_k_indices = torch.topk(label_counts, k=k)
    top_k_indices = top_k_indices.tolist()
    
    print(f"   Top {k} labels cover {top_k_counts.sum().item():.0f} samples")
    print(f"   Frequency range: {top_k_counts.min():.0f} - {top_k_counts.max():.0f}")
    
    # Create filtered label list
    filtered_label_list = [label_list[i] for i in top_k_indices]
    
    # Create label mapping (old index -> new index, or -1 if not in top-k)
    label_mapping = torch.full((len(label_list),), -1, dtype=torch.long)
    for new_idx, old_idx in enumerate(top_k_indices):
        label_mapping[old_idx] = new_idx
    
    # Filter dataset
    class FilteredDataset(Dataset):
        def __init__(self, original_dataset, label_mapping, k):
            self.original_dataset = original_dataset
            self.label_mapping = label_mapping
            self.k = k
            
            # Pre-filter valid samples (those with at least one top-k label)
            self.valid_indices = []
            for i in range(len(original_dataset)):
                old_labels = original_dataset[i]['labels']
                new_labels = self._convert_labels(old_labels)
                if new_labels.sum() > 0:  # Has at least one valid label
                    self.valid_indices.append(i)
        
        def _convert_labels(self, old_labels):
            """Convert old label indices to new label indices."""
            new_labels = torch.zeros(self.k)
            old_indices = torch.where(old_labels == 1)[0]
            for old_idx in old_indices:
                new_idx = self.label_mapping[old_idx]
                if new_idx >= 0:  # Valid label
                    new_labels[new_idx] = 1
            return new_labels
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            real_idx = self.valid_indices[idx]
            sample = self.original_dataset[real_idx]
            
            return {
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask'],
                'labels': self._convert_labels(sample['labels'])
            }
    
    filtered_dataset = FilteredDataset(dataset, label_mapping, k)
    
    print(f"   Original samples: {len(dataset)}")
    print(f"   Filtered samples: {len(filtered_dataset)}")
    print(f"   Retention rate: {len(filtered_dataset)/len(dataset)*100:.1f}%")
    
    return filtered_dataset, filtered_label_list, label_mapping


def get_strategy_config(strategy_name: str, train_dataset, num_labels: int, label_list: list):
    """
    Get configuration for a specific training strategy.
    
    Args:
        strategy_name: One of 'baseline', 'focal', 'weighted', 'oversampled', 'subset'
        train_dataset: Training dataset
        num_labels: Number of labels
        label_list: List of label names
        
    Returns:
        Dictionary with strategy configuration
    """
    configs = {
        'baseline': {
            'name': 'Baseline BCE',
            'description': 'Standard Binary Cross-Entropy Loss',
            'use_focal_loss': False,
            'pos_weight': None,
            'dataset': train_dataset,
            'num_labels': num_labels,
            'label_list': label_list,
            'custom_dataloader': None
        },
        
        'focal_weak': {
            'name': 'Focal Loss (gamma=2)',
            'description': 'Focal Loss with standard parameters',
            'use_focal_loss': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'pos_weight': None,
            'dataset': train_dataset,
            'num_labels': num_labels,
            'label_list': label_list,
            'custom_dataloader': None
        },
        
        'focal_strong': {
            'name': 'Focal Loss (gamma=5)',
            'description': 'Focal Loss with aggressive focusing',
            'use_focal_loss': True,
            'focal_alpha': 0.75,
            'focal_gamma': 5.0,
            'pos_weight': None,
            'dataset': train_dataset,
            'num_labels': num_labels,
            'label_list': label_list,
            'custom_dataloader': None
        },
        
        'weighted': {
            'name': 'Weighted BCE',
            'description': 'BCE with per-label class weights',
            'use_focal_loss': False,
            'pos_weight': compute_pos_weights(train_dataset, num_labels),
            'dataset': train_dataset,
            'num_labels': num_labels,
            'label_list': label_list,
            'custom_dataloader': None
        },
        
        'oversampled': {
            'name': 'Oversampled BCE',
            'description': 'BCE with oversampling of rare labels',
            'use_focal_loss': False,
            'pos_weight': None,
            'dataset': train_dataset,
            'num_labels': num_labels,
            'label_list': label_list,
            'custom_dataloader': lambda batch_size: create_oversampled_dataloader(
                train_dataset, batch_size, num_labels, oversample_factor=5
            )
        },
        
        'subset_50': {
            'name': 'Top-50 Labels',
            'description': 'Train only on 50 most frequent labels',
            'use_focal_loss': False,
            'pos_weight': None,
            'dataset': None,  # Will be set below
            'num_labels': 50,
            'label_list': None,  # Will be set below
            'custom_dataloader': None
        },
        
        'subset_100': {
            'name': 'Top-100 Labels',
            'description': 'Train only on 100 most frequent labels',
            'use_focal_loss': False,
            'pos_weight': None,
            'dataset': None,  # Will be set below
            'num_labels': 100,
            'label_list': None,  # Will be set below
            'custom_dataloader': None
        }
    }
    
    # Handle subset strategies
    if strategy_name == 'subset_50':
        filtered_ds, filtered_labels, _ = filter_top_k_labels(train_dataset, label_list, k=50)
        configs['subset_50']['dataset'] = filtered_ds
        configs['subset_50']['label_list'] = filtered_labels
    
    elif strategy_name == 'subset_100':
        filtered_ds, filtered_labels, _ = filter_top_k_labels(train_dataset, label_list, k=100)
        configs['subset_100']['dataset'] = filtered_ds
        configs['subset_100']['label_list'] = filtered_labels
    
    return configs[strategy_name]
