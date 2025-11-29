"""
Evaluation module for CTI-BERT TTP tagging.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple


def predict(model, dataloader, device, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from the model.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        threshold: Classification threshold
        
    Returns:
        Tuple of (predictions, true_labels)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            logits = outputs['logits']
            predictions = torch.sigmoid(logits)
            predictions = (predictions > threshold).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    return all_predictions, all_labels


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    # Micro-averaged metrics
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    micro_precision = precision_score(labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(labels, predictions, average='micro', zero_division=0)
    
    # Macro-averaged metrics
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    # Samples-averaged metrics
    samples_f1 = f1_score(labels, predictions, average='samples', zero_division=0)
    
    # Subset accuracy
    subset_accuracy = accuracy_score(labels, predictions)
    
    metrics = {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'samples_f1': samples_f1,
        'subset_accuracy': subset_accuracy
    }
    
    return metrics


def evaluate_model(
    model,
    test_dataset,
    batch_size: int = 16,
    device: str = 'cuda',
    threshold: float = 0.5,
    label_list: list = None
) -> Dict:
    """
    Complete evaluation pipeline.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        batch_size: Batch size for evaluation
        device: Device to run on
        threshold: Classification threshold
        label_list: List of label names (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    model = model.to(device)
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\n{'='*50}")
    print(f"Starting evaluation:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Threshold: {threshold}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"{'='*50}\n")
    
    # Get predictions
    predictions, labels = predict(model, test_loader, device, threshold)
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nMicro-averaged metrics (overall):")
    print(f"  F1 Score:  {metrics['micro_f1']:.4f}")
    print(f"  Precision: {metrics['micro_precision']:.4f}")
    print(f"  Recall:    {metrics['micro_recall']:.4f}")
    
    print(f"\nMacro-averaged metrics (per label):")
    print(f"  F1 Score:  {metrics['macro_f1']:.4f}")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall:    {metrics['macro_recall']:.4f}")
    
    print(f"\nOther metrics:")
    print(f"  Samples F1:      {metrics['samples_f1']:.4f}")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print("="*50 + "\n")
    
    # Add predictions and labels to metrics
    metrics['predictions'] = predictions
    metrics['labels'] = labels
    
    return metrics


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Use this module through main.py for complete pipeline.")
