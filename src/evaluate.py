"""
Evaluation module for CTI-BERT TTP tagging.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss
)
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple


def predict(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions (probabilities) from the model.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        
    Returns:
        Tuple of (probabilities, true_labels)
    """
    model.eval()
    
    all_probs = []
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
            
            # Get predictions (probabilities)
            logits = outputs['logits']
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    return all_probs, all_labels


def calculate_mean_average_precision(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Mean Average Precision (mAP) for multi-label classification.
    
    mAP rewards models that rank correct TTPs at the top of the prediction list.
    For each sample, it calculates Average Precision (area under precision-recall curve),
    then averages across all samples.
    
    Args:
        probs: Prediction probabilities (n_samples, n_labels)
        labels: True binary labels (n_samples, n_labels)
        
    Returns:
        Mean Average Precision score
    """
    average_precisions = []
    
    for i in range(len(labels)):
        true_labels = np.where(labels[i] == 1)[0]
        
        # Skip samples with no positive labels
        if len(true_labels) == 0:
            continue
        
        # Get predicted ranking (sorted by probability, descending)
        ranking = np.argsort(probs[i])[::-1]
        
        # Calculate precision at each relevant position
        precisions_at_k = []
        num_hits = 0
        
        for k, pred_label in enumerate(ranking, start=1):
            if pred_label in true_labels:
                num_hits += 1
                precision_at_k = num_hits / k
                precisions_at_k.append(precision_at_k)
        
        # Average Precision for this sample
        if len(precisions_at_k) > 0:
            ap = np.mean(precisions_at_k)
            average_precisions.append(ap)
    
    # Mean Average Precision across all samples
    return np.mean(average_precisions) if len(average_precisions) > 0 else 0.0


def calculate_at_k(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> Tuple[float, float]:
    """
    Calculate Precision@K and Recall@K for multi-label classification.
    
    Args:
        probs: Prediction probabilities
        labels: True labels
        k: Top k predictions to consider
        
    Returns:
        Tuple of (precision@k, recall@k)
    """
    # Get indices of top k probabilities for each sample
    # argsort sorts ascending, so we take last k and reverse
    top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    
    precisions = []
    recalls = []
    
    for i in range(len(labels)):
        true_labels = np.where(labels[i] == 1)[0]
        if len(true_labels) == 0:
            continue
            
        pred_labels = top_k_indices[i]
        
        # Intersection of predicted top-k and true labels
        hits = len(set(true_labels) & set(pred_labels))
        
        # Precision@K: hits / k
        precisions.append(hits / k)
        
        # Recall@K: hits / num_true_labels
        recalls.append(hits / len(true_labels))
        
    return np.mean(precisions), np.mean(recalls)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        probs: Prediction probabilities
        labels: True labels
        threshold: Threshold for binary predictions (used for F1)
        
    Returns:
        Dictionary of metrics
    """
    # Binarize predictions for F1
    predictions = (probs > threshold).astype(int)
    
    # Micro-averaged metrics
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    micro_precision = precision_score(labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(labels, predictions, average='micro', zero_division=0)
    
    # Calculate @K metrics
    p5, r5 = calculate_at_k(probs, labels, k=5)
    p10, r10 = calculate_at_k(probs, labels, k=10)
    
    # Calculate mAP (Mean Average Precision)
    # This measures ranking quality - how well correct TTPs are ranked at the top
    mean_ap = calculate_mean_average_precision(probs, labels)
    
    # Hamming Loss: fraction of labels that are incorrectly predicted
    # Lower is better (0 = perfect, 1 = all wrong)
    hamming = hamming_loss(labels, predictions)
    
    # Example-based accuracy (sample-wise exact match)
    # For each example, check if all predicted labels match all true labels
    example_based_accuracy = np.mean(np.all(predictions == labels, axis=1))
    
    metrics = {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'precision_at_5': p5,
        'recall_at_5': r5,
        'precision_at_10': p10,
        'recall_at_10': r10,
        'mean_average_precision': mean_ap,
        'hamming_loss': hamming,
        'example_based_accuracy': example_based_accuracy
    }
    
    return metrics


def evaluate_model(
    model,
    test_dataset=None,
    test_dataloader=None,
    batch_size: int = 16,
    device: str = 'cuda',
    threshold: float = 0.5,
    label_list: list = None,
    label_names: list = None  # Alias for label_list
) -> Dict:
    """
    Complete evaluation pipeline.
    
    Args:
        model: Trained model
        test_dataset: Test dataset (provide this OR test_dataloader)
        test_dataloader: Test DataLoader (provide this OR test_dataset)
        batch_size: Batch size for evaluation (only used if test_dataset provided)
        device: Device to run on
        threshold: Classification threshold
        label_list: List of label names (optional)
        label_names: Alias for label_list (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    model = model.to(device)
    
    # Handle label_names alias
    if label_names is not None and label_list is None:
        label_list = label_names
    
    # Create data loader if not provided
    if test_dataloader is None:
        if test_dataset is None:
            raise ValueError("Must provide either test_dataset or test_dataloader")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        test_loader = test_dataloader
    
    # Calculate number of test samples
    if test_dataset is not None:
        num_samples = len(test_dataset)
    else:
        num_samples = len(test_loader.dataset)
    
    print(f"\n{'='*50}")
    print(f"Starting evaluation:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Threshold: {threshold}")
    print(f"  Test samples: {num_samples}")
    print(f"{'='*50}\n")
    
    # Get predictions
    probs, labels = predict(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(probs, labels, threshold)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nMicro-averaged metrics (Threshold {threshold}):")
    print(f"  F1 Score:  {metrics['micro_f1']:.4f}")
    print(f"  Precision: {metrics['micro_precision']:.4f}")
    print(f"  Recall:    {metrics['micro_recall']:.4f}")
    
    print(f"\nRanking-based metrics:")
    print(f"  mAP (Mean Avg Precision): {metrics['mean_average_precision']:.4f}")
    print(f"  Recall@5:    {metrics['recall_at_5']:.4f}")
    print(f"  Precision@5: {metrics['precision_at_5']:.4f}")
    print(f"  Recall@10:   {metrics['recall_at_10']:.4f}")
    print(f"  Precision@10:{metrics['precision_at_10']:.4f}")
    
    print(f"\nExample-based metrics:")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  Accuracy:     {metrics['example_based_accuracy']:.4f}")
    print("="*50 + "\n")
    
    # Add predictions and labels to metrics
    metrics['predictions'] = probs # Storing probs as predictions
    metrics['labels'] = labels
    
    return metrics


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Use this module through run_strategy_test.ipynb for complete pipeline.")
