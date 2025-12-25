"""
Training and evaluation functions for AttentionXML and LightXML models.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_attention_xml(model, train_dataloader, num_epochs, learning_rate, device, 
                       use_focal_loss=False, pos_weight=None, focal_alpha=0.25, focal_gamma=2.0):
    """
    Train AttentionXML model.
    
    Args:
        model: AttentionXML model
        train_dataloader: Training data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        use_focal_loss: Use focal loss instead of BCE
        pos_weight: Positive class weights for BCE
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
    
    Returns:
        training_history: Dictionary with training metrics
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if use_focal_loss:
        from src.model import FocalLoss
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        if pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    
    training_history = {
        'train_loss': [],
        'timestamps': []
    }
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_dataloader)
        training_history['train_loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    return training_history


def evaluate_attention_xml(model, test_dataloader, device, label_names):
    """
    Evaluate AttentionXML model.
    
    Args:
        model: Trained AttentionXML model
        test_dataloader: Test data loader
        device: 'cuda' or 'cpu'
        label_names: List of label names
    
    Returns:
        test_results: Dictionary with evaluation metrics
    """
    from src.evaluate import calculate_multi_label_metrics
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    predictions = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    test_results = calculate_multi_label_metrics(labels, predictions, label_names=label_names)
    
    return test_results


def train_light_xml(model, train_dataloader, num_epochs, learning_rate, device,
                   use_focal_loss=False, pos_weight=None, focal_alpha=0.25, focal_gamma=2.0):
    """
    Train LightXML model.
    
    Args:
        model: LightXML model
        train_dataloader: Training data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        use_focal_loss: Use focal loss instead of BCE
        pos_weight: Positive class weights for BCE (not used in LightXML)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
    
    Returns:
        training_history: Dictionary with training metrics
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    training_history = {
        'train_loss': [],
        'timestamps': []
    }
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # LightXML returns (group_logits, candidate_logits, loss) during training
            # The loss is already computed inside the model
            _, _, loss = model(input_ids, attention_mask, labels=labels)
            
            # Optionally modify loss based on configuration
            # Note: LightXML has built-in BCE loss, so focal loss would require model modification
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_dataloader)
        training_history['train_loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    return training_history


def evaluate_light_xml(model, test_dataloader, device, label_names):
    """
    Evaluate LightXML model.
    
    Args:
        model: Trained LightXML model
        test_dataloader: Test data loader
        device: 'cuda' or 'cpu'
        label_names: List of label names
    
    Returns:
        test_results: Dictionary with evaluation metrics
    """
    from src.evaluate import calculate_multi_label_metrics
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # During inference, LightXML returns logits for all labels
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    predictions = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    test_results = calculate_multi_label_metrics(labels, predictions, label_names=label_names)
    
    return test_results
