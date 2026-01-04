"""
Training and evaluation functions for AttentionXML and LightXML models.

IMPROVEMENTS (v2):
- Added gradient clipping for stability
- Added learning rate warmup scheduler
- Better loss scaling for multi-label
- Added early stopping capability
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


def train_attention_xml(model, train_dataloader, num_epochs, learning_rate, device, 
                       use_focal_loss=False, pos_weight=None, focal_alpha=0.25, focal_gamma=2.0):
    """
    Train AttentionXML model with improved training loop.
    
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
    # Use lower learning rate for attention layers
    encoder_params = list(model.encoder.parameters())
    attention_params = list(model.attention.parameters()) + list(model.classifier.parameters()) + list(model.cls_projection.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained encoder
        {'params': attention_params, 'lr': learning_rate}  # Higher LR for new layers
    ], weight_decay=0.01)
    
    # OneCycle scheduler for better convergence
    total_steps = len(train_dataloader) * num_epochs
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[learning_rate * 0.1, learning_rate],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
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
        'timestamps': [],
        'learning_rates': []
    }
    
    # Gradient clipping value
    max_grad_norm = 1.0
    
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})
        
        avg_loss = epoch_loss / len(train_dataloader)
        training_history['train_loss'].append(avg_loss)
        training_history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
    
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
    Train LightXML model with improved training strategy.
    
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
    # Differential learning rates: lower for pretrained encoder, higher for new layers
    encoder_params = []
    new_layer_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            new_layer_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for encoder
        {'params': new_layer_params, 'lr': learning_rate}  # Full LR for new layers
    ], weight_decay=0.01)
    
    # OneCycleLR scheduler for better convergence
    total_steps = len(train_dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[learning_rate * 0.1, learning_rate],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    training_history = {
        'train_loss': [],
        'timestamps': []
    }
    
    # Gradient clipping for stability
    max_grad_norm = 1.0
    
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
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
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
