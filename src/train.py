"""
Training module for CTI-BERT TTP tagging.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm  # Use tqdm.auto for better Colab support
import os
from typing import Dict


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    # Colab-optimized progress bar with proper formatting
    progress_bar = tqdm(
        dataloader, 
        desc="Training",
        position=0,
        leave=True,
        ncols=120,
        ascii=False,
        colour='blue'
    )
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar with readable format
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(progress_bar.n+1):.4f}'
        })
    
    # Close progress bar properly and print final stats
    progress_bar.close()
    
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch completed - Avg Loss: {avg_loss:.4f}")
    return avg_loss


def train_model(
    model,
    train_dataloader=None,
    train_dataset=None,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    warmup_steps: int = 500,
    device: str = 'cuda',
    output_dir: str = "./models"
) -> Dict:
    """
    Complete training pipeline.
    
    Args:
        model: Model to train
        train_dataloader: Training DataLoader (provide this OR train_dataset)
        train_dataset: Training dataset (will create DataLoader if train_dataloader not provided)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size (only used if train_dataset provided)
        warmup_steps: Number of warmup steps
        device: Device to train on
        output_dir: Directory to save model
        
    Returns:
        Dictionary with training history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loader if not provided
    if train_dataloader is None:
        if train_dataset is None:
            raise ValueError("Must provide either train_dataloader or train_dataset")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        train_loader = train_dataloader
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    model = model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'epoch': []
    }
    
    print(f"\n{'='*50}")
    print(f"Starting training:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"{'='*50}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device
        )
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['epoch'].append(epoch + 1)
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    return history


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Use this module through run_strategy_test.ipynb for complete pipeline.")
