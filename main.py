"""
Main pipeline script for CTI-BERT TTP tagging project.
Can be run locally or in Google Colab.

Usage:
    python main.py --model bert-base-uncased --epochs 3 --batch_size 16
"""

import argparse
import torch
import json
import os
from datetime import datetime

from src.data_loader import prepare_data
from src.model import load_model
from src.train import train_model
from src.evaluate import evaluate_model


def main(args):
    """Main pipeline execution."""
    
    # Print configuration
    print("\n" + "="*70)
    print("CTI-BERT TTP TAGGING - TRAINING PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length}")
    print(f"  Device: {args.device}")
    print("="*70 + "\n")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model.replace('/', '_')}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70 + "\n")
    
    data = prepare_data(
        model_name=args.model,
        max_length=args.max_length
    )
    
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    label_list = data['label_list']
    num_labels = data['num_labels']
    
    # Save label list
    label_file = os.path.join(output_dir, "labels.json")
    with open(label_file, 'w') as f:
        json.dump(label_list, f, indent=2)
    print(f"\nLabel list saved to: {label_file}")
    
    # Step 2: Load model
    print("\n" + "="*70)
    print("STEP 2: MODEL INITIALIZATION")
    print("="*70 + "\n")
    
    model = load_model(
        model_name=args.model,
        num_labels=num_labels,
        device=args.device
    )
    
    # Step 3: Train model
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70 + "\n")
    
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        device=args.device
    )
    
    # Save training history
    history_file = os.path.join(output_dir, "training_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_file}")
    
    # Step 4: Evaluate model
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70 + "\n")
    
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
        label_list=label_list
    )
    
    # Save metrics
    metrics_to_save = {k: float(v) if isinstance(v, (float, int)) else v 
                       for k, v in metrics.items() 
                       if k not in ['predictions', 'labels']}
    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_file}")
    
    # Save final summary
    summary = {
        'model': args.model,
        'timestamp': timestamp,
        'configuration': vars(args),
        'data': {
            'num_labels': num_labels,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        },
        'training': {
            'final_loss': history['train_loss'][-1]
        },
        'evaluation': metrics_to_save
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey Metrics:")
    print(f"  Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Samples F1:  {metrics['samples_f1']:.4f}")
    print("="*70 + "\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTI-BERT TTP Tagging - Training Pipeline"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name (e.g., bert-base-uncased, jackaduma/SecBERT)"
    )
    
    # Data arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for models and results"
    )
    
    args = parser.parse_args()
    
    # Run main pipeline
    main(args)
