"""
Data loading and preprocessing module for CTI-BERT TTP tagging.
Handles loading the Security-TTP-Mapping dataset.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset
import ast


class CTIDataset(Dataset):
    """Custom Dataset for CTI text and TTP labels."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.labels)


def load_ttp_dataset(
    dataset_name: str = "tumeteor/Security-TTP-Mapping",
    use_validation: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TTP mapping dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        use_validation: If True, use validation split as test set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name)
    print(f"Available splits: {list(dataset.keys())}")
    
    train_df = pd.DataFrame(dataset['train'])
    
    if use_validation and 'validation' in dataset.keys():
        test_df = pd.DataFrame(dataset['validation'])
        print(f"Using validation split as test set")
    else:
        test_df = pd.DataFrame(dataset['test'])
        print(f"Using test split")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df


def get_label_list(train_df: pd.DataFrame, label_column: str = 'labels') -> list:
    """
    Extract unique labels from the dataset.
    
    Args:
        train_df: Training DataFrame
        label_column: Name of the column containing labels
        
    Returns:
        List of unique labels
    """
    all_labels = set()
    for labels in train_df[label_column]:
        if isinstance(labels, str):
            try:
                label_list = ast.literal_eval(labels)
                if isinstance(label_list, list):
                    all_labels.update(label_list)
                else:
                    all_labels.add(label_list)
            except:
                label_list = [l.strip() for l in labels.split(',') if l.strip()]
                all_labels.update(label_list)
        elif isinstance(labels, list):
            all_labels.update(labels)
    
    label_list = sorted(list(all_labels))
    print(f"Total unique labels (MITRE ATT&CK Techniques): {len(label_list)}")
    
    return label_list


def encode_labels(df: pd.DataFrame, label_list: list, label_column: str = 'labels') -> list:
    """
    Convert multi-label strings to binary vectors.
    
    Args:
        df: DataFrame with labels
        label_list: List of all possible labels
        label_column: Name of the column containing labels
        
    Returns:
        List of binary label vectors
    """
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    
    encoded_labels = []
    for labels in df[label_column]:
        label_vector = [0] * len(label_list)
        
        if isinstance(labels, str):
            try:
                label_items = ast.literal_eval(labels)
                if not isinstance(label_items, list):
                    label_items = [label_items]
            except:
                label_items = [l.strip() for l in labels.split(',') if l.strip()]
            
            for label in label_items:
                if label in label_to_id:
                    label_vector[label_to_id[label]] = 1
        elif isinstance(labels, list):
            for label in labels:
                if label in label_to_id:
                    label_vector[label_to_id[label]] = 1
        
        encoded_labels.append(label_vector)
    
    return encoded_labels


def prepare_data(
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    text_column: str = 'text1',
    label_column: str = 'labels',
    dataset_name: str = "tumeteor/Security-TTP-Mapping"
) -> Dict:
    """
    Complete data preparation pipeline.
    
    Args:
        model_name: Pretrained model name for tokenizer
        max_length: Maximum sequence length
        text_column: Name of the column containing text
        label_column: Name of the column containing labels
        dataset_name: Dataset name to load
        
    Returns:
        Dictionary containing datasets, tokenizer, and label information
    """
    # Load dataset
    train_df, test_df = load_ttp_dataset(dataset_name=dataset_name)
    
    # Get label list
    label_list = get_label_list(train_df, label_column)
    
    # Encode labels
    train_labels = encode_labels(train_df, label_list, label_column)
    test_labels = encode_labels(test_df, label_list, label_column)
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize texts
    print("Tokenizing texts...")
    train_encodings = tokenizer(
        train_df[text_column].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    test_encodings = tokenizer(
        test_df[text_column].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    # Create datasets
    train_dataset = CTIDataset(train_encodings, train_labels)
    test_dataset = CTIDataset(test_encodings, test_labels)
    
    print("Data preparation complete!")
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'tokenizer': tokenizer,
        'label_list': label_list,
        'num_labels': len(label_list)
    }


if __name__ == "__main__":
    # Test the data loading
    data = prepare_data()
    print(f"\nDataset prepared successfully!")
    print(f"Number of labels: {data['num_labels']}")
    print(f"Train dataset size: {len(data['train_dataset'])}")
    print(f"Test dataset size: {len(data['test_dataset'])}")
    print(f"\nFirst 10 techniques: {data['label_list'][:10]}")
