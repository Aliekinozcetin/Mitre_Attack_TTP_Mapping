"""Data loading and preprocessing module for CTI-BERT TTP tagging.

This project uses a single dataset by default:
- tumeteor/Security-TTP-Mapping

Model: ibm-research/CTI-BERT - Domain-specific BERT pre-trained on CTI data
Reference: https://huggingface.co/ibm-research/CTI-BERT
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import re


def normalize_cti_text(text: str) -> str:
    """
    Normalize defanged indicators in CTI text for consistent tokenization.
    
    Converts defanged IOCs (Indicators of Compromise) to standard tokens:
    - IP addresses: 1.2.3[.]4 → <IP_ADDR>
    - URLs: hxxp://evil[.]com → <URL>
    - Domains: evil[.]com → <DOMAIN>
    - File hashes: MD5/SHA256 → <MD5>/<SHA256>
    
    Args:
        text: Raw CTI text with potentially defanged indicators
        
    Returns:
        Normalized text with standardized tokens
    """
    if not isinstance(text, str):
        return text
    
    # IP addresses (defanged): 1.2.3[.]4 or 1[.]2[.]3[.]4
    text = re.sub(
        r'\b\d{1,3}\[\.\]\d{1,3}\[\.\]\d{1,3}\[\.\]\d{1,3}\b',
        '<IP_ADDR>',
        text
    )
    text = re.sub(
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        '<IP_ADDR>',
        text
    )
    
    # URLs (defanged): hxxp://, hxxps://, http[:]//
    text = re.sub(r'hxxps?://[^\s]+', '<URL>', text)
    text = re.sub(r'https?[\[\(]?:[\]\)]?//[^\s]+', '<URL>', text)
    
    # Domains (defanged): evil[.]com, evil(.)com
    text = re.sub(r'\b[\w\-]+\[\.\][\w\-\.]+\b', '<DOMAIN>', text)
    text = re.sub(r'\b[\w\-]+\(\.\)[\w\-\.]+\b', '<DOMAIN>', text)
    
    # Email addresses (defanged): user[@]domain[.]com
    text = re.sub(r'\b[\w\.\-]+\[@\][\w\.\-]+\b', '<EMAIL>', text)
    
    # File hashes
    # MD5: 32 hex characters
    text = re.sub(r'\b[a-fA-F0-9]{32}\b', '<MD5>', text)
    # SHA1: 40 hex characters
    text = re.sub(r'\b[a-fA-F0-9]{40}\b', '<SHA1>', text)
    # SHA256: 64 hex characters
    text = re.sub(r'\b[a-fA-F0-9]{64}\b', '<SHA256>', text)
    
    # File paths
    text = re.sub(r'[A-Z]:\\[^\s]+', '<FILE_PATH>', text)  # Windows
    text = re.sub(r'/[\w/\-\.]+/[\w\-\.]+', '<FILE_PATH>', text)  # Unix/Linux
    
    return text


def sliding_window_tokenize(
    text: str,
    tokenizer,
    max_length: int = 512,
    stride: int = 128,
    aggregate: str = 'max'
):
    """
    Tokenize long texts using sliding window approach.
    
    For texts longer than max_length, creates overlapping windows
    and aggregates predictions. Based on paper's methodology for
    handling long CTI reports.
    
    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 512)
        stride: Overlap between windows (default: 128)
        aggregate: How to combine multi-window predictions ('max', 'mean', 'vote')
    
    Returns:
        Dictionary with tokenization and window metadata
    """
    # Initial tokenization to check length
    tokens = tokenizer.tokenize(text)
    
    # If text fits in single window, use standard tokenization
    if len(tokens) <= max_length - 2:  # -2 for [CLS] and [SEP]
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        encoding['num_windows'] = 1
        encoding['aggregate_method'] = aggregate
        return encoding
    
    # Create sliding windows for long texts
    windows = []
    window_positions = []
    
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_length - 2, len(tokens))
        window_tokens = tokens[start_idx:end_idx]
        
        # Convert tokens back to text
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        
        # Encode window
        window_encoding = tokenizer(
            window_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        windows.append(window_encoding)
        window_positions.append((start_idx, end_idx))
        
        # Move to next window
        if end_idx >= len(tokens):
            break
        start_idx += (max_length - stride - 2)
    
    # For now, return first window (single prediction per sample)
    # Multi-window aggregation would require model architecture changes
    result = windows[0]
    result['num_windows'] = len(windows)
    result['aggregate_method'] = aggregate
    result['window_positions'] = window_positions
    
    return result


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
    Load single TTP mapping dataset from Hugging Face.
    
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
        if pd.isna(labels):
            continue
            
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
    print(f"Total unique MITRE ATT&CK Techniques: {len(label_list)}")
    print(f"First 10 techniques: {label_list[:10]}")
    
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
        
        if pd.isna(labels):
            encoded_labels.append(label_vector)
            continue
        
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
    model_name: str = "ibm-research/CTI-BERT",
    max_length: int = 512,
    dataset_name: str = "tumeteor/Security-TTP-Mapping"
) -> Dict:
    """
    Complete data preparation pipeline.
    
    Args:
        model_name: Pretrained model name for tokenizer
        max_length: Maximum sequence length
        dataset_name: Dataset name on Hugging Face Hub
        
    Returns:
        Dictionary containing datasets, tokenizer, and label information
    """
    # Load dataset (single dataset only)
    print(f"📦 Dataset: {dataset_name}\n")
    train_df, test_df = load_ttp_dataset(dataset_name=dataset_name)
    # Standardize column names for this dataset
    if 'text1' in train_df.columns:
        train_df = train_df.rename(columns={'text1': 'text'})
        test_df = test_df.rename(columns={'text1': 'text'})
    
    # Get label list
    label_list = get_label_list(train_df, 'labels')
    
    # Encode labels
    print("\n📊 Encoding labels...")
    train_labels = encode_labels(train_df, label_list, 'labels')
    test_labels = encode_labels(test_df, label_list, 'labels')
    
    # Initialize tokenizer
    print(f"\n🔤 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Normalize CTI text (defanged indicators)
    print("🔧 Normalizing defanged indicators...")
    train_texts = [normalize_cti_text(text) for text in train_df['text'].fillna('').tolist()]
    test_texts = [normalize_cti_text(text) for text in test_df['text'].fillna('').tolist()]
    
    # Tokenize texts
    print("⚙️  Tokenizing texts...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    # Create datasets
    train_dataset = CTIDataset(train_encodings, train_labels)
    test_dataset = CTIDataset(test_encodings, test_labels)
    
    print("\n✅ Data preparation complete!")
    print(f"   Train dataset: {len(train_dataset):,} samples")
    print(f"   Test dataset: {len(test_dataset):,} samples")
    print(f"   Number of labels: {len(label_list)}")
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'tokenizer': tokenizer,
        'label_list': label_list,
        'num_labels': len(label_list)
    }


def load_datasets_and_prepare_dataloaders(
    model_name: str = "ibm-research/CTI-BERT",
    batch_size: int = 16,
    max_length: int = 512,
    dataset_name: str = "tumeteor/Security-TTP-Mapping"
) -> Tuple:
    """
    Alias for prepare_data that returns dataloaders instead of datasets.
    Compatible with notebook code.
    
    Args:
        model_name: Pretrained model name for tokenizer
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        dataset_name: Dataset name on Hugging Face Hub
        
    Returns:
        Tuple of (train_loader, val_loader, test_dataset, label_encoder)
        Note: val_loader is None (we don't use separate validation)
    """
    data = prepare_data(
        model_name=model_name,
        max_length=max_length,
        dataset_name=dataset_name
    )
    
    # Create train dataloader
    train_loader = DataLoader(
        data['train_dataset'],
        batch_size=batch_size,
        shuffle=True
    )
    
    return (
        train_loader,           # train_loader
        None,                   # val_loader (not used)
        data['test_dataset'],   # test_dataset
        data['label_list']      # label_encoder (label_list)
    )


if __name__ == "__main__":
    # Smoke test (single dataset)
    print("="*70)
    print("Testing dataset loading...")
    print("="*70)
    
    data = prepare_data(max_length=128)
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Train samples: {len(data['train_dataset']):,}")
    print(f"Test samples: {len(data['test_dataset']):,}")
    print(f"Number of MITRE techniques: {data['num_labels']}")
    print(f"Sample techniques: {data['label_list'][:10]}")
    print(f"{'='*70}")
