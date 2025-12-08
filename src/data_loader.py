"""
Data loading and preprocessing module for CTI-BERT TTP tagging.
Handles loading and combining multiple MITRE ATT&CK datasets.

Hybrid Dataset Mode (Default):
- Combines tumeteor/Security-TTP-Mapping (14.9k samples)
- sarahwei/cyber_MITRE_attack_tactics-and-techniques (654 samples)
- Zainabsa99/mitre_attack (508 samples)
- Total: ~16k samples with improved label distribution
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, Tuple, List
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


def load_hybrid_dataset(test_split_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine multiple MITRE ATT&CK datasets from Hugging Face.
    
    Combines:
    - tumeteor/Security-TTP-Mapping (14.9k samples)
    - sarahwei/cyber_MITRE_attack_tactics-and-techniques (654 samples)
    - Zainabsa99/mitre_attack (508 samples)
    
    Args:
        test_split_ratio: Ratio of data to use for testing (for datasets without splits)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    datasets_config = [
        {
            'name': 'tumeteor/Security-TTP-Mapping',
            'text_column': 'text1',
            'label_column': 'labels',
            'use_validation': True
        },
        {
            'name': 'sarahwei/cyber_MITRE_attack_tactics-and-techniques',
            'text_column': 'text',
            'label_column': 'techniques',
            'use_validation': False
        },
        {
            'name': 'Zainabsa99/mitre_attack',
            'text_column': 'description',
            'label_column': 'technique_id',
            'use_validation': False
        }
    ]
    
    all_train_dfs = []
    all_test_dfs = []
    
    print(f"\n{'='*70}")
    print("🔥 HYBRID DATASET MODE - Combining Multiple Sources")
    print(f"{'='*70}\n")
    
    for config in datasets_config:
        dataset_name = config['name']
        print(f"📦 Loading: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name)
            
            # Handle different split configurations
            if config.get('use_validation', False) and 'validation' in dataset.keys():
                train_df = pd.DataFrame(dataset['train'])
                test_df = pd.DataFrame(dataset['validation'])
                print(f"   ✅ Using train + validation splits")
            elif 'train' in dataset.keys() and 'test' in dataset.keys():
                train_df = pd.DataFrame(dataset['train'])
                test_df = pd.DataFrame(dataset['test'])
                print(f"   ✅ Using train + test splits")
            elif 'train' in dataset.keys():
                full_df = pd.DataFrame(dataset['train'])
                split_idx = int(len(full_df) * (1 - test_split_ratio))
                train_df = full_df.iloc[:split_idx].reset_index(drop=True)
                test_df = full_df.iloc[split_idx:].reset_index(drop=True)
                print(f"   ✅ Split into {len(train_df)} train / {len(test_df)} test")
            else:
                print(f"   ⚠️  Unexpected structure, skipping")
                continue
            
            # Standardize column names
            text_col = config['text_column']
            label_col = config['label_column']
            
            if text_col in train_df.columns and label_col in train_df.columns:
                train_df = train_df[[text_col, label_col]].rename(
                    columns={text_col: 'text', label_col: 'labels'}
                )
                test_df = test_df[[text_col, label_col]].rename(
                    columns={text_col: 'text', label_col: 'labels'}
                )
                
                all_train_dfs.append(train_df)
                all_test_dfs.append(test_df)
                print(f"   ✅ Added {len(train_df)} train + {len(test_df)} test samples\n")
            else:
                print(f"   ⚠️  Missing columns, skipping\n")
                
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            continue
    
    if not all_train_dfs:
        raise ValueError("No datasets were successfully loaded!")
    
    # Combine all datasets
    print(f"{'='*70}")
    print("📊 Combining Datasets...")
    print(f"{'='*70}")
    
    combined_train = pd.concat(all_train_dfs, ignore_index=True)
    combined_test = pd.concat(all_test_dfs, ignore_index=True)
    
    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_test = combined_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Hybrid Dataset Created:")
    print(f"   Total train samples: {len(combined_train):,}")
    print(f"   Total test samples: {len(combined_test):,}")
    print(f"{'='*70}\n")
    
    return combined_train, combined_test


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
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    use_hybrid: bool = True,
    dataset_name: str = "tumeteor/Security-TTP-Mapping"
) -> Dict:
    """
    Complete data preparation pipeline.
    
    Args:
        model_name: Pretrained model name for tokenizer
        max_length: Maximum sequence length
        use_hybrid: If True, use hybrid dataset (tumeteor + sarahwei + Zainabsa99)
                   If False, use single dataset specified by dataset_name
        dataset_name: Single dataset name (only used if use_hybrid=False)
        
    Returns:
        Dictionary containing datasets, tokenizer, and label information
    """
    # Load dataset
    if use_hybrid:
        train_df, test_df = load_hybrid_dataset()
    else:
        print(f"📦 Single Dataset Mode: {dataset_name}\n")
        train_df, test_df = load_ttp_dataset(dataset_name=dataset_name)
        # Standardize column names for single dataset
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
    
    # Tokenize texts
    print("⚙️  Tokenizing texts...")
    train_encodings = tokenizer(
        train_df['text'].fillna('').tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    test_encodings = tokenizer(
        test_df['text'].fillna('').tolist(),
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


if __name__ == "__main__":
    # Test hybrid dataset loading
    print("="*70)
    print("Testing HYBRID dataset loading...")
    print("="*70)
    
    data = prepare_data(use_hybrid=True, max_length=128)
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Train samples: {len(data['train_dataset']):,}")
    print(f"Test samples: {len(data['test_dataset']):,}")
    print(f"Number of MITRE techniques: {data['num_labels']}")
    print(f"Sample techniques: {data['label_list'][:10]}")
    print(f"{'='*70}")
