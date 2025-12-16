"""
Text augmentation module for CTI reports.
Implements IoC replacement and back-translation for tail TTP augmentation.
"""

import re
import random
from typing import List, Optional
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore')


def replace_iocs(text: str, seed: Optional[int] = None) -> str:
    """
    Replace Indicators of Compromise (IoCs) with random but valid alternatives.
    
    This prevents the model from overfitting to specific IoCs and focuses on
    behavioral patterns instead.
    
    Args:
        text: Input CTI text
        seed: Random seed for reproducibility
        
    Returns:
        Text with replaced IoCs
    """
    if seed is not None:
        random.seed(seed)
    
    augmented = text
    
    # IP addresses (IPv4)
    # Pattern: 192.168.1.1
    def random_ip(match):
        return f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
    
    augmented = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', random_ip, augmented)
    
    # Windows file paths
    # Pattern: C:\Users\Alice\file.dll
    def random_windows_user(match):
        users = ['Admin', 'User', 'Alice', 'Bob', 'System', 'Guest', 'Default']
        return f"C:\\Users\\{random.choice(users)}\\"
    
    augmented = re.sub(r'C:\\Users\\(\w+)\\', random_windows_user, augmented)
    
    # Generic Windows paths
    # Pattern: C:\Windows\System32\file.exe
    def random_windows_path(match):
        paths = ['Windows', 'Program Files', 'ProgramData', 'Temp']
        return f"C:\\{random.choice(paths)}\\"
    
    augmented = re.sub(r'C:\\(Windows|Program Files|ProgramData|Temp)\\', random_windows_path, augmented)
    
    # Domains
    # Pattern: example.com, malicious.net
    def random_domain(match):
        prefixes = ['update', 'news', 'mail', 'data', 'service', 'cloud', 'api', 'web']
        return f"{random.choice(prefixes)}.{match.group(2)}"
    
    augmented = re.sub(r'\b([a-z0-9-]+)\.(com|net|org|io|info)\b', random_domain, augmented)
    
    # MD5 hashes (32 hex characters)
    def random_md5(match):
        return ''.join(random.choices('0123456789abcdef', k=32))
    
    augmented = re.sub(r'\b[a-f0-9]{32}\b', random_md5, augmented)
    
    # SHA1 hashes (40 hex characters)
    def random_sha1(match):
        return ''.join(random.choices('0123456789abcdef', k=40))
    
    augmented = re.sub(r'\b[a-f0-9]{40}\b', random_sha1, augmented)
    
    # SHA256 hashes (64 hex characters)
    def random_sha256(match):
        return ''.join(random.choices('0123456789abcdef', k=64))
    
    augmented = re.sub(r'\b[a-f0-9]{64}\b', random_sha256, augmented)
    
    # Registry keys (Windows)
    # Pattern: HKLM\SOFTWARE\..., HKCU\...
    def random_registry_hive(match):
        hives = ['HKLM', 'HKCU', 'HKCR', 'HKU']
        return random.choice(hives)
    
    augmented = re.sub(r'\b(HKLM|HKCU|HKCR|HKU)\b', random_registry_hive, augmented)
    
    # Email addresses (common in phishing scenarios)
    # Pattern: user@example.com
    def random_email(match):
        users = ['admin', 'support', 'info', 'contact', 'noreply', 'service']
        domains = ['example.com', 'mail.com', 'service.net', 'company.org']
        return f"{random.choice(users)}@{random.choice(domains)}"
    
    augmented = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', random_email, augmented)
    
    return augmented


def back_translate(
    text: str,
    pivot_lang: str = 'de',
    max_length: int = 512,
    device: str = 'cpu'
) -> str:
    """
    Perform back-translation for text augmentation.
    
    Translates text to a pivot language and back to English to create
    paraphrased versions while preserving semantic meaning.
    
    Args:
        text: Input text in English
        pivot_lang: Intermediate language for translation (default: German)
        max_length: Maximum sequence length for translation
        device: Device to run translation on ('cpu' or 'cuda')
        
    Returns:
        Back-translated text
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
    except ImportError:
        print("⚠️ transformers not installed. Install with: pip install transformers")
        return text
    
    try:
        # Model names for translation
        model_name_en_pivot = f'Helsinki-NLP/opus-mt-en-{pivot_lang}'
        model_name_pivot_en = f'Helsinki-NLP/opus-mt-{pivot_lang}-en'
        
        # Load models
        tokenizer_en_pivot = MarianTokenizer.from_pretrained(model_name_en_pivot)
        model_en_pivot = MarianMTModel.from_pretrained(model_name_en_pivot).to(device)
        
        tokenizer_pivot_en = MarianTokenizer.from_pretrained(model_name_pivot_en)
        model_pivot_en = MarianMTModel.from_pretrained(model_name_pivot_en).to(device)
        
        # EN → Pivot
        with torch.no_grad():
            inputs = tokenizer_en_pivot(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            pivot_ids = model_en_pivot.generate(**inputs, max_length=max_length)
            pivot_text = tokenizer_en_pivot.decode(pivot_ids[0], skip_special_tokens=True)
        
        # Pivot → EN
        with torch.no_grad():
            inputs = tokenizer_pivot_en(
                pivot_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            en_ids = model_pivot_en.generate(**inputs, max_length=max_length)
            back_translated = tokenizer_pivot_en.decode(en_ids[0], skip_special_tokens=True)
        
        return back_translated
        
    except Exception as e:
        print(f"⚠️ Back-translation failed: {e}")
        return text


def augment_tail_samples(
    texts: List[str],
    labels: List[List[int]],
    label_counts: dict,
    tail_threshold: int = 20,
    ioc_replacement: bool = True,
    back_translation: bool = True,
    oversample_factor: int = 3,
    device: str = 'cpu'
) -> tuple:
    """
    Augment tail TTP samples with multiple strategies.
    
    Args:
        texts: List of input texts
        labels: List of label sets (multi-label)
        label_counts: Dictionary mapping label_idx → count
        tail_threshold: Labels with count < threshold are considered tail
        ioc_replacement: Whether to apply IoC replacement
        back_translation: Whether to apply back-translation
        oversample_factor: How many augmented versions per tail sample
        device: Device for back-translation models
        
    Returns:
        Tuple of (augmented_texts, augmented_labels)
    """
    augmented_texts = list(texts)  # Copy original
    augmented_labels = list(labels)
    
    print(f"\n{'='*60}")
    print(f"🔄 TAIL TTP AUGMENTATION")
    print(f"{'='*60}")
    print(f"Tail threshold: < {tail_threshold} samples")
    print(f"IoC replacement: {ioc_replacement}")
    print(f"Back-translation: {back_translation}")
    print(f"Oversample factor: {oversample_factor}x")
    
    # Find tail labels
    tail_labels = set()
    for label_idx, count in label_counts.items():
        if count < tail_threshold:
            tail_labels.add(label_idx)
    
    print(f"Tail labels found: {len(tail_labels)}")
    
    # Find samples containing tail labels
    tail_sample_indices = []
    for i, label_set in enumerate(labels):
        label_indices = [idx for idx, val in enumerate(label_set) if val == 1]
        if any(idx in tail_labels for idx in label_indices):
            tail_sample_indices.append(i)
    
    print(f"Samples with tail TTPs: {len(tail_sample_indices)}")
    
    # Augment tail samples
    augmented_count = 0
    for idx in tail_sample_indices:
        original_text = texts[idx]
        original_labels = labels[idx]
        
        for _ in range(oversample_factor):
            augmented_text = original_text
            
            # Apply IoC replacement
            if ioc_replacement:
                augmented_text = replace_iocs(augmented_text)
            
            # Apply back-translation
            if back_translation:
                augmented_text = back_translate(augmented_text, device=device)
            
            # Add augmented sample
            augmented_texts.append(augmented_text)
            augmented_labels.append(original_labels)
            augmented_count += 1
    
    print(f"✅ Augmented samples created: {augmented_count}")
    print(f"Total samples: {len(texts)} → {len(augmented_texts)}")
    print(f"{'='*60}\n")
    
    return augmented_texts, augmented_labels


if __name__ == "__main__":
    # Test IoC replacement
    test_text = """
    The attacker used PowerShell to connect to 192.168.1.10 and download malware from
    http://malicious.com/payload.exe. The file was saved to C:\\Users\\Alice\\AppData\\mal.dll
    with MD5 hash 5d41402abc4b2a76b9719d911017c592.
    """
    
    print("Original:")
    print(test_text)
    print("\nIoC Replaced:")
    print(replace_iocs(test_text))
    
    # Test back-translation
    simple_text = "The attacker used PowerShell to execute malicious commands."
    print("\n\nOriginal:")
    print(simple_text)
    print("\nBack-translated:")
    print(back_translate(simple_text))
