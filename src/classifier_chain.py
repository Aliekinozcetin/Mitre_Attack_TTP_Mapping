"""
Multi-label classification with sklearn methods.
Combines BERT embeddings with ClassifierChain and MultiOutputClassifier.
"""

import torch
import numpy as np
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tqdm import tqdm
from typing import Dict, Tuple, Optional


class BERTClassifierChain:
    """
    Combines BERT for feature extraction with ClassifierChain for multi-label prediction.
    
    Strategy:
    1. Use BERT to extract embeddings from text
    2. Use ClassifierChain on top of embeddings to predict labels
    3. Chain captures label dependencies (each classifier uses previous predictions)
    """
    
    def __init__(
        self,
        bert_model,
        device: str = 'cuda',
        base_estimator: str = 'logistic',
        order: str = 'random',
        cv: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Args:
            bert_model: Pre-trained BERT model for feature extraction
            device: Device to run BERT on
            base_estimator: Base classifier ('logistic' or 'random_forest')
            order: Chain order ('random', None for natural order, or list of indices)
            cv: Cross-validation folds for chain (None = use true labels)
            random_state: Random seed
        """
        self.bert_model = bert_model
        self.device = device
        self.random_state = random_state
        
        # Select base estimator
        if base_estimator == 'logistic':
            self.base_clf = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver='lbfgs',
                class_weight='balanced'  # Handle imbalance in each binary problem
            )
        elif base_estimator == 'random_forest':
            self.base_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif base_estimator == 'extra_trees':
            self.base_clf = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1,
                bootstrap=False  # ExtraTrees default
            )
        else:
            raise ValueError(f"Unknown base_estimator: {base_estimator}")
        
        # Create ClassifierChain
        self.chain = ClassifierChain(
            estimator=self.base_clf,
            order=order,
            cv=cv,
            random_state=random_state,
            verbose=True
        )
        
        self.is_fitted = False
    
    def _extract_embeddings(self, dataloader, desc: str = "Extracting embeddings") -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract BERT embeddings from dataloader.
        
        Returns:
            Tuple of (embeddings, labels)
        """
        self.bert_model.eval()
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Get BERT outputs (use [CLS] token embedding)
                outputs = self.bert_model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use pooled output (CLS token representation)
                embeddings = outputs.pooler_output  # Shape: (batch_size, hidden_size)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
        
        embeddings = np.vstack(all_embeddings)
        labels = np.vstack(all_labels)
        
        return embeddings, labels
    
    def fit(self, train_dataloader):
        """
        Fit the classifier chain on training data.
        
        Args:
            train_dataloader: DataLoader with training data
        """
        print("\n" + "="*60)
        print("🔗 TRAINING CLASSIFIER CHAIN")
        print("="*60)
        
        # Extract embeddings
        print("\n📊 Step 1: Extracting BERT embeddings from training data...")
        X_train, Y_train = self._extract_embeddings(train_dataloader, desc="Training embeddings")
        
        print(f"   Embeddings shape: {X_train.shape}")
        print(f"   Labels shape: {Y_train.shape}")
        
        # Fit chain
        print("\n🔗 Step 2: Fitting Classifier Chain...")
        print(f"   Base estimator: {type(self.base_clf).__name__}")
        print(f"   Chain order: {self.chain.order}")
        print(f"   CV folds: {self.chain.cv}")
        
        self.chain.fit(X_train, Y_train)
        
        self.is_fitted = True
        print("\n✅ Classifier Chain training complete!")
        print("="*60)
        
        return self
    
    def predict(self, test_dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for test data.
        
        Args:
            test_dataloader: DataLoader with test data
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction!")
        
        print("\n📊 Extracting embeddings for prediction...")
        X_test, Y_test = self._extract_embeddings(test_dataloader, desc="Test embeddings")
        
        print("\n🔮 Predicting with Classifier Chain...")
        Y_pred = self.chain.predict(X_test)
        
        return Y_pred, Y_test
    
    def predict_proba(self, test_dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict label probabilities for test data.
        
        Args:
            test_dataloader: DataLoader with test data
            
        Returns:
            Tuple of (probabilities, true_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction!")
        
        print("\n📊 Extracting embeddings for prediction...")
        X_test, Y_test = self._extract_embeddings(test_dataloader, desc="Test embeddings")
        
        print("\n🔮 Predicting probabilities with Classifier Chain...")
        Y_proba = self.chain.predict_proba(X_test)
        
        return Y_proba, Y_test


def train_classifier_chain(
    bert_model,
    train_dataloader,
    device: str = 'cuda',
    base_estimator: str = 'logistic',
    order: str = 'random',
    cv: Optional[int] = None,
    random_state: int = 42
) -> BERTClassifierChain:
    """
    Train a BERT + ClassifierChain model.
    
    Args:
        bert_model: Pre-trained BERT model
        train_dataloader: Training data
        device: Device to run on
        base_estimator: Base classifier type ('logistic' or 'random_forest')
        order: Chain order ('random' or None)
        cv: Cross-validation folds (None = use true labels)
        random_state: Random seed
        
    Returns:
        Fitted BERTClassifierChain model
    """
    model = BERTClassifierChain(
        bert_model=bert_model,
        device=device,
        base_estimator=base_estimator,
        order=order,
        cv=cv,
        random_state=random_state
    )
    
    model.fit(train_dataloader)
    
    return model


def evaluate_classifier_chain(
    model: BERTClassifierChain,
    test_dataloader,
    label_names: list
) -> Dict:
    """
    Evaluate ClassifierChain model and compute metrics.
    
    Args:
        model: Fitted BERTClassifierChain model
        test_dataloader: Test data
        label_names: List of label names
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        hamming_loss
    )
    
    # Get predictions and probabilities
    Y_proba, Y_test = model.predict_proba(test_dataloader)
    Y_pred = (Y_proba > 0.5).astype(int)
    
    print("\n" + "="*60)
    print("📊 CLASSIFIER CHAIN EVALUATION")
    print("="*60)
    
    # Micro-averaged metrics
    micro_f1 = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    micro_precision = precision_score(Y_test, Y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(Y_test, Y_pred, average='micro', zero_division=0)
    
    # Hamming Loss
    hamming = hamming_loss(Y_test, Y_pred)
    
    # Calculate @K metrics (using probabilities)
    def calculate_at_k(probs: np.ndarray, labels: np.ndarray, k: int = 5):
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        
        precisions = []
        recalls = []
        
        for i in range(len(labels)):
            true_labels = np.where(labels[i] == 1)[0]
            if len(true_labels) == 0:
                continue
            
            pred_labels = top_k_indices[i]
            hits = len(set(true_labels) & set(pred_labels))
            
            precisions.append(hits / k)
            recalls.append(hits / len(true_labels))
        
        return np.mean(precisions), np.mean(recalls)
    
    p5, r5 = calculate_at_k(Y_proba, Y_test, k=5)
    p10, r10 = calculate_at_k(Y_proba, Y_test, k=10)
    
    # Example-based accuracy
    example_accuracy = np.mean(np.all(Y_pred == Y_test, axis=1))
    
    metrics = {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'hamming_loss': hamming,
        'precision_at_5': p5,
        'recall_at_5': r5,
        'precision_at_10': p10,
        'recall_at_10': r10,
        'example_based_accuracy': example_accuracy,
        'predictions': Y_proba,
        'labels': Y_test
    }
    
    # Print results
    print(f"\nMicro-averaged metrics:")
    print(f"  F1 Score:  {micro_f1:.4f}")
    print(f"  Precision: {micro_precision:.4f}")
    print(f"  Recall:    {micro_recall:.4f}")
    
    print(f"\nRanking-based metrics:")
    print(f"  Recall@5:    {r5:.4f}")
    print(f"  Precision@5: {p5:.4f}")
    print(f"  Recall@10:   {r10:.4f}")
    print(f"  Precision@10: {p10:.4f}")
    
    print(f"\nExample-based metrics:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Accuracy:     {example_accuracy:.4f}")
    print("="*60 + "\n")
    
    return metrics


class BERTMultiOutputClassifier:
    """
    Combines BERT for feature extraction with MultiOutputClassifier for multi-label prediction.
    
    Strategy:
    1. Use BERT to extract embeddings from text
    2. Use MultiOutputClassifier on top of embeddings (independent binary classifiers)
    3. Each label trained independently (no chain, faster training)
    """
    
    def __init__(
        self,
        bert_model,
        device: str = 'cuda',
        base_estimator: str = 'logistic',
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Args:
            bert_model: Pre-trained BERT model for feature extraction
            device: Device to run BERT on
            base_estimator: Base classifier ('logistic' or 'random_forest')
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed
        """
        self.bert_model = bert_model
        self.device = device
        self.random_state = random_state
        
        # Select base estimator
        if base_estimator == 'logistic':
            self.base_clf = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver='lbfgs',
                class_weight='balanced'
            )
        elif base_estimator == 'random_forest':
            self.base_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif base_estimator == 'extra_trees':
            self.base_clf = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1,
                bootstrap=False  # ExtraTrees default
            )
        else:
            raise ValueError(f"Unknown base_estimator: {base_estimator}")
        
        # Create MultiOutputClassifier
        self.multi_output = MultiOutputClassifier(
            estimator=self.base_clf,
            n_jobs=n_jobs
        )
        
        self.is_fitted = False
    
    def _extract_embeddings(self, dataloader, desc: str = "Extracting embeddings") -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract BERT embeddings from dataloader.
        
        Returns:
            Tuple of (embeddings, labels)
        """
        self.bert_model.eval()
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Get BERT outputs (use [CLS] token embedding)
                outputs = self.bert_model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use pooled output (CLS token representation)
                embeddings = outputs.pooler_output
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
        
        embeddings = np.vstack(all_embeddings)
        labels = np.vstack(all_labels)
        
        return embeddings, labels
    
    def fit(self, train_dataloader):
        """
        Fit the multi-output classifier on training data.
        
        Args:
            train_dataloader: DataLoader with training data
        """
        print("\n" + "="*60)
        print("🔀 TRAINING MULTI-OUTPUT CLASSIFIER")
        print("="*60)
        
        # Extract embeddings
        print("\n📊 Step 1: Extracting BERT embeddings from training data...")
        X_train, Y_train = self._extract_embeddings(train_dataloader, desc="Training embeddings")
        
        print(f"   Embeddings shape: {X_train.shape}")
        print(f"   Labels shape: {Y_train.shape}")
        
        # Fit multi-output
        print("\n🔀 Step 2: Fitting Multi-Output Classifier...")
        print(f"   Base estimator: {type(self.base_clf).__name__}")
        print(f"   Parallel jobs: {self.multi_output.n_jobs}")
        print(f"   Training: {Y_train.shape[1]} independent binary classifiers")
        
        self.multi_output.fit(X_train, Y_train)
        
        self.is_fitted = True
        print("\n✅ Multi-Output Classifier training complete!")
        print("="*60)
        
        return self
    
    def predict(self, test_dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for test data.
        
        Args:
            test_dataloader: DataLoader with test data
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction!")
        
        print("\n📊 Extracting embeddings for prediction...")
        X_test, Y_test = self._extract_embeddings(test_dataloader, desc="Test embeddings")
        
        print("\n🔮 Predicting with Multi-Output Classifier...")
        Y_pred = self.multi_output.predict(X_test)
        
        return Y_pred, Y_test
    
    def predict_proba(self, test_dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict label probabilities for test data.
        
        Args:
            test_dataloader: DataLoader with test data
            
        Returns:
            Tuple of (probabilities, true_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction!")
        
        print("\n📊 Extracting embeddings for prediction...")
        X_test, Y_test = self._extract_embeddings(test_dataloader, desc="Test embeddings")
        
        print("\n🔮 Predicting probabilities with Multi-Output Classifier...")
        # predict_proba returns list of arrays, we need to stack them
        Y_proba_list = self.multi_output.predict_proba(X_test)
        
        # Convert list of probabilities to 2D array
        # Each element is (n_samples, 2) for binary, we want probability of class 1
        Y_proba = np.column_stack([proba[:, 1] for proba in Y_proba_list])
        
        return Y_proba, Y_test


def train_multi_output_classifier(
    bert_model,
    train_dataloader,
    device: str = 'cuda',
    base_estimator: str = 'logistic',
    n_jobs: int = -1,
    random_state: int = 42
) -> BERTMultiOutputClassifier:
    """
    Train a BERT + MultiOutputClassifier model.
    
    Args:
        bert_model: Pre-trained BERT model
        train_dataloader: Training data
        device: Device to run on
        base_estimator: Base classifier type ('logistic' or 'random_forest')
        n_jobs: Number of parallel jobs (-1 = all cores)
        random_state: Random seed
        
    Returns:
        Fitted BERTMultiOutputClassifier model
    """
    model = BERTMultiOutputClassifier(
        bert_model=bert_model,
        device=device,
        base_estimator=base_estimator,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    model.fit(train_dataloader)
    
    return model


def evaluate_multi_output_classifier(
    model: BERTMultiOutputClassifier,
    test_dataloader,
    label_names: list
) -> Dict:
    """
    Evaluate MultiOutputClassifier model and compute metrics.
    
    Args:
        model: Fitted BERTMultiOutputClassifier model
        test_dataloader: Test data
        label_names: List of label names
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        hamming_loss
    )
    
    # Get predictions and probabilities
    Y_proba, Y_test = model.predict_proba(test_dataloader)
    Y_pred = (Y_proba > 0.5).astype(int)
    
    print("\n" + "="*60)
    print("📊 MULTI-OUTPUT CLASSIFIER EVALUATION")
    print("="*60)
    
    # Micro-averaged metrics
    micro_f1 = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    micro_precision = precision_score(Y_test, Y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(Y_test, Y_pred, average='micro', zero_division=0)
    
    # Hamming Loss
    hamming = hamming_loss(Y_test, Y_pred)
    
    # Calculate @K metrics (using probabilities)
    def calculate_at_k(probs: np.ndarray, labels: np.ndarray, k: int = 5):
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        
        precisions = []
        recalls = []
        
        for i in range(len(labels)):
            true_labels = np.where(labels[i] == 1)[0]
            if len(true_labels) == 0:
                continue
            
            pred_labels = top_k_indices[i]
            hits = len(set(true_labels) & set(pred_labels))
            
            precisions.append(hits / k)
            recalls.append(hits / len(true_labels))
        
        return np.mean(precisions), np.mean(recalls)
    
    p5, r5 = calculate_at_k(Y_proba, Y_test, k=5)
    p10, r10 = calculate_at_k(Y_proba, Y_test, k=10)
    
    # Example-based accuracy
    example_accuracy = np.mean(np.all(Y_pred == Y_test, axis=1))
    
    metrics = {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'hamming_loss': hamming,
        'precision_at_5': p5,
        'recall_at_5': r5,
        'precision_at_10': p10,
        'recall_at_10': r10,
        'example_based_accuracy': example_accuracy,
        'predictions': Y_proba,
        'labels': Y_test
    }
    
    # Print results
    print(f"\nMicro-averaged metrics:")
    print(f"  F1 Score:  {micro_f1:.4f}")
    print(f"  Precision: {micro_precision:.4f}")
    print(f"  Recall:    {micro_recall:.4f}")
    
    print(f"\nRanking-based metrics:")
    print(f"  Recall@5:    {r5:.4f}")
    print(f"  Precision@5: {p5:.4f}")
    print(f"  Recall@10:   {r10:.4f}")
    print(f"  Precision@10: {p10:.4f}")
    
    print(f"\nExample-based metrics:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Accuracy:     {example_accuracy:.4f}")
    print("="*60 + "\n")
    
    return metrics
