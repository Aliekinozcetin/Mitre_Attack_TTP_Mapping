# MITRE ATT&CK TTP Mapping with CTI-BERT

Multi-label classification system using **CTI-BERT** (IBM Research) to tag Tactics, Techniques, and Procedures (TTPs) from MITRE ATT&CK framework.

## 🤖 Model

**CTI-BERT** (`ibm-research/CTI-BERT`)
- Domain-specific BERT pre-trained on Cyber Threat Intelligence data
- Superior performance on security-related text understanding
- Optimized for MITRE ATT&CK technique recognition
- **Differential Learning Rate**: BERT encoder (2e-5) + Classifier head (1e-4)
- Reference: https://huggingface.co/ibm-research/CTI-BERT

## 🎯 Dataset

**Security-TTP-Mapping** (`tumeteor/Security-TTP-Mapping`)
- **Train:** 14,936 samples
- **Test:** 2,638 samples  
- **Labels:** 499 MITRE ATT&CK techniques (multi-label)
- **Challenge:** Severe class imbalance (1:458 positive-to-negative ratio)

## 🔬 Advanced CTI Preprocessing

### 1. Defanged Indicator Normalization
Standardizes obfuscated cyber threat indicators before tokenization:

```python
# IP Addresses
"192.168.1[.]1" → "<IP_ADDR>"
"10[.]0[.]0[.]1" → "<IP_ADDR>"

# URLs
"hxxp://evil[.]com/malware" → "<URL>"
"https[:]//phishing.com" → "<URL>"

# Domains
"evil[.]com" → "<DOMAIN>"
"malicious[.]org" → "<DOMAIN>"

# Hashes
"5d41402abc4b2a76b9719d911017c592" → "<MD5>"
"aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d" → "<SHA1>"
"2c26b46b68ffc68ff99b453c1d30413413422d706..." → "<SHA256>"

# Emails
"attacker[@]evil.com" → "<EMAIL>"

# File Paths
"C:\\Windows\\malware.exe" → "<FILE_PATH>"
"/usr/bin/backdoor" → "<FILE_PATH>"
```

**Benefits:**
- Reduces vocabulary noise from indicator variations
- Improves model generalization across different reports
- Preserves semantic meaning while standardizing format

### 2. Sliding Window Tokenization
Handles long CTI reports exceeding 512 token limit:

```python
# Configuration
Max Length: 512 tokens
Stride (Overlap): 128 tokens
Aggregation: First window (extendable to max/mean/vote)

# Example: 1000-token report
Window 1: tokens [0:512]    (used for prediction)
Window 2: tokens [384:896]  (128 overlap with W1)
Window 3: tokens [768:1000] (128 overlap with W2)
```

**Benefits:**
- No information loss from truncation
- Overlapping windows preserve context boundaries
- Extendable to multi-window prediction aggregation

### 3. Differential Learning Rate
Optimizes BERT encoder and classification head separately:

```python
BERT Encoder:
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Strategy: Slow fine-tuning preserves pretrained CTI knowledge

Classification Head:
- Learning Rate: 1e-4 (5× faster)
- Weight Decay: 0.0
- Strategy: Fast convergence on task-specific patterns
```

**Benefits:**
- Preserves CTI-BERT's domain expertise
- Accelerates task-specific layer adaptation
- Prevents catastrophic forgetting of pretrained knowledge

## 📁 Structure

```
├── run_strategy_test.ipynb  # Hybrid strategy testing (4 losses × 4 classifiers)
├── src/
│   ├── data_loader.py       # CTI preprocessing + sliding windows
│   ├── model.py             # CTI-BERT with Focal/Weighted BCE
│   ├── train.py             # Differential LR training loop
│   ├── evaluate.py          # Multi-label metrics
│   └── classifier_chain.py  # Sklearn-based meta-classifiers
├── outputs/                  # Training checkpoints & results
└── requirements.txt
```

## 🚀 Quick Start

### Google Colab (Recommended)

1. Upload `run_training.ipynb` to Colab
2. Set Runtime to **GPU (T4 or better)**
3. Run all cells to test 22 strategies

### Local

```bash
pip install -r requirements.txt
jupyter notebook run_training.ipynb
```

## 🧪 Hybrid Strategy Experiments (22 Strategies)

Testing **4 Loss Functions × 4 Classification Methods + 6 Individual Strategies**:

### Loss Functions
1. **Baseline BCE** - Standard Binary Cross-Entropy
2. **Weighted BCE** - Per-label frequency-based weights (1:458 ratio compensation)
3. **Focal Loss (γ=2)** - Moderate hard example focus
4. **Focal Loss (γ=5)** - Strong hard example focus

### Classification Methods
1. **ClassifierChain** - Models label dependencies sequentially
2. **MultiOutputClassifier** - Independent per-label classifiers
3. **RandomForestClassifier** - Ensemble decision trees
4. **ExtraTreesClassifier** - Randomized ensemble trees

### Individual Strategies (1-6)
- Strategy 1-4: Loss functions with standard classifier
- Strategy 5-6: Sklearn tree ensembles only

### Hybrid Strategies (7-22)
- **Strategy 7-10:** Baseline BCE × 4 classifiers
- **Strategy 11-14:** Weighted BCE × 4 classifiers
- **Strategy 15-18:** Focal γ=2 × 4 classifiers
- **Strategy 19-22:** Focal γ=5 × 4 classifiers

## 📊 Evaluation Metrics

### Core Metrics
- **Micro-F1:** Overall performance (main metric)
- **Macro-F1:** Per-class average (imbalance indicator)
- **Example-Based Accuracy (Subset Accuracy):** Exact match per sample

### Ranking Metrics (SOC Analyst Perspective)
- **mAP (Mean Average Precision):** Measures ranking quality - rewards models that place correct TTPs at the top of the prediction list. Critical for SOC analysts who review top predictions.
- **Recall@5/10:** How many true TTPs appear in top-K predictions
- **Precision@5/10:** What fraction of top-K predictions are correct

### Why These Metrics?
This is effectively a **recommendation system for SOC analysts**:
- **mAP** evaluates the entire ranking (better than Recall@K alone)
- **Recall@5** matters because analysts typically review top 5-10 predictions
- **Subset Accuracy** is strict but shows perfect classification capability

Results saved to `outputs/bert-base-uncased_[timestamp]/`:
- `final_model.pt` - Best checkpoint
- `evaluation_metrics.json` - All metrics (including mAP)
- `training_history.json` - Loss/accuracy curves
- `summary.json` - Configuration + results

## 🔧 Implementation Notes

### Why These Features?
Based on state-of-the-art CTI classification research:
1. **Differential LR**: Prevents CTI-BERT overfitting while enabling fast task adaptation
2. **Defanged Normalization**: Addresses real-world CTI report obfuscation practices
3. **Sliding Windows**: Handles variable-length threat intelligence documents

### Known Limitations
- Sliding windows currently use first window only (single prediction)
- Multi-window aggregation requires architecture changes
- Normalization regex may miss novel obfuscation patterns

---

**Academic use only**
