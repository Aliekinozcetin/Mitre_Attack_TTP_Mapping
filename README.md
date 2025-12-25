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
├── run_strategy_test.ipynb  # Main experiment notebook (24 strategies)
├── src/
│   ├── data_loader.py       # CTI preprocessing + sliding windows
│   ├── model.py             # CTI-BERT with Focal/Weighted BCE
│   ├── train.py             # Differential LR training loop
│   ├── evaluate.py          # Multi-label metrics (including mAP)
│   ├── augmentation.py      # IoC replacement, back-translation, oversampling
│   ├── classifier_chain.py  # Sklearn-based ClassifierChain (clean progress bar)
│   ├── attention_xml.py     # AttentionXML implementation
│   ├── light_xml.py         # LightXML implementation
│   └── xml_utils.py         # Training/eval utilities for XMC methods
├── outputs/                  # Training checkpoints & results
└── requirements.txt
```

## 🚀 Quick Start

### Google Colab (Recommended)

1. Upload `run_strategy_test.ipynb` to Colab
2. Set Runtime to **GPU (T4 or better)**
3. **Follow the recommended execution order below**

### Local

```bash
pip install -r requirements.txt
jupyter notebook run_strategy_test.ipynb
```

## 🔬 Experiment Structure (Recommended Execution Order)

### **PART A: Data Augmentation (Run First!)**
Test augmentation strategies to improve data quality, especially for rare (tail) TTPs:

**5 Strategies (A-1 to A-5):**
1. **AUG-1: Baseline** - No augmentation (reference)
2. **AUG-2: IoC Replacement** - Randomize indicators (IP, hash, domain) to prevent overfitting
3. **AUG-3: Back-translation** - Paraphrase via EN→DE→EN for semantic variety
4. **AUG-4: Oversampling** - Replicate tail TTPs (3x-10x based on frequency)
5. **AUG-5: Combined** - All three methods together

**Expected Improvements:**
- Tail TTP Recall: +40-60%
- mAP (ranking): +20-30%
- Micro F1: +30-50%

**Duration:** ~4-5 hours (5 strategies × 45-60 min each)

---

### **PART B: Loss Function Strategies (Run with Best Augmentation)**
Test different loss functions using the best augmentation from PART A:

#### **Section 1: Loss Functions (4 strategies)**
1. **STR-1: Baseline BCE** - Standard Binary Cross-Entropy
2. **STR-2: Weighted BCE** - Frequency-based pos_weight (compensates 1:458 imbalance)
3. **STR-3: Focal Loss (γ=2)** - Moderate hard example focus
4. **STR-4: Focal Loss (γ=5)** - Strong hard example focus

**Best Performer:** Weighted BCE (frequency-based pos_weight)

**Duration:** ~3-4 hours (4 strategies × 45-60 min each)

#### **Section 2: Capacity Testing (1 strategy, 5 variants)**
5. **STR-5: Top-K Label Analysis** - Test model capacity with different label subset sizes
   - K = 5, 10, 20, 50, 100 labels
   - Each uses baseline BCE
   - Understand learning behavior at different scales

**Duration:** ~2-2.5 hours (5 models × 25-30 min each)

---

### **PART C: Hybrid Strategies (Final Optimization)**
Test comprehensive combinations of loss functions × classification methods:

**10 Strategies = 2 Loss × 5 Methods**

#### **Loss Functions (from Part B best performers):**
1. **Weighted BCE** - Frequency-based weights
2. **Focal Loss (γ=5)** - Strong hard example focusing

#### **Classification Methods:**
1. **ClassifierChain** - Models label dependencies sequentially (BERT embeddings → 499 chained binary classifiers)
2. **ExtraTreesClassifier** - Extremely randomized trees (faster, less overfitting)
3. **RandomForestClassifier** - Ensemble decision trees (optimal splits, higher accuracy)
4. **AttentionXML** - Multi-label attention mechanism (NeurIPS 2019) - each label has its own attention weights
5. **LightXML** - Two-stage dynamic negative sampling (AAAI 2021) - label grouping + candidate ranking

#### **Strategy Matrix:**
```
                    Chain  ExtraTrees  RandomForest  AttentionXML  LightXML
Weighted BCE        C-1    C-2         C-3           C-4           C-5
Focal Loss (γ=5)    C-6    C-7         C-8           C-9           C-10
```

**Duration:** ~7.5-10 hours (10 strategies × 45-60 min each)

**Total Experiments:** 5 + 4 + 5 + 10 = **24 strategies**

---

## 📊 Why This Order?

```
PART A (Augmentation)
   ↓ Select best method (e.g., AUG-5 Combined)
   ↓
PART B Section 1 (Loss Functions)  
   ↓ Use best augmentation, find best loss (e.g., Weighted BCE)
   ↓
PART B Section 2 (Capacity Test)
   ↓ Understand model learning behavior at different scales
   ↓
PART C (Hybrid Strategies)
   ↓ Use best augmentation, test 2 best losses × 5 methods
   ↓
Final Best: AUG-5 + Weighted BCE + [Best Method]
```

**Benefits:**
- **Scientific:** Each stage builds on previous optimization
- **Efficient:** Reuse best augmentation across all strategies
- **Clear:** Easy to identify which component contributed to improvement
- **Flexible:** Each part is independent - can run in any order

**Time Estimates:**
- **Quick Test:** A-1 + STR-2 + C-1 → ~1.5 hours (baseline comparison)
- **Part A Only:** 5 strategies → ~4-5 hours
- **Part B Only:** 4 + 5 strategies → ~5-6 hours
- **Part C Only:** 10 strategies → ~7.5-10 hours
- **Complete Run:** 24 strategies → ~17-21 hours

---

## 📊 Classification Methods Deep Dive

### **Traditional Methods (BERT Embeddings → Sklearn)**

#### **ClassifierChain**
- **Approach:** 499 sequential binary classifiers (LogisticRegression)
- **Feature:** Each classifier uses predictions from previous labels
- **Benefit:** Captures label dependencies (e.g., T1059 → T1059.001 likely)
- **Trade-off:** Longer training, order-dependent
- **Progress:** Clean single progress bar (not 499 lines!)

#### **ExtraTrees vs RandomForest**
- **ExtraTrees:** Random splits, faster, less overfitting, more diverse
- **RandomForest:** Optimal splits, slightly higher accuracy
- **Both:** Ensemble methods with class_weight='balanced'

### **Extreme Multi-Label Classification (XMC) Methods**

#### **AttentionXML** (NeurIPS 2019)
- **Innovation:** Multi-label attention mechanism
- **How:** Each of 499 labels has its own attention query vector
- **Benefit:** Label-specific text regions (different parts of text for different TTPs)
- **Training:** End-to-end BERT fine-tuning with label-specific attention
- **Implementation:** Simplified for 499 labels (original designed for 100K+ labels)

#### **LightXML** (AAAI 2021)
- **Innovation:** Two-stage with dynamic negative sampling
- **Stage 1:** Label grouping - 499 labels → 50 groups (coarse classification)
- **Stage 2:** Candidate ranking - refine within selected groups (fine-grained)
- **Features:** 
  - Label embeddings (128-dim semantic space)
  - Dynamic negative sampling during training
  - Efficient for large label spaces
- **Benefit:** Faster training, better scalability than AttentionXML

---

## 🧪 All Experiments Summary

### **PART A: Augmentation (5 strategies)**
- A-1: Baseline (no augmentation)
- A-2: IoC Replacement only
- A-3: Back-translation only
- A-4: Oversampling only
- A-5: Combined (all 3 methods)

### **PART B Section 1: Loss Functions (4 strategies)**
- STR-1: Baseline BCE
- STR-2: Weighted BCE
- STR-3: Focal Loss (γ=2)
- STR-4: Focal Loss (γ=5)

### **PART B Section 2: Capacity Testing (5 variants)**
- STR-5: Top-K (K = 5, 10, 20, 50, 100)

### **PART C: Hybrid Strategies (10 strategies)**
**Weighted BCE (C-1 to C-5):**
- C-1: Weighted BCE + ClassifierChain
- C-2: Weighted BCE + ExtraTrees
- C-3: Weighted BCE + RandomForest
- C-4: Weighted BCE + AttentionXML
- C-5: Weighted BCE + LightXML

**Focal Loss γ=5 (C-6 to C-10):**
- C-6: Focal γ=5 + ClassifierChain
- C-7: Focal γ=5 + ExtraTrees
- C-8: Focal γ=5 + RandomForest
- C-9: Focal γ=5 + AttentionXML
- C-10: Focal γ=5 + LightXML

**Total: 24 experiments** across all parts

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

### Recent Updates
- ✅ **AttentionXML & LightXML added** - State-of-the-art XMC methods for large label spaces
- ✅ **ClassifierChain progress bar fixed** - No more 499 lines spam, clean single progress bar
- ✅ **Part C expanded to 10 strategies** - Full loss × method matrix testing
- ✅ **Augmentation module complete** - IoC, back-translation, oversampling all working
- ✅ **mAP metric integrated** - Ranking quality for SOC analyst workflow

### Why These Features?
Based on state-of-the-art CTI classification research:
1. **Differential LR**: Prevents CTI-BERT overfitting while enabling fast task adaptation
2. **Defanged Normalization**: Addresses real-world CTI report obfuscation practices
3. **Sliding Windows**: Handles variable-length threat intelligence documents
4. **XMC Methods**: AttentionXML and LightXML for efficient large-scale multi-label classification
5. **mAP Metric**: Evaluates ranking quality (critical for recommendation systems)

### Architecture Highlights
- **AttentionXML**: Label-specific attention (499 attention query vectors)
- **LightXML**: Two-stage architecture (50 label groups → candidate ranking)
- **ClassifierChain**: Sequential dependency modeling with clean progress tracking
- **Tree Ensembles**: Balanced class weights for imbalanced multi-label

### Known Limitations
- Sliding windows currently use first window only (single prediction)
- Multi-window aggregation requires architecture changes
- Normalization regex may miss novel obfuscation patterns
- XMC methods simplified for 499 labels (original papers target 100K+ labels)

---

**Academic use only**
