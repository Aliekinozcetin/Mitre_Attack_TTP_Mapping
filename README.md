# MITRE ATT&CK TTP Mapping with CTI-BERT

Multi-label classification system to map Cyber Threat Intelligence reports to MITRE ATT&CK TTPs using **CTI-BERT** (IBM Research).

## 🎯 Overview

**Model:** CTI-BERT (`ibm-research/CTI-BERT`)
- Domain-specific BERT pre-trained on security data
- Differential learning rate (encoder: 2e-5, classifier: 1e-4)

**Dataset:** Security-TTP-Mapping (`tumeteor/Security-TTP-Mapping`)
- 14,936 train + 2,638 test samples
- 499 MITRE ATT&CK technique labels (multi-label)
- Severe class imbalance (1:458 ratio)

## 📁 Project Structure

```
├── run_strategy_test.ipynb  # Main experiment notebook (24 strategies)
├── src/
│   ├── data_loader.py       # CTI preprocessing + sliding windows
│   ├── model.py             # CTI-BERT with Focal/Weighted BCE
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Metrics (F1, mAP, Recall@K)
│   ├── augmentation.py      # IoC replacement, back-translation, oversampling
│   ├── classifier_chain.py  # Sklearn ClassifierChain
│   ├── attention_xml.py     # AttentionXML (NeurIPS 2019)
│   ├── light_xml.py         # LightXML (AAAI 2021)
│   └── xml_utils.py         # XMC training utilities
└── outputs/                  # Results & checkpoints
```

## 🚀 Quick Start

**Google Colab:**
1. Upload `run_strategy_test.ipynb`
2. Set Runtime → GPU (T4+)
3. Run cells sequentially

**Local:**
```bash
pip install -r requirements.txt
jupyter notebook run_strategy_test.ipynb
```

## 🧪 Experiments (24 Total Strategies)

### **PART A: Data Augmentation (5 strategies)**
Improve tail TTP performance:
- **A-1:** Baseline (no augmentation)
- **A-2:** IoC Replacement (randomize IPs, hashes, domains)
- **A-3:** Back-translation (EN→DE→EN paraphrasing)
- **A-4:** Oversampling (replicate rare TTPs 3x-10x)
- **A-5:** Combined (all methods)

**Duration:** ~4-5 hours

---

### **PART B: Loss Functions (9 strategies)**

**Section 1: Loss Comparison (4 strategies)**
- **STR-1:** Baseline BCE
- **STR-2:** Weighted BCE (frequency-based, handles 1:458 imbalance)
- **STR-3:** Focal Loss (γ=2)
- **STR-4:** Focal Loss (γ=5)

**Section 2: Capacity Testing (5 variants)**
- **STR-5:** Top-K analysis (K = 5, 10, 20, 50, 100 labels)

**Duration:** ~5-6 hours

---

### **PART C: Hybrid Strategies (10 strategies)**
Test 2 best losses × 5 classification methods:

**Classification Methods:**
1. **ClassifierChain** - Sequential label dependencies
2. **ExtraTrees** - Fast randomized ensemble
3. **RandomForest** - Optimal split ensemble
4. **AttentionXML** - Label-specific attention (NeurIPS 2019)
5. **LightXML** - Two-stage + negative sampling (AAAI 2021)

**Matrix:**
```
                    Chain  ExtraTrees  RandomForest  AttentionXML  LightXML
Weighted BCE        C-1    C-2         C-3           C-4           C-5
Focal γ=5           C-6    C-7         C-8           C-9           C-10
```

**Duration:** ~7.5-10 hours

---

## ⏱️ Execution Guide

**Recommended Order:**
```
PART A → Find best augmentation (e.g., A-5)
   ↓
PART B → Find best loss (e.g., Weighted BCE)
   ↓
PART C → Find best classifier (test all 10 combos)
```

**Flexible:** Each part is independent - run in any order

**Time Estimates:**
- Quick test: A-1 + STR-2 + C-1 → ~1.5 hours
- Full run: 24 strategies → ~17-21 hours

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
