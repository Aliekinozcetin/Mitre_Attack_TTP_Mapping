# MITRE ATT&CK TTP Mapping with CTI-BERT

Multi-label classification system to map Cyber Threat Intelligence reports to MITRE ATT&CK TTPs using **CTI-BERT** (IBM Research).

## 🎯 Overview

**Model:** CTI-BERT (`ibm-research/CTI-BERT`)
- Domain-specific BERT pre-trained on security data
- Differential learning rate (encoder: 2e-5, classifier: 1e-4)

**Dataset:** Security-TTP-Mapping (`tumeteor/Security-TTP-Mapping`)
- 14,936 train + 3,170 test samples
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
- **Micro-F1:** Overall performance across all labels
- **Micro-Precision/Recall:** Component metrics for F1
- **Hamming Loss:** Fraction of wrong labels (lower is better)
- **Example-Based Accuracy:** Exact match per sample (strict metric)

### Ranking Metrics (SOC Analyst Perspective)
- **mAP (Mean Average Precision):** Ranking quality - rewards correct TTPs at top of list ⭐ **Most Important**
- **Recall@5:** Percentage of true TTPs in top-5 predictions
- **Precision@5:** Accuracy of top-5 predictions
- **Recall@10:** Percentage of true TTPs in top-10 predictions (CSV only)
- **Precision@10:** Accuracy of top-10 predictions (CSV only)

### Why These Metrics?
This is effectively a **recommendation system for SOC analysts**:
- **mAP** evaluates entire ranking (better than Recall@K alone)
- **Recall@5** matters because analysts review top 5 predictions first
- **Hamming Loss** shows overall prediction accuracy
- Top-10 metrics included in CSV exports for detailed analysis

### Results Output

Each experiment generates comprehensive outputs:

**Model Checkpoints:**
- `outputs/bert-base-uncased_[timestamp]/final_model.pt`
- `outputs/bert-base-uncased_[timestamp]/checkpoint_epoch_*.pt`

**Metrics & Logs:**
- `evaluation_metrics.json` - All metrics (F1, mAP, Recall@K, etc.)
- `training_history.json` - Loss/accuracy curves per epoch
- `summary.json` - Configuration + final results
- `labels.json` - Label mappings

**Comparison Tables (CSV):**
- `outputs/augmentation_comparison.csv` - Part A results
- `outputs/loss_function_comparison.csv` - Part B-1 results
- `outputs/topk_analysis.csv` - Part B-2 results
- `outputs/hybrid_strategies_comparison.csv` - Part C results

All CSVs include Training_Time_min and all metrics (including @10)

**Visualizations (Line Charts with Best Score Markers):**

Each comparison generates line charts with:
- Smooth lines connecting strategy performances
- Filled area under curves
- Red star (⭐) markers at best scores
- Value labels on data points
- Professional styling (300 DPI)

Example plots:
- `outputs/augmentation_plots/micro_f1_comparison.png`
- `outputs/loss_function_plots/map_comparison.png`
- `outputs/hybrid_strategies_plots/recall_at_5.png`

**Note:** Visualizations show @5 metrics only. @10 metrics available in CSV files.

## 🔧 Implementation Notes

### Why These Features?
Based on state-of-the-art CTI classification research:
1. **Differential LR**: Prevents CTI-BERT overfitting while enabling fast task adaptation
2. **Defanged Normalization**: Addresses real-world CTI report obfuscation practices
3. **Sliding Windows**: Handles variable-length threat intelligence documents
4. **XMC Methods**: AttentionXML and LightXML for efficient large-scale multi-label classification
5. **mAP Metric**: Evaluates ranking quality (critical for recommendation systems)
6. **Line Chart Visualizations**: Track performance trends across strategies with best score markers
7. **Comprehensive Exports**: CSV files with all metrics for detailed post-analysis

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
