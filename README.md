# MITRE ATT&CK TTP Mapping

Multi-label classification system using BERT to tag Tactics, Techniques, and Procedures (TTPs) from MITRE ATT&CK framework.

# MITRE ATT&CK TTP Mapping

Multi-label classification system using BERT to tag Tactics, Techniques, and Procedures (TTPs) from MITRE ATT&CK framework.

## 🎯 Dataset

**Hybrid Dataset (Default)** - Combines 3 sources for improved quality:
- **tumeteor/Security-TTP-Mapping** (14,936 samples) - CTI reports
- **sarahwei/cyber_MITRE_attack...** (654 samples) - MITRE Q&A
- **Zainabsa99/mitre_attack** (508 samples) - Technique descriptions
- **Total:** ~16,098 samples
- **Labels:** MITRE ATT&CK techniques (T-codes)
- **Advantage:** Better label distribution, reduced class imbalance

## 📁 Structure

```
├── run_strategy_test.ipynb  # Modular strategy testing notebook
├── src/                      # Source code
│   ├── data_loader.py       # Dataset loading & preprocessing
│   ├── model.py             # BERT model with Focal Loss support
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation metrics
│   └── strategies.py        # Class imbalance strategies
├── data/                     # Dataset cache
├── outputs/                  # Training results
└── requirements.txt
```

## 🚀 Quick Start

### Google Colab (Recommended)

1. Upload `run_strategy_test.ipynb` to Colab
2. Set Runtime to GPU (T4)
3. Run setup cells (1-4)
4. Test strategies one by one

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook run_strategy_test.ipynb
```

## 🧪 Class Imbalance Strategies

The project implements 5 different strategies to handle severe class imbalance (1:458 ratio):

1. **Baseline BCE** - Standard Binary Cross-Entropy
2. **Weighted BCE** - Frequency-based per-label weighting (most promising)
3. **Focal Loss (γ=2)** - Moderate focusing on hard examples
4. **Focal Loss (γ=5)** - Strong focusing on hard examples
5. **Top-100 Subset** - Train on 100 most frequent labels

## 📊 Results

Results are saved to `outputs/strategy_comparison_[timestamp]/`:
- `strategy_comparison_[timestamp].json` - All strategy results
- `strategy_comparison_[timestamp].csv` - Comparison table
- **Metrics:** Micro F1, Recall@5, Precision@5, Recall@10, Precision@10, Example-Based Accuracy

### Why These Metrics?
- **Micro F1:** Overall performance across all labels
- **Recall@K / Precision@K:** Ranking-based metrics (suitable for sparse multi-label)
- **Example-Based Accuracy:** Sample-wise exact match (strict metric)
- ❌ Removed: Subset Accuracy, Hamming Loss (misleading for ultra-sparse labels)

---

**Academic use only**
