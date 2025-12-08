# MITRE ATT&CK TTP Mapping

Multi-label classification system using BERT to tag Tactics, Techniques, and Procedures (TTPs) from MITRE ATT&CK framework.

## 🎯 Dataset

**tumeteor/Security-TTP-Mapping** (20,736 samples)
- 14,936 training / 2,630 validation / 3,170 test
- 499 unique MITRE ATT&CK techniques
- Real CTI reports and threat descriptions

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
- Metrics: F1 (Top-5/10), Precision, Recall, Hamming Loss

---

**Academic use only**
