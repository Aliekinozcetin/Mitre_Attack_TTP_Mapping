# CTI-BERT TTP Tagging Project

This project implements a multi-label classification system for Cyber Threat Intelligence (CTI) using BERT-based models to tag Tactics, Techniques, and Procedures (TTPs) from the MITRE ATT&CK framework.

## 🎯 Dataset

**tumeteor/Security-TTP-Mapping** (20,736 samples)
- 14,936 training samples
- 2,630 validation samples  
- 3,170 test samples
- 499 unique MITRE ATT&CK techniques
- Multi-label classification task
- Real CTI reports and threat descriptions

## 📁 Project Structure

```
.
├── colab_training.ipynb   # Main training notebook (Google Colab + VS Code)
├── outputs/               # Saved models and results
├── data/                  # Dataset cache
├── models/                # Model checkpoints
├── COLAB_GUIDE.md        # Detailed Colab usage guide
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### Option 1: VS Code + Google Colab (Recommended)

1. Open `colab_training.ipynb` in VS Code
2. Connect to Google Colab runtime (GPU enabled)
3. Run all cells

### Option 2: Direct Google Colab

1. Upload `colab_training.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → GPU (T4)
3. Runtime → Run all

## 📊 Expected Performance

| Metric | CPU (Local) | GPU (T4 Colab) |
|--------|-------------|----------------|
| Training Time (3 epochs) | ~7.5 hours | ~40 minutes |
| Speed Improvement | 1x | **15x faster** |

## 📈 Roadmap

### ✅ Step 1: Hello World Pipeline (COMPLETED)
- ✅ Load tumeteor/Security-TTP-Mapping dataset (20k samples)
- ✅ Train BERT-base-uncased with BCEWithLogitsLoss
- ✅ F1, Precision, Recall metrics
- ✅ Google Colab integration

### 🔄 Step 2: Advanced Techniques (In Progress)
- [ ] Hierarchical F1 metric (MITRE ATT&CK hierarchy)
- [ ] Classifier Chains for label dependencies
- [ ] SecBERT (domain-specific model) comparison
- [ ] Attention visualization

### 🎯 Step 3: Production Ready
- [ ] Model serving API
- [ ] Real-time TTP extraction
- [ ] Dashboard for results

## 📚 Documentation

- **[COLAB_GUIDE.md](COLAB_GUIDE.md)** - Detailed Colab training guide
- **[CTI_MITRE_ATTACK_DATASETS.md](CTI_MITRE_ATTACK_DATASETS.md)** - Dataset research

## 🛠️ Dependencies

```bash
pip install torch transformers datasets scikit-learn tqdm matplotlib
```

See `requirements.txt` for full list.

## 📄 License

Academic use only.
