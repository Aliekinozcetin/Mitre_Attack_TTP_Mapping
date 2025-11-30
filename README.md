# MITRE ATT&CK TTP Mapping

Multi-label classification system using BERT-based models to tag Tactics, Techniques, and Procedures (TTPs) from MITRE ATT&CK framework.

## 🎯 Dataset

**tumeteor/Security-TTP-Mapping** (20,736 samples)
- 14,936 training / 2,630 validation / 3,170 test
- 499 unique MITRE ATT&CK techniques
- Real CTI reports and threat descriptions

## 📁 Structure

```
├── run_training.ipynb    # Google Colab training notebook
├── main.py               # CLI training script
├── src/                  # Source code
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── data/                 # Dataset cache
├── outputs/              # Training results
└── requirements.txt
```

## 🚀 Quick Start (Google Colab)

1. Open in Colab: `run_training.ipynb`
2. Runtime → Change runtime type → **GPU (T4)**
3. Run cells sequentially:
   - Setup
   - Training (BERT or SecBERT)
   - Download results as ZIP

**Training time:** ~40 minutes (T4 GPU)

## 🛠️ Local Training

```bash
pip install -r requirements.txt
python main.py --model bert-base-uncased --epochs 3 --batch_size 16
```

## 📊 Models

- **BERT-base-uncased** - General purpose (recommended)
- **jackaduma/SecBERT** - Security domain specific

## 📈 Performance Metrics

- Micro/Macro F1, Precision, Recall
- Multi-label classification with BCEWithLogitsLoss
- Results saved to `outputs/`

## 📚 Documentation

- **[CTI_MITRE_ATTACK_DATASETS.md](CTI_MITRE_ATTACK_DATASETS.md)** - Dataset details

---

**Academic use only**
