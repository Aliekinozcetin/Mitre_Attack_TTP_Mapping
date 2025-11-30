# MITRE ATT&CK TTP Mapping

Multi-label classification system using BERT-based models to tag Tactics, Techniques, and Procedures (TTPs) from MITRE ATT&CK framework.

## 🎯 Dataset

**tumeteor/Security-TTP-Mapping** (20,736 samples)
- 14,936 training / 2,630 validation / 3,170 test
- 499 unique MITRE ATT&CK techniques
- Real CTI reports and threat descriptions

## 📁 Structure

```
├── main.py               # Training script
├── src/                  # Source code
│   ├── data_loader.py   # Dataset loading & preprocessing
│   ├── model.py         # BERT model setup
│   ├── train.py         # Training loop
│   └── evaluate.py      # Evaluation metrics
├── data/                 # Dataset cache
├── outputs/              # Training results
└── requirements.txt
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train BERT model
python main.py --model bert-base-uncased --epochs 3 --batch_size 16

# Train SecBERT (security-specific)
python main.py --model jackaduma/SecBERT --epochs 3 --batch_size 16
```

## 📊 Available Models

- **bert-base-uncased** - General purpose (recommended)
- **jackaduma/SecBERT** - Security domain specific
- **roberta-base** - Alternative baseline
- **distilbert-base-uncased** - Faster, lighter version

## 📈 Results

Results are saved to `outputs/[model-name]_[timestamp]/`:
- `final_model.pt` - Trained model
- `evaluation_metrics.json` - F1, Precision, Recall
- `training_history.json` - Loss curves
- `labels.json` - Label mapping

## 🛠️ CLI Options

```bash
python main.py \
  --model bert-base-uncased \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --max_length 512 \
  --device cuda
```

---

**Academic use only**
