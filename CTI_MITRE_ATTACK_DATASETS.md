# Cyber Threat Intelligence Datasets with MITRE ATT&CK Labels

## Summary
This document describes the dataset used in this project for training and evaluating CTI-BERT on MITRE ATT&CK technique mapping.

---

## 🎯 Current Usage in This Project

**Single Dataset Configuration (Default)**

This project uses a single dataset for all experiments:

1. **tumeteor/Security-TTP-Mapping** (14,936 train + 3,170 test samples)

Notes:
- The repository may list other MITRE-related datasets as references, but they are **not used** in this project run.
- The data pipeline loads a single Hugging Face dataset via `prepare_data(dataset_name=...)`.

---

## Dataset Details

### tumeteor/Security-TTP-Mapping

- **Platform:** Hugging Face
- **Dataset:** https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping
- **Task:** Multi-label mapping from CTI text to MITRE ATT&CK technique IDs

This dataset is the sole data source for all experiments in this repository.

