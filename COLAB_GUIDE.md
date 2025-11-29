# Google Colab Training Guide

## Hızlı Başlangıç

1. **Colab'ı Aç:**
   - `colab_training.ipynb` dosyasını Google Colab'a yükle
   - Veya direkt link: [Open in Colab](https://colab.research.google.com/)

2. **GPU'yu Aktifleştir:**
   - Runtime → Change runtime type → T4 GPU (ücretsiz)
   - Save

3. **Çalıştır:**
   - Runtime → Run all
   - Veya her hücreyi sırayla çalıştır

## Beklenen Süre (T4 GPU)

- **Data Loading:** ~1-2 dakika
- **Tokenization:** ~2-3 dakika
- **Training (3 epochs):** ~30-45 dakika
- **Evaluation:** ~2-3 dakika
- **Toplam:** ~40-50 dakika

## CPU vs GPU Karşılaştırması

| İşlem | CPU (MacBook) | GPU (T4 Colab) |
|-------|---------------|----------------|
| Epoch | ~2.5 saat | ~10-15 dakika |
| 3 Epoch | ~7.5 saat | ~30-45 dakika |
| **Hız farkı** | **1x** | **~15x** |

## Model Seçenekleri

Notebook'ta `MODEL_NAME` değişkenini değiştirerek farklı modeller deneyebilirsin:

```python
# Seçenek 1: BERT-base (genel domain)
MODEL_NAME = "bert-base-uncased"

# Seçenek 2: SecBERT (security domain)
MODEL_NAME = "jackaduma/SecBERT"

# Seçenek 3: RoBERTa-base
MODEL_NAME = "roberta-base"
```

## Hyperparameter Tuning

Daha iyi sonuçlar için şunları deneyebilirsin:

```python
# Daha uzun eğitim
NUM_EPOCHS = 5  # Default: 3

# Daha büyük batch size (GPU memory yeterse)
BATCH_SIZE = 32  # Default: 16

# Farklı learning rate
LEARNING_RATE = 3e-5  # Default: 2e-5
```

## Sonuçları İndirme

Notebook sonunda otomatik olarak şunlar indirilir:
- `cti_bert_model.pt` - Eğitilmiş model
- `results.json` - Metrikler ve training history

Bu dosyaları local projeye kopyala:
```bash
# Mac/Linux
cp ~/Downloads/cti_bert_model.pt ./models/
cp ~/Downloads/results.json ./outputs/
```

## Sorun Giderme

### GPU memory hatası
```python
# Batch size'ı küçült
BATCH_SIZE = 8  # veya 4
```

### Timeout hatası
```python
# Epoch sayısını azalt
NUM_EPOCHS = 1  # veya 2
```

### Dataset yüklenmiyor
- Colab'ın internet bağlantısını kontrol et
- Runtime'ı restart et: Runtime → Restart runtime

## Sonraki Adımlar

Eğitim tamamlandıktan sonra:

1. **Sonuçları Analiz Et:** F1, Precision, Recall skorlarını incele
2. **Model Kaydet:** İndirilen modeli local'e taşı
3. **Adım 2'ye Geç:** Hierarchical F1 ve Classifier Chains ekle
4. **Farklı Modeller Dene:** SecBERT, RoBERTa vb.

---

**Not:** Colab ücretsiz versiyonu günde ~12 saat GPU kullanımı sağlar. Eğitim ~1 saat süreceği için tek seferde bitecektir.
