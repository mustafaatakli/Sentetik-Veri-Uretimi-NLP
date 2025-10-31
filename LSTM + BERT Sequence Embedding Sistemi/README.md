# LSTM + BERT Sequence Embedding Duygu Analizi Sistemi

Türkçe metinler için BERT sequence embedding ve Bidirectional LSTM kullanan hibrit duygu analizi sistemi.

## Özellikler

- **BERT Sequence Embedding**: Her kelime için ayrı embedding vektörü (768-boyutlu)
- **Bidirectional LSTM**: İki katmanlı çift yönlü LSTM ağı
- **Türkçe Destek**: dbmdz/bert-base-turkish-cased modeli
- **3 Sınıf Duygu Analizi**: Pozitif, Negatif, Nötr
- **Gemini Karşılaştırma**: Gemini AI ile tahmin karşılaştırması
- **GPU Optimizasyonu**: TensorFlow ve PyTorch GPU desteği

## Sistem Gereksinimleri

### Donanım
- **GPU**: NVIDIA GPU (önerilen: Tesla T4 veya üzeri)
- **RAM**: Minimum 8GB (önerilen: 16GB+)
- **Depolama**: ~2GB model ve kütüphaneler için

### Yazılım
- Python 3.8+
- CUDA 11.x (GPU kullanımı için)
- Kaggle ortamı (önerilen)

## Kurulum

### 1. Gerekli Kütüphaneler

```bash
pip install transformers
pip install openpyxl
pip install tensorflow
pip install torch
pip install scikit-learn
pip install pandas numpy
```

### 2. Veri Setleri

Projenin çalışması için iki Excel dosyası gereklidir:

1. **Eğitim Verisi** (`egitim-veriseti.xlsx`):
   - Sütunlar: `text`, `sentiment`
   - Format: text (metin), sentiment (pozitif/negatif/nötr)

2. **Test Verisi** (`etiketsiz-test-gemini-etiketlenmis.xlsx`):
   - Sütunlar: `text`, `sentiment` (Gemini etiketleri)
   - Format: text (metin), sentiment (pozitif/negatif/nötr)

## Kullanım

### Kaggle Ortamında Çalıştırma

1. **Kaggle Notebook Oluşturun**:
   - Kaggle'da yeni notebook açın
   - GPU'yu aktif edin (Settings > Accelerator > GPU T4)

2. **Veri Setlerini Yükleyin**:
   - `egitim-veriseti.xlsx` dosyasını yükleyin
   - `etiketsiz-test-gemini-etiketlenmis.xlsx` dosyasını yükleyin

3. **Dosya Yollarını Güncelleyin**:
   ```python
   EGITIM_DOSYASI = '/kaggle/input/[dataset-name]/egitim-veriseti.xlsx'
   ETIKETSIZ_DOSYA = '/kaggle/input/[dataset-name]/etiketsiz-test-gemini-etiketlenmis.xlsx'
   ```

4. **Kodu Çalıştırın**:
   - `main.py` dosyasını notebook hücresine yapıştırın
   - Tüm hücreleri çalıştırın (Run All)

### Yerel Ortamda Çalıştırma

```bash
python main.py
```

**Not**: Dosya yollarını yerel sisteminize göre düzenleyin:
```python
EGITIM_DOSYASI = 'path/to/egitim-veriseti.xlsx'
ETIKETSIZ_DOSYA = 'path/to/etiketsiz-test-gemini-etiketlenmis.xlsx'
CIKTI_DOSYASI = 'path/to/bert_vs_gemini_sonuc.xlsx'
```

## Model Mimarisi

### 1. BERT Embedding Katmanı
- **Model**: dbmdz/bert-base-turkish-cased
- **Çıktı**: Sequence embeddings (max_length × 768)
- **Her token için ayrı vektör**

### 2. LSTM Ağı
```
Input (64 × 768)
    ↓
Masking Layer (padding için)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dense (64, ReLU) + Dropout (0.3)
    ↓
Dense (32, ReLU) + Dropout (0.3)
    ↓
Dense (3, Softmax)
```

### Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Max Sequence Length | 64 |
| LSTM Units | 128 (1. katman), 64 (2. katman) |
| Dropout | 0.3 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Max Epochs | 20 |
| Early Stopping Patience | 5 |

## Eğitim Süreci

### Adımlar

1. **BERT Modelini Yükle**: Türkçe BERT modeli yüklenir
2. **Veri Yükle**: Excel dosyasından eğitim verisi okunur
3. **Veri Bölme**: %80 eğitim, %10 doğrulama, %10 test
4. **BERT Embedding**: Her metin için sequence embedding çıkarılır
5. **Model Oluştur**: Bidirectional LSTM modeli oluşturulur
6. **Eğitim**: Model eğitilir (early stopping ile)
7. **Değerlendirme**: Test seti üzerinde performans ölçülür
8. **Tahmin**: Yeni veriler etiketlenir
9. **Karşılaştırma**: Gemini ile tahminler karşılaştırılır

### Callback'ler

- **ModelCheckpoint**: En iyi modeli kaydeder
- **EarlyStopping**: Aşırı öğrenmeyi önler (5 epoch patience)
- **ReduceLROnPlateau**: Öğrenme oranını dinamik azaltır

## Çıktılar

### 1. Eğitim Sonuçları

Konsol çıktısında:
- Test accuracy, precision, recall, F1-score
- Sınıf bazında detaylı rapor
- Confusion matrix

### 2. Tahmin Dosyası

`bert_vs_gemini_sonuc.xlsx` dosyası aşağıdaki sütunları içerir:

| Sütun | Açıklama |
|-------|----------|
| text | Orijinal metin |
| gemini_sentiment | Gemini'nin tahmini |
| real_lstm_bert_sentiment | LSTM+BERT'ün tahmini |
| real_lstm_bert_conf_negatif | Negatif sınıfı güven skoru |
| real_lstm_bert_conf_notr | Nötr sınıfı güven skoru |
| real_lstm_bert_conf_pozitif | Pozitif sınıfı güven skoru |
| real_lstm_bert_conf_score | En yüksek güven skoru |

### 3. Model Dosyaları

- `real_lstm_bert_model.h5`: Eğitilmiş LSTM modeli
- `real_lstm_bert_config.pickle`: Model konfigürasyonu
- `best_real_lstm_bert_model.h5`: En iyi performanslı model

## Performans

### Beklenen Metrikler

- **Accuracy**: %89-92
- **Precision**: ~%90
- **Recall**: ~%90
- **F1-Score**: ~%90

### Önceki Versiyondan Farklar

Bu sistem, BERT + Dense layer versiyonundan **%2-4 daha iyi** performans gösterir:

**Neden?**
- Sequence embedding kullanır (kelime sırası korunur)
- Bidirectional LSTM ile bağlamsal analiz
- İki katmanlı LSTM ile daha derin öğrenme

## Gemini Karşılaştırma

Sistem, Gemini AI ile otomatik karşılaştırma yapar:

### Metrikler
- Uyuşma oranı (agreement rate)
- Sınıf bazında accuracy
- Confusion matrix
- Güven skoruna göre analiz

### Örnek Çıktı
```
UYUŞMA ANALİZİ:
  ✅ Uyuşan tahminler: 850 (%85.00)
  ❌ Farklı tahminler: 150 (%15.00)

GÜVEN SKORUNA GÖRE ANALİZ:
  Uyuşan tahminlerde LSTM+BERT güveni: 0.9234
  Farklı tahminlerde LSTM+BERT güveni: 0.7156
```

## Sorun Giderme

### GPU Bulunamıyor

```
⚠️  GPU bulunamadı, CPU kullanılıyor
```

**Çözüm**:
- Kaggle'da GPU aktif edin (Settings > Accelerator > GPU T4)
- Yerel ortamda CUDA kurulumunu kontrol edin

### Dosya Bulunamadı

```
❌ HATA: Excel dosyasında 'text' sütunu bulunamadı!
```

**Çözüm**:
- Excel dosyasında `text` ve `sentiment` sütunları olduğundan emin olun
- Dosya yollarını kontrol edin

### Bellek Hatası (OOM)

```
ResourceExhaustedError: OOM when allocating tensor
```

**Çözüm**:
- `BATCH_SIZE` değerini azaltın (örn. 8)
- `MAX_LENGTH` değerini azaltın (örn. 48)
- GPU memory growth ayarını kontrol edin

## Örnek Kullanım

```python
# Model yükleme
from tensorflow.keras.models import load_model
import pickle

model = load_model('real_lstm_bert_model.h5')

with open('real_lstm_bert_config.pickle', 'rb') as f:
    config = pickle.load(f)

# Yeni metin tahmini
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
bert_model = AutoModel.from_pretrained(config['bert_model'])

# Metin embedding çıkarma
text = "Bu ürün gerçekten harika!"
# ... (embedding çıkarma kodu)

# Tahmin
prediction = model.predict(embedding)
sentiment = ['negatif', 'nötr', 'pozitif'][prediction.argmax()]
```

### Kullanılan Bileşenler

Bu proje aşağıdaki açık kaynak bileşenleri kullanmaktadır:

- **BERT Model (dbmdz/bert-base-turkish-cased)**: Apache 2.0 License
- **TensorFlow**: Apache 2.0 License
- **PyTorch**: BSD License
- **Transformers (Hugging Face)**: Apache 2.0 License
- **scikit-learn**: BSD License

## Katkıda Bulunanlar

- **BERT Model**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) - Stefan Schweter
- **Gemini AI**: Google DeepMind
- **Geliştirici**: [Adınız]

## Teşekkürler

- Hugging Face ekibine Transformers kütüphanesi için
- TensorFlow ve PyTorch topluluklarına
- dbmdz ekibine Türkçe BERT modeli için
- Google DeepMind'a Gemini AI için

---

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

### Yıldız Vermeyi Unutmayın! ⭐

Bu projeyi faydalı bulduysanız, GitHub'da yıldız vererek destek olabilirsiniz!