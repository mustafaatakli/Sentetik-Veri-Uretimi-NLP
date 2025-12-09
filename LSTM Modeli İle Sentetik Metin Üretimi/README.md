# LSTM Modeli İle Sentetik Metin Üretimi

Bu proje, Character-level LSTM (Long Short-Term Memory) sinir ağı kullanarak Türkçe teknoloji haber başlıklarından sentetik metin üretimi gerçekleştirmektedir. Sıfırdan eğitilen özel LSTM modeli ile karakter seviyesinde üretim yapılmaktadır.

## Proje Açıklaması

Proje, 100 adet gerçek teknoloji haber başlığından yola çıkarak, sıfırdan eğitilen character-level LSTM modeli ile 3000 adet sentetik haber başlığı üretmektedir. Üretilen veriler, diğer yöntemlerle (BERT MLM, Gemini API, GPT-2) karşılaştırılabilir standart metriklerle değerlendirilmektedir.

## Ana Özellikler

### 1. Character-Level LSTM Modeli

#### Model Mimarisi
```python
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.3):
        # Embedding Layer: vocab_size → 256
        # LSTM Layers: 2 katman, 512 hidden units
        # Output Layer: 512 → vocab_size
        # Dropout: 0.3 (overfitting önleme)
```

#### Model Özellikleri
- **Parametre Sayısı**: ~3.74 milyon
- **Vocabulary**: 75 karakter (Türkçe karakterler + özel tokenlar)
- **Embedding Dim**: 256
- **Hidden Dim**: 512
- **Layers**: 2 LSTM katmanı
- **Dropout**: 0.3

#### Özel Tokenlar
```python
'<PAD>': 0    # Padding
'<START>': 1  # Cümle başlangıcı
'<END>': 2    # Cümle sonu
'<UNK>': 3    # Bilinmeyen karakter
```

### 2. Custom Character Tokenizer

Character-level yaklaşım avantajları:
- **Vocabulary Sınırlaması Yok**: Her karakter tanınır
- **OOV Problemi Yok**: Out-of-vocabulary sorunu olmaz
- **Türkçe Uyumlu**: Tüm Türkçe karakterler desteklenir (ğ, ü, ş, ı, ö, ç)
- **Kelime Segmentasyonu Gerektirmez**: Boşluklar da karakter olarak işlenir

```python
class CharTokenizer:
    def fit(texts):        # Karakterleri öğren
    def encode(text):      # Metni token ID'lere çevir
    def decode(tokens):    # Token ID'leri metne çevir
```

### 3. Eğitim Stratejisi

#### Next Character Prediction
```python
# Input:  "<START>Google şov yapaca"
# Target: "Google şov yapacak<END>"

# LSTM bir sonraki karakteri tahmin etmeyi öğrenir
```

#### Eğitim Parametreleri
```python
epochs = 50
learning_rate = 0.001
batch_size = 32
optimizer = Adam
criterion = CrossEntropyLoss (PAD token ignore edilir)
gradient_clipping = 5.0  # Gradient explosion önleme
```

#### Loss Grafiği (Örnek)
```
Epoch 1:  Loss: 4.14
Epoch 10: Loss: 2.15
Epoch 25: Loss: 1.35
Epoch 50: Loss: 0.85
```

### 4. Generation Stratejisi

#### Temperature Sampling
```python
temperature = 0.9 + random.uniform(-0.2, 0.2)  # 0.7-1.1 arası
temperature = max(0.5, min(1.5, temperature))

# Yüksek temperature → Daha çeşitli (riskli)
# Düşük temperature → Daha tutarlı (güvenli)
```

#### Prefix-Based Generation
```python
# Rastgele bir prefix kullan (5-15 karakter)
prefix_len = random.randint(5, min(15, len(seed_text) - 5))
prefix = seed_text[:prefix_len]

# Örnek:
# Seed: "Google şov yapacak Google I/O 2025"
# Prefix: "Google şov"
# → LSTM devam ettirir: "Google şov tanıtımı 2025 etkinliğinde"
```

#### Autoregressive Generation
```python
# Her adımda bir karakter üretilir
# Üretilen karakter bir sonraki inputa eklenir
# <END> token'ı görülene kadar devam eder

for _ in range(max_len):
    next_char = model.predict(current_input, temperature)
    generated += next_char
    current_input = next_char

    if next_char == '<END>':
        break
```

### 5. Kalite Kontrol

#### A. Geçerlilik Kontrolleri
- Minimum uzunluk: 10 karakter
- Maksimum uzunluk: 200 karakter
- Kelime sayısı: 3-20 kelime
- Orijinalden farklılık: En az 2 kelime

#### B. Çeşitlilik Mekanizmaları
```python
# 1. Rastgele prefix seçimi
# 2. Temperature varyasyonu (±0.2)
# 3. Her cümleden n_varyant kadar farklı üretim
# 4. Normalize edilmiş kontrol (tekrar önleme)
```

#### C. BERT Perplexity (Raporlama Amaçlı)
**ÖNEMLİ**: LSTM için perplexity **FİLTRELEME YAPILMAZ**
- Adil karşılaştırma için 3000 cümle garanti edilir
- Perplexity sadece kalite raporu için hesaplanır
- LSTM karakteristik olarak daha düşük kaliteli üretim yapar

### 6. 5 Temel Metrik Analizi

Diğer yöntemlerle standart karşılaştırma:

#### [1] Tekil Oran (Uniqueness)
- Her cümlenin benzersizliği
- İdeal: ≥ %90

#### [2] BERTScore F1 (Anlamsal Benzerlik)
- BERT embeddings ile anlamsal benzerlik
- İdeal: ≥ 0.80

#### [3] Kelime Kapsama (Vocabulary Coverage)
- Orijinal kelimelerin korunma oranı
- İdeal: ≥ %85

#### [4] Benzerlik Skoru (TF-IDF Cosine Similarity)
- Kelime tabanlı benzerlik
- İdeal: 0.45 - 0.75

#### [5] Perplexity Skoru (Anlamsal Doğallık)
- BERT MLM ile cümle doğallığı
- İdeal: ≤ 60 (LSTM için gevşek eşik)

## Dosya Yapısı

```
LSTM Modeli İle Sentetik Metin Üretimi/
│
├── lstm_sentetik_uretim.py                          # Ana Python scripti
├── tekonoloji-haber-baslıkları.csv                  # Orijinal veri (100 haber başlığı)
├── lstm_sentetik_teknoloji_haberleri_3000.csv       # Üretilen sentetik veri (3000 başlık)
└── lstm.txt                                         # Eğitim ve üretim log dosyası
```

## Gereksinimler

```
torch
numpy
pandas
tqdm
scikit-learn
matplotlib
transformers
bert-score
```

## Kurulum

```bash
pip install torch numpy pandas tqdm scikit-learn matplotlib transformers bert-score
```

## Kullanım

```bash
python lstm_sentetik_uretim.py
```

### Çalışma Akışı

1. **GPU Kontrolü**: CUDA kullanılabilirliği kontrol edilir
2. **BERT Yükleme**: Metrik hesaplamaları için BERT modeli yüklenir
3. **Veri Yükleme**: 100 orijinal cümle okunur
4. **Tokenizer Hazırlama**: Character tokenizer oluşturulur (75 karakter)
5. **LSTM Model Oluşturma**: 3.74M parametreli model (~2 MB)
6. **Dataset Hazırlama**: 100 örnek, batch_size=32
7. **Eğitim** (5-10 dakika):
   - 50 epoch
   - Cross-entropy loss
   - Gradient clipping
   - Progress bar ile ilerleme
8. **Sentetik Üretim** (değişken süre):
   - Her cümleden varyantlar üretilir
   - Prefix-based generation
   - Temperature sampling
   - Her 10 cümlede bir durum raporu
9. **Tekrar Temizleme**: Çift cümleler çıkarılır
10. **Eksik Kontrolü**: Hedef sayıya ulaşılmazsa ek üretim
11. **Perplexity Hesaplama**: BERT ile kalite değerlendirmesi (filtreleme YOK)
12. **CSV Kayıt**: Sonuçlar kaydedilir
13. **5 Temel Metrik Analizi**: Kapsamlı değerlendirme

## Teknik Detaylar

### LSTM Forward Pass

```python
# 1. Embedding
embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]

# 2. LSTM
output, (h_n, c_n) = self.lstm(embedded, hidden)  # [batch, seq_len, hidden_dim]

# 3. Dropout + Linear
output = self.dropout(output)
output = self.fc(output)  # [batch, seq_len, vocab_size]
```

### Training Loop

```python
for epoch in range(epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs, _ = model(inputs)

        # Loss hesaplama
        loss = criterion(outputs.view(-1, vocab_size),
                        targets.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
```

### Generation Loop

```python
def generate_text_lstm(model, tokenizer, seed_text, temperature):
    text = '<START>' + seed_text
    generated = seed_text

    for _ in range(max_len):
        # Bir sonraki karakteri tahmin et
        logits = model(current_input)
        probs = softmax(logits / temperature)

        # Sampling
        next_char = sample(probs)

        if next_char == '<END>':
            break

        generated += next_char
        current_input = next_char

    return generated
```

## Performans

| Metrik | Değer |
|--------|-------|
| Eğitim Süresi | ~5-10 dakika (50 epoch) |
| Üretim Süresi | Değişken (15-30 dakika) |
| Model Boyutu | ~15 MB (PyTorch state_dict) |
| Model Parametreleri | ~3.74 milyon |
| GPU Memory | ~500 MB (eğitim), ~200 MB (üretim) |
| Vocabulary Size | 75 karakter |
| Batch Size | 32 (eğitim) |

## Model Karşılaştırması

### LSTM vs GPT-2 vs BERT MLM vs Gemini

| Özellik | LSTM | GPT-2 | BERT MLM | Gemini |
|---------|------|-------|----------|--------|
| Üretim Yöntemi | Char-level RNN | Causal LM | Masked LM | LLM API |
| Eğitim | Sıfırdan | Pre-trained | Pre-trained | Pre-trained |
| Doğallık | Düşük-Orta | Yüksek | Orta-Yüksek | Çok Yüksek |
| Çeşitlilik | Orta | Çok Yüksek | Orta | Yüksek |
| Anlamsal Tutarlılık | Düşük | Orta-Yüksek | Orta | Yüksek |
| Eğitim Süresi | 5-10 dk | - | - | - |
| Üretim Süresi | Orta | Hızlı | Hızlı | Orta |
| Model Boyutu | ~15 MB | ~500 MB | ~500 MB | - |
| Kontrol | Orta | Orta | Yüksek | Yüksek |
| Veri Gereksinimi | 100 cümle | Milyonlarca | Milyonlarca | - |

### Avantajlar

✅ **Küçük Veri**: 100 cümle ile eğitilebilir
✅ **Küçük Model**: Sadece 3.74M parametre (~15 MB)
✅ **Hızlı Eğitim**: 5-10 dakika eğitim süresi
✅ **Özelleştirilebilir**: Sıfırdan kendi modelinizi eğitin
✅ **OOV Yok**: Character-level yaklaşım
✅ **Düşük Memory**: ~500 MB GPU memory

### Dezavantajlar

⚠️ **Düşük Kalite**: Pre-trained modellere göre düşük kaliteli üretim
⚠️ **Anlamsal Sorunlar**: Bağlam kaybı ve anlamsız kelimeler
⚠️ **Yavaş Üretim**: Karakter karakter üretim zaman alır
⚠️ **Sınırlı Genelleme**: Sadece eğitim verisi stilini öğrenir
⚠️ **Yüksek Perplexity**: BERT standartlarına göre düşük doğallık

## Sınırlamalar

1. **Küçük Eğitim Verisi**: 100 cümle sınırlı öğrenme sağlar
2. **Character-Level**: Kelime anlambilimine erişemez
3. **Yerel Bağlam**: Uzun mesafeli bağımlılıkları yakalayamaz
4. **Perplexity Filtreleme Yok**: Düşük kaliteli cümleler de dahil edilir
5. **Pre-training Yok**: Domain bilgisi sınırlı
6. **Gradient Vanishing**: Çok uzun sekanslar için sorunlu olabilir

## İyileştirme Önerileri

### Model Mimarisi

```python
# Daha derin model (daha iyi kalite, yavaş eğitim)
embedding_dim = 512
hidden_dim = 1024
num_layers = 3
dropout = 0.4

# Daha hafif model (daha hızlı, düşük kalite)
embedding_dim = 128
hidden_dim = 256
num_layers = 1
dropout = 0.2
```

### Eğitim Parametreleri

```python
# Daha uzun eğitim (daha iyi öğrenme)
epochs = 100
learning_rate = 0.0005

# Daha büyük batch (daha hızlı eğitim)
batch_size = 64  # (varsayılan: 32)

# Learning rate scheduling
scheduler = ReduceLROnPlateau(optimizer, patience=5)
```

### Temperature Ayarı

```python
# Daha tutarlı üretim (düşük çeşitlilik)
temperature = 0.6

# Daha yaratıcı üretim (yüksek çeşitlilik, riskli)
temperature = 1.2
```

### Veri Artırma

```python
# Daha fazla eğitim verisi kullan
seed_cumleler = [...][:500]  # 100 yerine 500 cümle

# Data augmentation teknikleri
# - Sinonim değiştirme
# - Rastgele kelime silme
# - Kelime sırası değiştirme
```

## Gelişmiş Teknikler

### Beam Search (Temperature Sampling Yerine)

```python
def beam_search_generate(model, seed, beam_width=5):
    # En iyi k olasılıklı yolu takip et
    # Daha tutarlı ama daha az çeşitli
    pass
```

### Attention Mechanism

```python
class LSTMWithAttention(nn.Module):
    # LSTM + Attention → Daha iyi bağlam modelleme
    # Ancak daha karmaşık ve yavaş
    pass
```

### Bidirectional LSTM

```python
self.lstm = nn.LSTM(
    embedding_dim, hidden_dim, num_layers,
    bidirectional=True  # İleri+geri yönlü
)
# Not: Generation için uygun değil (sadece encoding için)
```

## Değerlendirme Kriterleri

### İdeal Metrik Değerleri (LSTM için Gevşek)

| Metrik | BERT/GPT-2 İdeal | LSTM Hedef |
|--------|------------------|------------|
| Tekil Oran | ≥ %95 | ≥ %90 |
| BERTScore F1 | ≥ 0.85 | ≥ 0.75 |
| Kelime Kapsama | ≥ %90 | ≥ %80 |
| Benzerlik Skoru | 0.5 - 0.7 | 0.45 - 0.75 |
| Perplexity | ≤ 50 | ≤ 70 |

**Not**: LSTM için perplexity ve BERTScore eşikleri daha gevşek tutulmuştur çünkü character-level yaklaşım semantik tutarlılık açısından daha zayıftır.

### Başarı Değerlendirmesi

- **5/5 metrik**: ✓ MÜKEMMEL (LSTM için çok nadir)
- **4/5 metrik**: ✓ MÜKEMMEL
- **3/5 metrik**: ✓ İYİ - Kullanılabilir
- **2/5 metrik**: ~ ORTA - İyileştirme gerekir
- **<2 metrik**: ⚠ İYİLEŞTİRİLEBİLİR

## Sorun Giderme

### Eğitim Loss Düşmüyor
```
Loss sabit kalıyor veya çok yavaş düşüyor
```
**Çözüm**:
```python
# Learning rate'i artır
lr = 0.003  # (varsayılan: 0.001)

# Daha fazla epoch
epochs = 100

# Gradient clipping değerini kontrol et
clip_value = 10.0  # (varsayılan: 5.0)
```

### Anlamsız Karakterler Üretiliyor
```
"xyzabc123!@#" gibi anlamsız çıktılar
```
**Çözüm**:
```python
# Temperature'ı düşür
temperature = 0.6  # (varsayılan: 0.9)

# Daha uzun eğitim
epochs = 100

# Daha büyük model
hidden_dim = 1024  # (varsayılan: 512)
```

### Sadece Eğitim Cümlelerini Kopyalıyor
```
Üretilen cümleler eğitim setinin kopyası
```
**Çözüm**:
```python
# Temperature'ı artır
temperature = 1.2  # (varsayılan: 0.9)

# Dropout'u artır
dropout = 0.4  # (varsayılan: 0.3)

# Farklı prefix kullan
prefix_len = random.randint(3, 8)  # Daha kısa prefix
```

### GPU Memory Hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**:
```python
# Batch size'ı azalt
batch_size = 16  # (varsayılan: 32)

# Model boyutunu küçült
hidden_dim = 256  # (varsayılan: 512)
embedding_dim = 128  # (varsayılan: 256)
```

## Alternatif Yaklaşımlar

### Word-Level LSTM
```python
# Character yerine kelime seviyesi
# + Daha iyi semantik
# - Vocabulary sınırlaması
# - OOV problemi
```

### GRU (Gated Recurrent Unit)
```python
self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers)
# LSTM'e benzer ama daha basit ve hızlı
```

### Transformer Decoder
```python
# Attention-based model
# + Daha iyi uzun mesafe bağımlılıkları
# - Daha karmaşık
# - Daha fazla veri gerektirir
```

## LSTM Teorisi

### Neden LSTM?

**Vanishing Gradient Problemi**:
```
Basit RNN: Uzun sekanslar → Gradient sıfıra gider
LSTM: Gated structure → Gradientler korunur
```

**LSTM Kapıları**:
```
1. Forget Gate: Hangi bilgileri unutalım?
2. Input Gate: Hangi yeni bilgileri ekleyelim?
3. Output Gate: Hangi bilgileri çıktı olarak verelim?
4. Cell State: Uzun mesafe memory
```

## Referanslar

- **LSTM**: [Hochreiter & Schmidhuber, 1997](http://www.bioinf.jku.at/publications/older/2604.pdf)
- **Character-Level Models**: [Karpathy et al., 2015](https://arxiv.org/abs/1506.02078)
- **BERTScore**: [Zhang et al., 2019](https://arxiv.org/abs/1904.09675)
- **Turkish BERT**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

---

**Not**: Bu proje araştırma ve eğitim amaçlıdır. LSTM yaklaşımı, modern pre-trained modellere (GPT-2, BERT, Gemini) kıyasla daha düşük kaliteli sonuçlar üretir ancak küçük veri ile sıfırdan model eğitimi için iyi bir örnektir.
