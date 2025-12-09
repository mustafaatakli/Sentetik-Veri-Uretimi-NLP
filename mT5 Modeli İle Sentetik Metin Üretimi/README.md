# mT5 Modeli İle Sentetik Metin Üretimi

Bu proje, Google'ın mT5 (multilingual T5) ve Türkçe'ye özelleştirilmiş T5 modellerini kullanarak teknoloji haber başlıklarından sentetik metin üretimi gerçekleştirmektedir. Text-to-Text Transfer Transformer (T5) mimarisi ile paraphrase ve rewrite görevleri üzerinden üretim yapılmaktadır.

## Proje Açıklaması

Proje, 100 adet gerçek teknoloji haber başlığından yola çıkarak, T5 modelleri ile 3000 adet sentetik haber başlığı üretmektedir. İki farklı model kullanılmıştır:
1. **google/mt5-base**: Genel amaçlı çok dilli T5
2. **Turkish-NLP/t5-efficient-base-turkish**: Türkçe'ye özel optimize edilmiş T5

Üretilen veriler, diğer yöntemlerle (BERT, GPT-2, Gemini, LSTM) karşılaştırılabilir standart metriklerle değerlendirilmektedir.

## Ana Özellikler

### 1. T5 (Text-to-Text Transfer Transformer) Modeli

#### Model Mimarisi
T5, tüm NLP görevlerini text-to-text formatına dönüştürür:
```
Input:  "paraphrase: Google şov yapacak Google I/O 2025 canlı yayını"
Output: "Google I/O 2025 etkinliği canlı yayınlanacak"

Input:  "rewrite: 12 taksitle alınabilecek en iyi akıllı telefonlar"
Output: "Taksitle satın alınabilecek en iyi telefonlar"
```

#### Kullanılan Modeller

**1. google/mt5-base**
- **Dil**: 101 dil (çok dilli)
- **Parametre**: ~580 milyon
- **Eğitim**: mC4 dataset (101 dil)
- **Avantaj**: Genel amaçlı, çok dilli
- **Dezavantaj**: Türkçe'ye özel değil

**2. Turkish-NLP/t5-efficient-base-turkish**
- **Dil**: Türkçe odaklı
- **Parametre**: ~220 milyon (efficient)
- **Eğitim**: Türkçe corpus
- **Avantaj**: Türkçe'ye optimize, daha küçük
- **Dezavantaj**: Sadece Türkçe

### 2. Çoklu Prompt Stratejileri

T5 modeli farklı görev tanımları ile kullanılır:

```python
strategies = [
    f"paraphrase: {prompt_text}",      # Parafraz üretimi
    f"generate similar: {prompt_text}", # Benzer cümle üretimi
    f"rewrite: {prompt_text}",         # Yeniden yazma
    f"{prompt_text}",                  # Prefix olmadan (serbest)
]

# Her üretimde rastgele bir strateji seçilir
input_text = random.choice(strategies)
```

### 3. Generation Parametreleri

#### Temperature Sampling
```python
temperature = random.uniform(0.7, 1.1)  # Kontrollü çeşitlilik
# Daha düşük → Daha anlamlı ve tutarlı
# Daha yüksek → Daha çeşitli ama riskli
```

#### Top-K Sampling
```python
top_k = random.randint(30, 60)  # Sınırlı kelime havuzu
# Düşük kaliteli kelimeleri filtreler
```

#### Top-P (Nucleus) Sampling
```python
top_p = random.uniform(0.85, 0.95)  # Kümülatif olasılık eşiği
# Daha konsantre seçim
```

#### Tekrar Önleme
```python
no_repeat_ngram_size = 3        # 3-gram tekrarlarını engelle
repetition_penalty = 1.2        # Tekrar edilen kelimelere ceza
early_stopping = True           # EOS token'da dur
```

### 4. Agresif Temizleme Pipeline'ı

T5 modelleri bazen istenmeyen çıktılar üretebilir. 15 katmanlı temizleme süreci:

#### A. Temel Temizleme
```python
# 1. Sadece ilk satır
generated = generated.split('\n')[0].strip()

# 2. T5 özel token'larını kaldır
generated = re.sub(r'<[^>]+>', '', generated)  # <extra_id_0>
generated = re.sub(r'\[UNK\]', '', generated)
```

#### B. Dil Filtreleme
```python
# 3. Kiril alfabesi (Rusça) kontrolü
if re.search(r'[а-яА-ЯёЁ]', generated):
    continue

# 4. Yunanca kontrolü
if re.search(r'[α-ωΑ-Ω]', generated):
    continue

# 5. Çince kontrolü
if re.search(r'[\u4e00-\u9fff]', generated):
    continue
```

#### C. Metadata Temizleme
```python
# 6. Özel ayırıcılardan sonrasını kaldır
generated = re.split(r'\s+[-–—»«•|:]\s+', generated)[0].strip()

# 7. Link/site isimleri
generated = re.sub(r'https?://\S+', '', generated)
generated = re.sub(r'\w+\.com', '', generated)

# 8. Tarih/saat formatları
generated = re.sub(r'\d{1,2}\.\d{1,2}\.\d{4}', '', generated)

# 9. Parantez içi bilgiler
generated = re.sub(r'\([^)]*\)', '', generated)

# 10. Emoji ve özel semboller
generated = re.sub(r'[😀-🙏🌀-🗿🚀-🛿]', '', generated)
```

#### D. Türkçe Karakter Kontrolü
```python
# 11. En az %70 Türkçe karakter olmalı
turkce_karakterler = len(re.findall(r'[a-zA-ZçÇğĞıİöÖşŞüÜ]', generated))
toplam_karakterler = len(re.findall(r'\S', generated))

turkce_orani = turkce_karakterler / toplam_karakterler
if turkce_orani < 0.7:
    continue
```

#### E. Kalite Kontrolleri
```python
# 12. Kelime sayısı: 5-20 kelime
if not (5 <= word_count <= 20 and len(generated) >= 15):
    continue

# 13. Her kelime en az 2 harf
if not all(len(word) >= 2 for word in words):
    continue

# 14. Aynı kelime 3+ kez tekrarlanamaz
word_counts = Counter(words)
if any(count >= 3 for count in word_counts.values()):
    continue

# 15. Ardışık aynı kelime kontrolü
for i in range(len(words) - 1):
    if words[i].lower() == words[i+1].lower():
        continue  # "lar lar lar" gibi
```

### 5. Batch Generation

```python
num_return_sequences = 10  # Her seferde 10 varyant
max_new_tokens = 50        # Maksimum 50 yeni token

# Progress bar ile takip
pbar = tqdm(total=3000, desc="T5 Uretim")
```

### 6. 5 Temel Metrik Analizi

Diğer yöntemlerle standart karşılaştırma:

#### [1] Tekil Oran (Uniqueness)
- Her cümlenin benzersizliği
- İdeal: ≥ %95

#### [2] BERTScore F1 (Anlamsal Benzerlik)
- BERT embeddings ile anlamsal benzerlik
- İdeal: ≥ 0.70

#### [3] Kelime Kapsama (Vocabulary Coverage)
- Orijinal kelimelerin korunma oranı
- İdeal: ≥ %80

#### [4] Benzerlik Skoru (TF-IDF Cosine Similarity)
- Kelime tabanlı benzerlik
- İdeal: 0.50 - 0.75

#### [5] Perplexity Skoru (Anlamsal Doğallık)
- BERT MLM ile cümle doğallığı
- İdeal: ≤ 100

## Dosya Yapısı

```
mT5 Modeli İle Sentetik Metin Üretimi/
│
├── t5_sentetik_uretim.py                            # google/mt5-base modeli
├── t5_turkish_sentetik_uretim.py                    # Turkish-NLP modeli
├── tekonoloji-haber-baslıkları.csv                  # Orijinal veri (100 başlık)
├── t5_sentetik_teknoloji_haberleri_3000.csv         # mt5-base çıktısı
├── t5_turkish_sentetik_teknoloji_haberleri_3000.csv # Turkish-NLP çıktısı
└── t5-base-duz-model.txt                            # mt5-base log dosyası
```

## Gereksinimler

```
torch
transformers
pandas
numpy
scikit-learn
tqdm
bert-score
```

## Kurulum

```bash
pip install torch transformers pandas numpy scikit-learn tqdm bert-score
```

## Kullanım

### Google mT5-base Modeli
```bash
python t5_sentetik_uretim.py
```

### Turkish-NLP T5 Modeli (Önerilen)
```bash
python t5_turkish_sentetik_uretim.py
```

## Çalışma Akışı

1. **GPU Kontrolü**: CUDA kullanılabilirliği kontrol edilir
2. **Model Yükleme**:
   - mt5-base: ~580M parametre (~2.3 GB)
   - Turkish-NLP: ~220M parametre (~900 MB)
3. **Veri Yükleme**: 100 orijinal cümle okunur
4. **Sentetik Üretim** (1.5-2 saat):
   - Rastgele orijinal cümle seçimi
   - Rastgele prompt stratejisi
   - Batch generation (10 cümle/batch)
   - 15 katmanlı temizleme
   - Tekil kontrolü
   - Progress bar ile ilerleme
5. **CSV Kayıt**: Sonuçlar kaydedilir
6. **BERT Yükleme**: Metrik hesaplamaları için
7. **5 Temel Metrik Analizi**: Kapsamlı değerlendirme

## Teknik Detaylar

### T5 Encoder-Decoder Mimarisi

```
Input Sequence:
  "paraphrase: Google şov yapacak Google I/O 2025 canlı yayını"

Encoder:
  → Self-attention layers
  → Contextual representations

Decoder:
  → Cross-attention to encoder
  → Self-attention layers
  → Autoregressive generation

Output Sequence:
  "Google I/O 2025 etkinliği canlı yayınlanacak"
```

### Generation Process

```python
# 1. Tokenize input
inputs = tokenizer(input_text, return_tensors='pt').to(device)

# 2. Generate with sampling
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_return_sequences=10,
    temperature=0.9,
    top_k=50,
    top_p=0.90,
    do_sample=True,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2
)

# 3. Decode outputs
for output in outputs:
    text = tokenizer.decode(output, skip_special_tokens=True)
    # Temizleme ve filtreleme...
```

## Performans

### google/mt5-base

| Metrik | Değer |
|--------|-------|
| Üretim Süresi | ~1.5-2 saat |
| Model Boyutu | ~2.3 GB |
| Parametre Sayısı | ~580 milyon |
| GPU Memory | ~4-5 GB |
| Tekil Oran | %100 |
| BERTScore F1 | 0.46 (Düşük) |
| Kelime Kapsama | %50.81 (Düşük) |

### Turkish-NLP/t5-efficient-base-turkish

| Metrik | Değer |
|--------|-------|
| Üretim Süresi | ~1-1.5 saat |
| Model Boyutu | ~900 MB |
| Parametre Sayısı | ~220 milyon |
| GPU Memory | ~2-3 GB |
| Performans | mt5-base'den daha iyi (Türkçe için) |

## Model Karşılaştırması

### mT5 vs GPT-2 vs BERT vs Gemini vs LSTM

| Özellik | mT5 | GPT-2 | BERT MLM | Gemini | LSTM |
|---------|-----|-------|----------|--------|------|
| Mimari | Encoder-Decoder | Decoder-only | Encoder-only | LLM | RNN |
| Üretim Yöntemi | Seq2Seq | Causal LM | Masked LM | LLM API | Char-level |
| Doğallık | Orta | Yüksek | Orta-Yüksek | Çok Yüksek | Düşük |
| Çeşitlilik | Yüksek | Çok Yüksek | Orta | Yüksek | Orta |
| Anlamsal Tutarlılık | Orta | Orta-Yüksek | Orta | Yüksek | Düşük |
| Türkçe Kalitesi | Orta (mt5-base) | Orta | Yüksek | Yüksek | Düşük |
| Türkçe Kalitesi | Yüksek (Turkish-NLP) | - | - | - | - |
| Hız | Yavaş | Hızlı | Hızlı | Orta | Orta |
| Model Boyutu | ~900MB - 2.3GB | ~500MB | ~500MB | - | ~15MB |
| Temizleme İhtiyacı | Yüksek | Orta | Düşük | Düşük | Orta |

### Avantajlar

✅ **Encoder-Decoder**: Hem anlama hem üretme kapasitesi
✅ **Çok Dilli**: 101 dil desteği (mt5-base)
✅ **Task Flexibility**: Farklı görevler için kullanılabilir
✅ **Türkçe Özel**: Turkish-NLP modeli optimize edilmiş
✅ **Yüksek Çeşitlilik**: Farklı prompt stratejileri
✅ **Pre-trained**: Büyük veri ile eğitilmiş

### Dezavantajlar

⚠️ **Yavaş Üretim**: 1.5-2 saat (3000 cümle için)
⚠️ **Büyük Model**: 2.3 GB (mt5-base)
⚠️ **Yüksek Memory**: 4-5 GB GPU memory
⚠️ **Temizleme Gereksinimi**: 15 katmanlı temizleme
⚠️ **Düşük Kalite**: mt5-base Türkçe için optimize değil
⚠️ **Dil Karışması**: Bazen diğer dillere kayabilir

## Sınırlamalar

1. **mt5-base Türkçe Sorunu**: Çok dilli model Türkçe'de düşük performans
2. **Dil Kirliliği**: Bazen Rusça, Yunanca karakterler üretir
3. **Anlamsız Çıktılar**: Özel token'lar (<extra_id_X>)
4. **Yavaş Süreç**: Batch generation ile bile yavaş
5. **Yüksek Memory**: Büyük model boyutu
6. **Paraphrase Sınırlaması**: Bazen orijinale çok benzer

## İyileştirme Önerileri

### Model Seçimi
```python
# ✓ ÖNERİLEN: Türkçe'ye özel model
MODEL_NAME = 'Turkish-NLP/t5-efficient-base-turkish'

# ✗ ÖNERİLMEZ: Genel amaçlı (Türkçe'de zayıf)
MODEL_NAME = 'google/mt5-base'
```

### Generation Parametreleri
```python
# Daha anlamlı üretim için
temperature = random.uniform(0.5, 0.9)  # (varsayılan: 0.7-1.1)
top_k = random.randint(20, 40)         # (varsayılan: 30-60)
top_p = random.uniform(0.80, 0.90)     # (varsayılan: 0.85-0.95)

# Daha çeşitli üretim için
temperature = random.uniform(0.9, 1.3)
top_k = random.randint(50, 80)
top_p = random.uniform(0.90, 0.98)
```

### Batch Size Artırma
```python
num_return_sequences = 15  # (varsayılan: 10)
# Daha hızlı üretim ama daha fazla memory
```

### Prompt Optimizasyonu
```python
# Sadece etkili prompt'ları kullan
strategies = [
    f"paraphrase: {prompt_text}",
    f"rewrite: {prompt_text}",
]
# "generate similar" ve boş prefix'i çıkar
```

## Sorun Giderme

### Düşük BERTScore F1 (mt5-base)
```
BERTScore F1: 0.46 (Düşük)
```
**Çözüm**:
```python
# Turkish-NLP modelini kullan
MODEL_NAME = 'Turkish-NLP/t5-efficient-base-turkish'
```

### Dil Kirliliği (Rusça, Yunanca)
```
Üretilen: "технология новости Samsung"
```
**Çözüm**: Zaten kod içinde Kiril/Yunan alfabesi filtreleme var (çalışıyor)

### GPU Memory Hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**:
```python
# Batch size'ı azalt
num_return_sequences = 5  # (varsayılan: 10)

# Veya daha küçük model kullan
MODEL_NAME = 'Turkish-NLP/t5-efficient-base-turkish'  # 220M < 580M
```

### Çok Yavaş Üretim
```
T5 Uretim: 2%|█ | 60/3000 [10:00<8:20:00]
```
**Çözüm**:
```python
# Batch size artır (memory yeterse)
num_return_sequences = 15

# Veya hedef sayıyı azalt
TARGET_SENTENCES = 1000  # (varsayılan: 3000)
```

### Anlamsız Özel Token'lar
```
Üretilen: "teknoloji <extra_id_0> haberleri <extra_id_1>"
```
**Çözüm**: Zaten kod içinde temizleme var:
```python
generated = re.sub(r'<[^>]+>', '', generated)
```

## Gelişmiş Teknikler

### Fine-tuning T5 (İleri Seviye)

```python
# Kendi domain'inize fine-tune edin
from transformers import Trainer, TrainingArguments

# Dataset hazırlayın (paraphrase çiftleri)
train_dataset = [
    ("paraphrase: Orijinal cümle 1", "Parafraz 1"),
    ("paraphrase: Orijinal cümle 2", "Parafraz 2"),
    ...
]

# Fine-tune
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./t5-teknoloji-finetuned',
        num_train_epochs=3,
        per_device_train_batch_size=8
    ),
    train_dataset=train_dataset
)

trainer.train()
```

### Constrained Decoding

```python
# Sadece belirli kelimeleri kullan
from transformers import LogitsProcessor

class KeywordLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Teknoloji keyword'lerini tercih et
        return scores

model.generate(..., logits_processor=[KeywordLogitsProcessor()])
```

### Beam Search (Greedy Search Yerine)

```python
outputs = model.generate(
    **inputs,
    num_beams=5,           # Beam search (do_sample=False olmalı)
    num_return_sequences=5,
    early_stopping=True
)
# Daha tutarlı ama daha az çeşitli
```

## T5 Teorisi

### Text-to-Text Framework

T5, tüm NLP görevlerini text-to-text formatına dönüştürür:

```
Translation:       "translate English to German: Hello" → "Hallo"
Summarization:     "summarize: Long text..." → "Short summary"
Question Answering: "question: What is X?" → "Answer"
Paraphrase:        "paraphrase: Original" → "Paraphrased"
```

### Encoder-Decoder Advantage

```
BERT (Encoder-only):   ✓ Understanding  ✗ Generation
GPT-2 (Decoder-only):  ✗ Understanding  ✓ Generation
T5 (Encoder-Decoder):  ✓ Understanding  ✓ Generation
```

## Model Karşılaştırması Özeti

### Ne Zaman T5 Kullanılmalı?

✅ **Paraphrase/Rewrite Görevleri**: T5 bu görevler için tasarlanmış
✅ **Çok Dilli Uygulama**: mt5-base 101 dil destekler
✅ **Task Flexibility**: Farklı görevler için aynı model
✅ **Türkçe Özel Model Var**: Turkish-NLP optimize edilmiş

❌ **Hız Öncelikli**: GPT-2 veya BERT daha hızlı
❌ **Düşük Memory**: LSTM daha hafif
❌ **En Yüksek Kalite**: Gemini API daha iyi
❌ **Türkçe + Genel Model**: mt5-base Türkçe'de zayıf

## Referanslar

- **T5 Paper**: [Raffel et al., 2020 - Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- **mT5 Paper**: [Xue et al., 2021 - mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)
- **google/mt5-base**: [Hugging Face Model](https://huggingface.co/google/mt5-base)
- **Turkish-NLP T5**: [Hugging Face Model](https://huggingface.co/Turkish-NLP/t5-efficient-base-turkish)
- **BERTScore**: [Zhang et al., 2019](https://arxiv.org/abs/1904.09675)

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

---

**Not**: Bu proje araştırma ve eğitim amaçlıdır. mT5 modeli, Türkçe için Turkish-NLP/t5-efficient-base-turkish modeli ile kullanıldığında daha iyi sonuçlar verir. google/mt5-base çok dilli olduğu için Türkçe'de düşük performans göstermektedir.
