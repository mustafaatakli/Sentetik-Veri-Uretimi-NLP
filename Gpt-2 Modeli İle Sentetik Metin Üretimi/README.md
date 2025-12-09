# GPT-2 Modeli İle Sentetik Metin Üretimi

Bu proje, Türkçe GPT-2 (Generative Pre-trained Transformer 2) modelini kullanarak teknoloji haber başlıklarından sentetik metin üretimi gerçekleştirmektedir. Causal Language Modeling (CLM) yaklaşımı ile doğal ve çeşitli cümleler üretilmektedir.

## Proje Açıklaması

Proje, 100 adet gerçek teknoloji haber başlığından yola çıkarak, GPT-2 Türkçe modeli ile 3000 adet sentetik haber başlığı üretmektedir. Üretilen veriler, BERT tabanlı metriklerle değerlendirilmekte ve diğer yöntemlerle (BERT MLM, Gemini API, LSTM) karşılaştırılabilir formatta raporlanmaktadır.

## Ana Özellikler

### 1. GPT-2 Türkçe Modeli
- **Model**: `ytu-ce-cosmos/turkish-gpt2`
- **Parametre Sayısı**: ~124 milyon parametre
- **Üretim Yöntemi**: Causal Language Modeling (CLM)
- **GPU Desteği**: CUDA uyumlu GPU'larda otomatik hızlandırma
- **Batch Generation**: 10 cümle aynı anda üretim (5x hızlı)

### 2. Gelişmiş Generation Stratejisi

#### Temperature Sampling
```python
temperature = random.uniform(1.0, 1.5)  # Yüksek çeşitlilik
```
- Yüksek temperature (1.0-1.5) → Daha yaratıcı/çeşitli kelime seçimleri
- Her üretimde rastgele temperature → Maksimum varyasyon

#### Top-K Sampling
```python
top_k = random.randint(40, 80)  # En iyi 40-80 kelime arasından seçim
```
- Düşük kaliteli kelimeleri filtreler
- Geniş aralık (40-80) → Dengeli çeşitlilik

#### Top-P (Nucleus) Sampling
```python
top_p = random.uniform(0.90, 0.98)  # Kümülatif olasılık eşiği
```
- Dinamik kelime seçimi
- Yüksek kalite ve çeşitlilik dengesi

#### No Repeat N-gram
```python
no_repeat_ngram_size=2  # 2-gram tekrarlarını engelle
```
- Tekrarlayan kelimeleri önler
- Daha doğal cümle yapısı

### 3. Kapsamlı Temizleme Pipeline'ı

GPT-2'nin ürettiği metinler çok katmanlı temizleme sürecinden geçer:

#### A. Yapısal Temizleme
```python
# 1. Sadece ilk cümle alınır
generated = generated.split('\n')[0].strip()

# 2. Pipe karakterinden sonrası kaldırılır (kategori bilgisi)
if '|' in generated:
    generated = generated.split('|')[0].strip()
```

#### B. Site/Kaynak Bilgisi Temizleme
```python
# 3. Tire ve özel karakterlerden sonrası kaldırılır
# Desteklenen: - – — » « • | :
generated = re.split(r'\s+[-–—»«•|:]\s+', generated)[0].strip()

# 4. Link/site isimleri kaldırılır
generated = re.sub(r' - \w+\.\w+', '', generated)  # " - Site.com"
generated = re.sub(r'https?://\S+', '', generated)  # HTTP linkler
generated = re.sub(r'www\.\S+', '', generated)      # www linkler
```

#### C. Metadata Temizleme
```python
# 5. Tarih/saat formatları kaldırılır
generated = re.sub(r'\d{1,2}\.\d{1,2}\.\d{4}', '', generated)
generated = re.sub(r'\d{1,2}:\d{2}', '', generated)

# 6. Parantez içi kaynak bilgileri kaldırılır
generated = re.sub(r'\([^)]*\)', '', generated)
generated = re.sub(r'\[[^\]]*\]', '', generated)
```

#### D. Final Temizleme
```python
# 7. Fazla noktalama/boşluk temizlenir
generated = re.sub(r'\s*-\s*$', '', generated)
generated = re.sub(r'\s+', ' ', generated).strip()
generated = re.sub(r'[.!?]+$', '', generated).strip()
```

### 4. Batch Optimization

#### Generation Optimization
```python
num_return_sequences=10  # 10 cümle aynı anda üretim
max_new_tokens=50        # Token limiti (max_length yerine daha hızlı)
```
- **5-10x Hız Artışı**: Batch processing sayesinde
- **GPU Kullanımı**: Optimum GPU memory kullanımı

#### Perplexity Optimization
```python
batch_size=16  # 16 cümle/batch BERT perplexity hesaplaması
```
- Hızlı kalite değerlendirmesi
- Verimli GPU kullanımı

### 5. Kalite Kontrol

#### A. Geçerlilik Kontrolleri
- Minimum uzunluk: 10 karakter
- Kelime sayısı: 5-20 kelime (haber başlığı formatı)
- Orijinalden farklılık: En az 2 kelime
- Tekil cümle kontrolü

#### B. Farklılık Kontrolü
```python
# Orijinalden en az 2 kelime farklı olmalı
orig_kelimeler = set(cumle.lower().split())
gen_kelimeler = set(generated.lower().split())
fark_sayisi = len(orig_kelimeler.symmetric_difference(gen_kelimeler))

if fark_sayisi >= 2:
    varyantlar.add(generated)
```

#### C. BERT Perplexity (Raporlama Amaçlı)
- **Not**: GPT-2 için filtreleme YAPILMAZ
- Adil karşılaştırma için 3000 cümle garanti edilir
- Perplexity skorları sadece kalite raporu için hesaplanır

### 6. 5 Temel Metrik Analizi

Diğer yöntemlerle (BERT MLM, Gemini, LSTM) karşılaştırılabilir standart metrikler:

#### [1] Tekil Oran (Uniqueness)
- Her cümlenin benzersizliği
- İdeal: ≥ %95

#### [2] BERTScore F1 (Anlamsal Benzerlik)
- BERT embeddings ile anlamsal benzerlik
- Precision, Recall, F1 skorları
- İdeal: ≥ 0.75

#### [3] Kelime Kapsama (Vocabulary Coverage)
- Orijinal kelimelerin korunma oranı
- İdeal: ≥ %80

#### [4] Benzerlik Skoru (TF-IDF Cosine Similarity)
- Kelime tabanlı benzerlik
- İdeal: 0.5 - 0.7 (dengeli)

#### [5] Perplexity Skoru (Anlamsal Doğallık)
- BERT MLM ile cümle doğallığı
- İdeal: ≤ 70

## Dosya Yapısı

```
Gpt-2 Modeli İle Sentetik Metin Üretimi/
│
├── gpt2_sentetik_uretim.py                          # Ana Python scripti
├── tekonoloji-haber-baslıkları.csv                  # Orijinal veri (100 haber başlığı)
├── gpt2_sentetik_teknoloji_haberleri_3000.csv       # Üretilen sentetik veri (3000 başlık)
└── gpt2-cıktılar.txt                                # Üretim süreci log dosyası
```

## Gereksinimler

```
torch
transformers
pandas
numpy
bert-score
scikit-learn
matplotlib
seaborn
tqdm
```

## Kurulum

```bash
pip install torch transformers pandas numpy bert-score scikit-learn matplotlib seaborn tqdm
```

## Kullanım

```bash
python gpt2_sentetik_uretim.py
```

### Çalışma Akışı

1. **GPU Kontrolü**: CUDA kullanılabilirliği kontrol edilir
2. **Veri Yükleme**: 100 orijinal cümle okunur
3. **GPT-2 Model Yükleme**: Türkçe GPT-2 modeli yüklenir (~124M parametre)
4. **Sentetik Üretim** (22-25 dakika):
   - Her cümleden ~35 varyant üretilir
   - Batch generation (10 cümle/batch)
   - Progress bar ile ilerleme
   - Her 20 cümlede bir durum raporu
5. **Tekrar Temizleme**: Çift cümleler çıkarılır
6. **Eksik Kontrolü**: Hedef sayıya ulaşılmazsa ek üretim
7. **GPU Memory Temizleme**: GPT-2 modeli bellekten çıkarılır
8. **BERT Yükleme**: Metrik hesaplamaları için BERT yüklenir
9. **Perplexity Hesaplama**: Batch mode ile hızlı hesaplama
10. **CSV Kayıt**: Sonuçlar kaydedilir
11. **5 Temel Metrik Analizi**: Kapsamlı değerlendirme

## Teknik Detaylar

### GPT-2 Generation Parametreleri

```python
model.generate(
    **inputs,
    max_new_tokens=50,              # Yeni üretilecek token sayısı
    temperature=1.0-1.5,            # Rastgele çeşitlilik (yüksek)
    top_k=40-80,                    # En iyi K kelime filtresi
    top_p=0.90-0.98,                # Nucleus sampling eşiği
    do_sample=True,                 # Sampling aktif
    num_return_sequences=10,        # Batch: 10 cümle aynı anda
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=2          # 2-gram tekrar engelleme
)
```

### Prompt Stratejisi

```python
# TÜM CÜMLE prompt olarak kullanılır (prefix yerine)
prefix = cumle  # "Google şov yapacak Google I/O 2025 canlı yayını"

# GPT-2 bu cümleyi devam ettirir veya yeniden yazar
# Sonuç: Yüksek kaliteli, bağlamsal olarak tutarlı varyantlar
```

### Memory Management

```python
# GPT-2 kullanımından sonra
del model, tokenizer
torch.cuda.empty_cache()

# BERT metrik hesaplamaları için yer açılır
```

## Performans

| Metrik | Değer |
|--------|-------|
| Üretim Süresi | ~22-25 dakika (100 → 3000 cümle) |
| Model Boyutu | ~124 milyon parametre |
| GPU Memory | ~2-3 GB (üretim), ~1.5 GB (BERT metrikleri) |
| Batch Size | 10 cümle/batch (generation) |
| Batch Size | 16 cümle/batch (perplexity) |
| Hızlanma | 5-10x (batch optimization sayesinde) |

## Örnek Çıktılar

### Orijinal Veri
```
Google şov yapacak Google I/O 2025 canlı yayını
12 taksitle alınabilecek en iyi akıllı telefonlar
Hastaneyi bileğe getiren saat Huawei Watch 5 çok farklı olmuş
```

### Üretilen Sentetik Örnekler (Log'dan)
```
[Script çalıştırıldığında örnekler görüntülenir]
```

## Model Karşılaştırması

### GPT-2 vs BERT MLM vs Gemini API

| Özellik | GPT-2 CLM | BERT MLM | Gemini API |
|---------|-----------|----------|------------|
| Üretim Yöntemi | Causal LM | Masked LM | LLM API |
| Doğallık | Yüksek | Orta-Yüksek | Çok Yüksek |
| Çeşitlilik | Çok Yüksek | Orta | Yüksek |
| Anlamsal Tutarlılık | Orta-Yüksek | Orta | Yüksek |
| Hız | Hızlı (Batch) | Hızlı (GPU) | Orta (API) |
| Maliyet | Ücretsiz | Ücretsiz | API Kotası |
| Kontrol | Orta (Parametreler) | Yüksek (Maskeleme) | Yüksek (Prompt) |
| Temizleme İhtiyacı | Yüksek | Düşük | Düşük |

### Avantajlar

✅ **Yüksek Çeşitlilik**: Temperature ve sampling parametreleri ile
✅ **Doğal Cümleler**: CLM ile akıcı metin üretimi
✅ **Hızlı**: Batch generation ile optimize edilmiş
✅ **Ücretsiz**: Tamamen açık kaynak
✅ **Kontrol**: Generation parametreleriyle ayarlanabilir

### Dezavantajlar

⚠️ **Temizleme Gereksinimi**: Üretilen metinler çok katmanlı temizleme gerektirir
⚠️ **Anlamsal Sapmalar**: Bazen bağlamdan uzaklaşabilir
⚠️ **GPU Memory**: ~2-3 GB GPU memory gerekir
⚠️ **Metadata Üretimi**: Site isimleri, kategoriler gibi istenmeyen bilgiler üretebilir

## Sınırlamalar

1. **Temizleme Bağımlılığı**: Kapsamlı regex temizleme gerekli
2. **Anlamsal Kontrol**: BERT MLM'e göre daha az anlamsal kontrol
3. **Bağlam Kayması**: Uzun üretimlerde bağlam kaybı riski
4. **Türkçe Model Sınırlamaları**: `ytu-ce-cosmos/turkish-gpt2` modeli sınırlı veri ile eğitilmiş
5. **Perplexity Filtreleme Yok**: Adil karşılaştırma için filtreleme yapılmaz

## İyileştirme Önerileri

### Generation Parametrelerini Ayarlama

```python
# Daha konservatif (daha tutarlı, az çeşitli)
temperature = random.uniform(0.7, 1.0)
top_k = random.randint(20, 40)
top_p = random.uniform(0.85, 0.92)

# Daha yaratıcı (daha çeşitli, riski yüksek)
temperature = random.uniform(1.3, 1.8)
top_k = random.randint(50, 100)
top_p = random.uniform(0.92, 0.99)
```

### Batch Size Optimizasyonu

```python
# Daha fazla GPU memory kullanımı için
batch_size = 15  # (varsayılan: 10)

# Daha az GPU memory için
batch_size = 5
```

### Temizleme Kurallarını Genişletme

```python
# Ek site/platform isimleri
generated = re.sub(r' - (Twitter|Facebook|Instagram)', '', generated)

# Ek özel karakterler
generated = re.split(r'\s+[→←↑↓•◆■□▪▫]\s+', generated)[0].strip()
```

## Değerlendirme Kriterleri

### İdeal Metrik Değerleri

| Metrik | İdeal Değer | GPT-2 Hedef |
|--------|-------------|-------------|
| Tekil Oran | ≥ %95 | ≥ %95 |
| BERTScore F1 | ≥ 0.85 | ≥ 0.75 |
| Kelime Kapsama | ≥ %90 | ≥ %80 |
| Benzerlik Skoru | 0.5 - 0.7 | 0.5 - 0.7 |
| Perplexity | ≤ 50 | ≤ 70 |

**Not**: GPT-2 için BERTScore ve Perplexity hedefleri biraz daha gevşek tutulmuştur çünkü CLM yaklaşımı daha çeşitli (ama bazen bağlamdan uzak) sonuçlar üretir.

### Başarı Değerlendirmesi

- **5/5 metrik**: ✓ MÜKEMMEL
- **4/5 metrik**: ✓ MÜKEMMEL
- **3/5 metrik**: ✓ İYİ - Kabul edilebilir kalite
- **2/5 metrik**: ~ ORTA - İyileştirme gerekebilir
- **<2 metrik**: ✗ ZAYIF - Ciddi iyileştirme gerekli

## Sorun Giderme

### GPU Memory Hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**:
```python
# Batch size'ı azaltın
batch_size = 5  # (varsayılan: 10)

# Veya max_new_tokens'ı azaltın
max_new_tokens = 30  # (varsayılan: 50)
```

### Model Yükleme Hatası
```
[HATA] Model yukleme hatasi
```
**Çözüm**:
- İnternet bağlantınızı kontrol edin
- Hugging Face'ten model manuel indirin
- Transformers kütüphanesini güncelleyin: `pip install --upgrade transformers`

### Düşük Kalite Üretim
```
[SONUC] ✗ ZAYIF - Ciddi iyilestirme gerekli
```
**Çözüm**:
- Temperature değerlerini düşürün (0.8-1.2 arası)
- Top-K değerini azaltın (30-50 arası)
- Daha fazla seed cümle kullanın
- Temizleme kurallarını genişletin

## GPU Memory Profiling

```python
# Üretim öncesi memory
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# GPT-2 ile üretim: ~2-3 GB
# BERT metrikleri: ~1.5 GB
# Toplam peak: ~4 GB (GTX 1660 ve üzeri önerilir)
```

## Alternatif Türkçe GPT-2 Modelleri

```python
# Daha küçük model (daha hızlı, az kaliteli)
model_name = "redrussianarmy/gpt2-turkish-cased"

# Daha büyük model (daha yavaş, yüksek kaliteli)
model_name = "dbmdz/turkish-gpt2-large"  # (eğer varsa)
```

## Referanslar

- **GPT-2**: [Radford et al., 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Turkish GPT-2**: [ytu-ce-cosmos/turkish-gpt2](https://huggingface.co/ytu-ce-cosmos/turkish-gpt2)
- **BERTScore**: [Zhang et al., 2019](https://arxiv.org/abs/1904.09675)
- **Turkish BERT**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

---

**Not**: Bu proje araştırma ve eğitim amaçlıdır. Üretilen sentetik veriler gerçek haber içeriği değildir. GPT-2 modeli Hugging Face üzerinden kullanılmaktadır.
