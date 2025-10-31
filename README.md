# 🤖 Türkçe Sentetik Veri Üretimi ve Duygu Analizi: Karşılaştırmalı Çalışma

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

Bu proje, **Türkçe metinler** için sentetik veri üretimi ve duygu analizi üzerine **dört farklı derin öğrenme yaklaşımının** karşılaştırmalı analizini sunar. Proje kapsamında **LSTM**, **BiLSTM**, **GAN** ve **BERT** modelleri kullanılarak hem veri üretimi hem de duygu sınıflandırması gerçekleştirilmiş ve sonuçları **Google Gemini AI** ile karşılaştırılmıştır.

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Yöntemler ve Sonuçlar](#-yöntemler-ve-sonuçlar)
- [Karşılaştırmalı Analiz](#-karşılaştırmalı-analiz)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Performans Metrikleri](#-performans-metrikleri)
- [Literatür Araştırması](#-literatür-araştırması)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

Bu araştırma projesi, **elektrikli arabalar** konusunda Türkçe sentetik veri üretimi ve duygu analizi için dört farklı derin öğrenme yaklaşımını kapsamlı bir şekilde incelemektedir:

### 🔬 Araştırma Soruları
1. Hangi model Türkçe metinler için en yüksek duygu analizi doğruluğunu sağlar?
2. GAN tabanlı sentetik metin üretimi ne kadar özgün ve kaliteli sonuçlar verir?
3. BERT ve LSTM modellerinin performans-kaynak dengesi nasıldır?
4. Gemini AI ile geleneksel modeller arasındaki uyuşma oranı nedir?

### 🎓 Kullanım Alanları
- Doğal Dil İşleme (NLP) araştırmaları
- Sentetik veri üretimi için benchmark çalışmaları
- Türkçe duygu analizi model karşılaştırmaları
- Eğitim ve akademik projeler
- Literatür taraması ve state-of-the-art teknik incelemesi

---

## 🚀 Yöntemler ve Sonuçlar

### 1️⃣ **BERT ile Duygu Analizi**
📂 Klasör: `/bert-sentiment-analysis/`

**Özellikler:**
- `dbmdz/bert-base-turkish-cased` modeli kullanımı
- 3 sınıflı duygu analizi (Pozitif, Negatif, Nötr)
- Fine-tuning ile Türkçe'ye özelleştirilmiş model
- Gemini AI ile karşılaştırmalı değerlendirme

**Performans:**
- ✅ **Test Doğruluğu:** %92.6
- ✅ **Gemini Uyuşma:** %92.3
- ✅ **En İyi Sınıf:** Negatif (%98)
- ⚠️ **Zayıf Nokta:** Nötr sınıf (%84)

**Kullanılan Teknolojiler:**
- Transformers (Hugging Face)
- PyTorch
- pandas, scikit-learn

---

### 2️⃣ **BiLSTM + BERT Sequence Embedding**
📂 Klasör: `/bilstm-bert-hybrid/`

**Özellikler:**
- BERT sequence embeddings (768-boyutlu vektörler)
- 2 katmanlı Bidirectional LSTM
- Kelime sırasını koruyan hibrit mimari
- GPU optimize edilmiş eğitim

**Performans:**
- ✅ **Test Doğruluğu:** %89-92
- ✅ **Sınıf Bazında F1-Score:** ~%90
- ✅ **BERT Dense Layer'dan %2-4 daha iyi**
- ✅ **Gemini ile yüksek uyuşma**

**Model Mimarisi:**
```
BERT Embedding (64 × 768)
    ↓
Bidirectional LSTM (128 units)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dense + Dropout
    ↓
Softmax (3 sınıf)
```

---

### 3️⃣ **LSTM Duygu Analizi**
📂 Klasör: `/lstm-sentiment/`

**Özellikler:**
- Saf Bidirectional LSTM mimarisi
- Hafif ve hızlı model
- Early stopping ve learning rate scheduling
- Gemini ile detaylı karşılaştırma

**Performans:**
- ✅ **Test Doğruluğu:** %86.8
- ✅ **Gemini Uyuşma:** %87.5
- ✅ **Eğitim Süresi:** ~10 dakika (BERT'ten hızlı)
- ✅ **Model Boyutu:** 1-2M parametre (BERT: 110M)
- ✅ **Bellek Kullanımı:** 2-3 GB (BERT: 6-8 GB)

**Öne Çıkanlar:**
- En hafif ve hızlı model
- Kaynak kısıtlı ortamlar için ideal
- Makul performans/verimlilik dengesi

---

### 4️⃣ **GAN ile Sentetik Metin Üretimi**
📂 Klasör: `/gan-text-generation/`

**Özellikler:**
- LSTM tabanlı Generator ve Discriminator
- Türkçe Vikipedi verileriyle eğitim
- Cosine similarity ile benzersizlik kontrolü
- Kalite skorlama sistemi

**Veri Üretimi Performansı:**
- ✅ **Üretilen Cümle:** 1000+ özgün cümle
- ✅ **Ortalama Kelime:** 7-8 kelime/cümle
- ✅ **Kalite Skoru:** 0.688/1.0
- ✅ **Benzerlik Kontrolü:** %77 özgünlük

**Kullanım Alanları:**
- Veri augmentation
- Eğitim veri seti genişletme
- Sentetik benchmark veri setleri

---

### 5️⃣ **Gemini ile Veri Seti Oluşturma ve Duygu Analizi**
📂 Klasör: `/gemini-dataset-generation/`

**Özellikler:**
- Google Gemini 2.5 Flash API kullanımı
- Batch generation (100 cümle/istek)
- Dual hybrid kalite skorlama (Faktörel + Perplexity)
- Semantik benzerlik filtresi

**Üretim Metrikleri:**
- ✅ **Üretilen Cümle:** 1000 adet
- ✅ **API İsteği:** 42 batch
- ✅ **Süre:** 50.5 dakika
- ✅ **Kalite Skoru:** 0.688 ortalama
- ✅ **Sentiment Dağılımı:** %40 pozitif, %40 nötr, %20 negatif

**Perplexity Modeli:**
- `ytu-ce-cosmos/turkish-gpt2` ile doğallık kontrolü

---

## 📊 Karşılaştırmalı Analiz

### 🏆 Model Performansları (Test Seti)

| Model | Accuracy | Precision | Recall | F1-Score | Gemini Uyuşma |
|-------|----------|-----------|--------|----------|---------------|
| **BERT** | **%92.6** 🥇 | 0.926 | 0.926 | 0.926 | **%92.3** 🥇 |
| **BiLSTM+BERT** | %89-92 🥈 | ~0.90 | ~0.90 | ~0.90 | Yüksek |
| **LSTM** | %86.8 🥉 | 0.870 | 0.868 | 0.867 | %87.5 |

### ⚡ Verimlilik Karşılaştırması

| Metrik | BERT | BiLSTM+BERT | LSTM |
|--------|------|-------------|------|
| **Eğitim Süresi** | ~15-20 dk | ~12-15 dk | **~10 dk** ✅ |
| **Model Boyutu** | 110M param | ~60M param | **1-2M param** ✅ |
| **Bellek (GPU)** | 6-8 GB | 4-6 GB | **2-3 GB** ✅ |
| **Çıkarım Hızı** | Yavaş | Orta | **Hızlı** ✅ |

### 🎯 Sınıf Bazında Performans

#### **Negatif Sınıf** (En Başarılı)
- BERT: **%98** 🏆
- BiLSTM+BERT: ~%95
- LSTM: %93

#### **Pozitif Sınıf**
- BERT: **%94** 🏆
- BiLSTM+BERT: ~%92
- LSTM: %90

#### **Nötr Sınıf** ⚠️ (Tüm Modellerde Zayıf)
- BERT: **%84** 🏆
- BiLSTM+BERT: ~%82
- LSTM: %75

### 🔍 Temel Bulgular

1. **BERT En Yüksek Doğruluk**: %92.6 ile en iyi performans
2. **LSTM En Verimli**: En az kaynak, en hızlı eğitim
3. **BiLSTM+BERT İyi Denge**: Performans-verimlilik dengesi
4. **Nötr Sınıf Zorlu**: Tüm modellerde iyileştirme gerekli
5. **Gemini Tutarlılık Yüksek**: %87-92 arası uyuşma
6. **GAN Başarılı Üretim**: 1000+ özgün Türkçe cümle

### 🤔 Hangi Modeli Seçmeli?

| Senaryo | Önerilen Model | Neden? |
|---------|----------------|--------|
| **Maksimum Doğruluk** | BERT | En yüksek accuracy (%92.6) |
| **Mobil/Embedded** | LSTM | En hafif model (1-2M param) |
| **Dengeli Çözüm** | BiLSTM+BERT | İyi performans + kabul edilebilir kaynak |
| **Gerçek Zamanlı** | LSTM | En hızlı çıkarım süresi |
| **Veri Üretimi** | GAN + Gemini | Özgün ve kaliteli sentetik veri |

---

## 📁 Proje Yapısı

```
synthetic-data-generation-nlp/
│
├── README.md                                      # Ana dokümantasyon (bu dosya)
├── LICENSE.md                                     # Lisans bilgisi
│
├── bert-sentiment-analysis/                       # BERT Duygu Analizi
│   ├── main.py
│   ├── README.md
│   ├── egitim-veriseti-5k.xlsx
│   ├── bert_vs_gemini_sonuc_1k.xlsx
│   └── etiketsiz-test-gemini-etiketlenmis-1k.xlsx
│
├── bilstm-bert-hybrid/                            # BiLSTM + BERT Hibrit
│   ├── main.py
│   ├── README.md
│   ├── egitim-veriseti.xlsx
│   ├── etiketsiz-test-gemini-etiketlenmis.xlsx
│   └── bert_vs_gemini_sonuc.xlsx
│
├── lstm-sentiment/                                # LSTM Duygu Analizi
│   ├── main.py
│   ├── README.md
│   ├── egitim-veriseti-5k.xlsx
│   ├── lstm_vs_gemini_sonuc_1k.xlsx
│   └── etiketsiz-test-gemini-etiketlenmis-1k.xlsx
│
├── gan-text-generation/                           # GAN Metin Üretimi
│   ├── main.py
│   ├── README.md
│   ├── sentences.txt                              # 5000 cümle (eğitim)
│   ├── wiki.tr.txt                                # Tam veri seti
│   ├── uretilen_cumleler.csv
│   └── training_history.png
│
├── gemini-dataset-generation/                     # Gemini Veri Üretimi
│   ├── main10.py
│   ├── README.md
│   ├── main10.pdf
│   └── elektrikli_araba_1000_batch.xlsx
│
└── Literatürdeki Sentetik Veri Üretimi İle İlgili Makaleler/
    │                                              # 📚 Literatür Araştırması
    ├── metin/                                     # Metin tabanlı sentetik veri
    │   ├── Genel(arxiv.org vb.)/                                 # Genel çalışmalar
    │   └── ScienceDirect & IEEE Xplore/           # Akademik veritabanları
    │
    ├── görüntü/                                   # Görüntü tabanlı sentetik veri
    │
    └── ses/                                       # Ses tabanlı sentetik veri
```

---

## 🛠️ Kurulum

### Sistem Gereksinimleri

**Donanım:**
- GPU: NVIDIA GPU (önerilen: Tesla T4 veya üzeri)
- RAM: Minimum 8GB (önerilen: 16GB+)
- Depolama: ~5GB (tüm modeller için)

**Yazılım:**
- Python 3.8+
- CUDA 11.x (GPU için)
- pip veya conda

### Temel Kütüphaneler

```bash
# Tüm projeler için ortak
pip install pandas numpy openpyxl scikit-learn

# BERT projeleri için
pip install torch transformers

# LSTM projeleri için
pip install tensorflow

# GAN projesi için
pip install torch sentence-transformers

# Gemini projesi için
pip install google-generativeai
```

### Proje Bazında Kurulum

Her alt klasördeki README.md dosyasında detaylı kurulum talimatları mevcuttur.

---

## 🚀 Kullanım

### Hızlı Başlangıç

1. **Repo'yu Klonlayın**
```bash
git clone https://github.com/kullanici-adi/synthetic-data-generation.git
cd synthetic-data-generation
```

2. **İlgilendiğiniz Projeye Gidin**
```bash
cd bert-sentiment-analysis/  # veya lstm-sentiment, gan-text-generation vb.
```

3. **README Talimatlarını Takip Edin**
Her klasördeki README.md dosyası, o projeye özel kurulum ve çalıştırma adımlarını içerir.

### Kaggle/Colab Kullanımı

Tüm projeler **GPU destekli** ortamlarda en iyi performansı gösterir:

1. Kaggle Notebook veya Google Colab açın
2. GPU T4 x2 hızlandırıcıyı aktif edin
3. İlgili veri setlerini yükleyin
4. `main.py` kodunu çalıştırın

---

## 📈 Performans Metrikleri

### Accuracy (Doğruluk) Karşılaştırması

```
█████████████████████████████████████████████████ 92.6% BERT
████████████████████████████████████████████      91.0% BiLSTM+BERT (ortalama)
█████████████████████████████████████████         86.8% LSTM
```

### Model Boyutu Karşılaştırması

```
████████████████████████████████████████████████████████████ 110M BERT
██████████████████████████████                               60M BiLSTM+BERT
█                                                            1.5M LSTM
```

### Eğitim Süresi (GPU T4 x2)

```
████████████████████ 20 dk BERT
███████████████      15 dk BiLSTM+BERT
██████████           10 dk LSTM
```

---

## 🎓 Öğrenilen Dersler

### ✅ Başarılar

1. **BERT Çok Güçlü**: Transfer learning ile Türkçe'de mükemmel sonuçlar
2. **LSTM Hala Değerli**: Kaynak kısıtlı senaryolar için hızlı ve etkili
3. **Hibrit Yaklaşım İyi**: BiLSTM+BERT denge noktası sağlıyor
4. **GAN Çalışıyor**: Türkçe metin üretimi için uygulanabilir
5. **Gemini Güvenilir**: Etiketleme ve karşılaştırma için tutarlı

### ⚠️ Zorluklar

1. **Nötr Sınıf Zor**: Tüm modellerde en düşük performans
2. **Pozitif-Nötr Karışımı**: Model ve Gemini arasında en çok burada farklılık
3. **GAN Eğitimi Hassas**: Hyperparameter tuning kritik
4. **Kaynak Yoğunluğu**: BERT modelleri büyük GPU bellek gerektiriyor
5. **Türkçe Veri Kıtlığı**: Kaliteli etiketli veri bulmak zorlu

### 🔮 Gelecek İyileştirmeler

- [ ] Nötr sınıf için özel model eğitimi
- [ ] Daha büyük veri setleri (10K+ cümle)
- [ ] Transformer-XL gibi yeni mimariler
- [ ] Multi-task learning yaklaşımları
- [ ] Ensemble modelleme (BERT + LSTM)
- [ ] Fine-tuned Türkçe GPT modelleri

---

## 📚 Literatür Araştırması

Bu proje, sentetik veri üretimi konusunda kapsamlı bir literatür taraması içermektedir. **Literatürdeki Sentetik Veri Üretimi İle İlgili Makaleler** klasöründe, farklı veri türleri için akademik çalışmalar kategorize edilmiştir:

### 📝 Metin Tabanlı Sentetik Veri
- **Genel**: Çeşitli kaynaklardan derlenen genel çalışmalar
- **ScienceDirect & IEEE Xplore**: Akademik veritabanlarından seçilmiş makaleler
- GAN, LSTM, BERT ve transformer tabanlı metin üretimi çalışmaları
- Türkçe ve çok dilli sentetik veri üretimi yaklaşımları

### 🖼️ Görüntü Tabanlı Sentetik Veri
- Image generation için GAN, VAE ve Diffusion modelleri
- Sentetik görüntü kalite değerlendirme metrikleri
- Computer vision uygulamaları için veri augmentation

### 🔊 Ses Tabanlı Sentetik Veri
- TTS (Text-to-Speech) sistemleri
- Ses senteziyle veri augmentation
- Konuşma tanıma sistemleri için sentetik veri

> **Not**: Bu klasör, projenin teorik temelini oluşturan kaynaklardan oluşmaktadır ve araştırmacılar için referans niteliğindedir.

---

## 📚 Referanslar ve Kaynaklar

### Modeller

- **BERT**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- **Turkish GPT-2**: [ytu-ce-cosmos/turkish-gpt2](https://huggingface.co/ytu-ce-cosmos/turkish-gpt2)
- **Gemini AI**: [Google DeepMind](https://deepmind.google/technologies/gemini/)

### Veri Setleri

- **Türkçe Vikipedi**: [Turkish Sentences Dataset](https://www.kaggle.com/datasets/mahdinamidamirchi/turkish-sentences-dataset)

### Kütüphaneler

- **Transformers**: [Hugging Face](https://huggingface.co/docs/transformers/)
- **TensorFlow**: [tensorflow.org](https://www.tensorflow.org/)
- **PyTorch**: [pytorch.org](https://pytorch.org/)

---

## 🤝 Katkıda Bulunma

Bu proje şu anda **kapalı kaynak** olup, katkılar kabul edilmemektedir. Ancak:

- 🐛 Bug bildirimleri için Issue açabilirsiniz
- 💡 Öneri ve geri bildirimlerinizi paylaşabilirsiniz
- ⭐ Projeyi beğendiyseniz yıldız verebilirsiniz

---

## 📧 İletişim

Proje hakkında sorularınız için:

- **Geliştirici**: Mustafa Ataklı
- **GitHub**: [github.com/kullanici-adi](https://github.com/kullanici-adi)
- **Email**: [email@example.com](mailto:email@example.com)

---

## 📄 Lisans

```
Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.

İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.
```

---

## ⭐ Yıldız Vermeyi Unutmayın!

Bu projeyi faydalı bulduysanız, GitHub'da ⭐ vererek destek olabilirsiniz!

### 🏆 Proje İstatistikleri

- **Toplam Model**: 4 farklı yaklaşım
- **Toplam Veri**: 10K+ etiketli cümle
- **Toplam Kod**: 2000+ satır Python
- **Geliştirme Süresi**: 3+ ay
- **GPU Saati**: 100+ saat
- **Literatür**: 3 kategori (Metin, Görüntü, Ses)
- **Akademik Kaynak**: ScienceDirect & IEEE Xplore