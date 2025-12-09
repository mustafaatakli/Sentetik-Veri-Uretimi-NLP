# BERT Modeli İle Sentetik Metin Üretimi

Bu proje, BERT (Bidirectional Encoder Representations from Transformers) Masked Language Model (MLM) kullanarak Türkçe teknoloji haber başlıklarından sentetik metin üretimi gerçekleştirmektedir.

## Proje Açıklaması

Proje, 100 adet gerçek teknoloji haber başlığından yola çıkarak, BERT tabanlı maskeleme ve tahmin yöntemiyle 3000 adet sentetik haber başlığı üretmektedir. Üretilen veriler, doğallık, çeşitlilik ve anlamsal kalite açısından detaylı metriklerle değerlendirilmektedir.

## Özellikler

### 1. BERT Masked Language Model (MLM)
- **Model**: `dbmdz/bert-base-turkish-cased`
- **GPU Desteği**: CUDA uyumlu GPU'larda otomatik hızlandırma
- **Konservatif Maskeleme**: Kısa cümlelerde 1, orta cümlelerde 2, uzun cümlelerde maksimum 3 kelime maskeleme
- **Temperature Sampling**: 1.2 temperature değeriyle çeşitlilik artırımı

### 2. Kalite Kontrol Mekanizmaları

#### A. Perplexity Filtreleme
- BERT MLM ile her cümlenin perplexity değeri hesaplanır
- Eşik değer: 50.0 (düşük = doğal, yüksek = anlamsız)
- Anlamsız cümleler otomatik olarak elenir

#### B. Trigram Çeşitlilik Kontrolü
- 3-kelimelik kalıplar izlenir
- Maksimum tekrar sayısı: 8
- Tekrarlayan yapıların önüne geçilir

#### C. Orijinal Cümle Kontrolü
- Üretilen cümleler orijinal veri seti ile karşılaştırılır
- Normalize edilmiş (küçük harf, noktalama temizlenmiş) formatta kontrol
- En az 2 kelime farklılık zorunluluğu

### 3. Kapsamlı Metrik Analizi

#### Çeşitlilik Metrikleri
- **Tekil Oran (Uniqueness)**: Benzersiz cümle oranı
- **Type-Token Ratio**: Kelime çeşitliliği oranı

#### Anlamsal Kalite Metrikleri
- **BERTScore**: BERT embeddings ile anlamsal benzerlik (Precision/Recall/F1)
- **TF-IDF Cosine Similarity**: Kelime tabanlı benzerlik
- **Perplexity**: BERT MLM ile doğallık ölçümü

#### Kapsam Metrikleri
- **Vocabulary Coverage**: Orijinal kelimelerin korunma oranı
- **Vocabulary Enrichment**: Yeni kelime zenginleştirmesi

### 4. Görselleştirme
- Kelime sayısı dağılımı
- Benzerlik dağılımı
- En sık kullanılan kelimeler
- Kelime çeşitliliği
- Özet metrikler

## Dosya Yapısı

```
BERT Modeli İle Sentetik Metin Üretimi/
│
├── temp.py                                          # Ana Python scripti
├── tekonoloji-haber-baslıkları.csv                  # Orijinal veri (100 haber başlığı)
├── sentetik_teknoloji_haberleri_3000.csv            # Üretilen sentetik veri (3000 başlık)
└── sentetik_veri_metrikleri.png                     # Görselleştirme çıktısı
```

## Gereksinimler

```
transformers
torch
pandas
tqdm
numpy
scikit-learn
matplotlib
seaborn
bert-score
```

## Kurulum

```bash
pip install transformers torch pandas tqdm numpy scikit-learn matplotlib seaborn bert-score
```

## Kullanım

```bash
python temp.py
```

### Çalışma Akışı

1. **Veri Yükleme**: `tekonoloji-haber-baslıkları.csv` dosyasından 100 cümle okunur
2. **GPU Kontrolü**: Varsa CUDA GPU kullanılır, yoksa CPU'da çalışır
3. **Model Yükleme**: Türkçe BERT modeli yüklenir
4. **Sentetik Üretim**: Her cümleden ~30 varyant üretilir
   - Konservatif maskeleme (1-3 kelime)
   - Temperature sampling ile çeşitlilik
   - Trigram kontrolü ile tekrar önleme
5. **Perplexity Filtreleme**: Anlamsız cümleler elenir (eşik: 50.0)
6. **Kayıt**: Sonuçlar CSV dosyasına kaydedilir
7. **Metrik Analizi**: Detaylı istatistiksel analiz ve değerlendirme
8. **Görselleştirme**: Grafiklerin oluşturulması

## Teknik Detaylar

### Maskeleme Stratejisi

```python
# Kısa cümle (≤4 kelime): 1 kelime maskele
# Orta cümle (5-7 kelime): Maksimum 2 kelime maskele
# Uzun cümle (>7 kelime): Maksimum 3 kelime maskele
```

### Temperature Sampling

Temperature = 1.2 ile daha çeşitli kelime seçimi:
- Yüksek temperature → Daha yaratıcı/çeşitli seçimler
- Düşük temperature → Daha muhafazakar/güvenli seçimler

### Top-K Filtreleme

Her maskelenmiş kelime için:
- En iyi 15 kelime seçilir (Top-K = 15)
- Olasılığa göre ağırlıklı rastgele seçim yapılır

## Örnek Çıktılar

### Orijinal Veri
```
Google şov yapacak Google I/O 2025 canlı yayını
12 taksitle alınabilecek en iyi akıllı telefonlar
Hastaneyi bileğe getiren saat Huawei Watch 5 çok farklı olmuş
```

### Üretilen Sentetik Örnekler
```
[Script çalıştırıldığında ilk 10 örnek görüntülenir]
```

## Performans

- **Üretim Hızı**: GPU kullanımıyla önemli ölçüde artar
- **Kalite**: Perplexity filtreleme ile yüksek kaliteli sonuçlar
- **Çeşitlilik**: Trigram kontrolü ile tekrarların minimizasyonu
- **Kapsam**: Orijinal veri setinin %90+ kelime kapsamı korunur

## Değerlendirme Kriterleri

### İdeal Metrik Değerleri

| Metrik | İdeal Değer | Açıklama |
|--------|-------------|----------|
| Tekil Oran | ≥ 0.95 | Benzersiz cümle oranı |
| Type-Token Ratio | ≥ 0.9 × Orijinal | Kelime çeşitliliği |
| BERTScore F1 | ≥ 0.85 | Anlamsal benzerlik |
| Cosine Similarity | 0.5 - 0.7 | Dengeli benzerlik |
| Perplexity | ≤ 50 | Doğal cümle yapısı |
| Vocabulary Coverage | ≥ 0.90 | Kelime kapsamı |
| Vocabulary Enrichment | ≥ 1.1× | Yeni kelime oranı |

## Avantajlar

1. **Yüksek Kalite**: Perplexity filtreleme ile doğal cümleler
2. **Çeşitlilik**: Trigram kontrolü ve temperature sampling
3. **Ölçeklenebilirlik**: GPU desteği ile hızlı üretim
4. **Şeffaflık**: Detaylı metrik analizi ve görselleştirme
5. **Türkçe Odaklı**: Türkçe BERT modeli kullanımı

## Sınırlamalar

1. Sentetik veriler orijinal verinin yapısına bağımlıdır
2. Çok kısa cümleler (<2 kelime) için sınırlı varyasyon
3. Anlam kayması riski (perplexity filtreleme ile minimize edilir)
4. İlk çalıştırmada model indirme süresi

## Geliştirme Önerileri

- Farklı temperature değerleriyle deneme
- Perplexity eşik değerini ayarlama (40-60 arası)
- Daha büyük seed veri seti kullanımı
- Fine-tuned domain-specific BERT modeli kullanımı

## Referanslar

- BERT: [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
- BERTScore: [Zhang et al., 2019](https://arxiv.org/abs/1904.09675)
- Turkish BERT: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

---

**Not**: Bu proje araştırma ve eğitim amaçlıdır. Üretilen sentetik veriler gerçek haber içeriği değildir.
