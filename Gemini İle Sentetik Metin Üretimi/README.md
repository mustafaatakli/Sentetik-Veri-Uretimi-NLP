# Gemini İle Sentetik Metin Üretimi

Bu proje, Google'ın Gemini 2.5 Flash LLM modeli kullanarak Türkçe teknoloji haber başlıklarından sentetik metin üretimi gerçekleştirmektedir. API tabanlı yaklaşım sayesinde yüksek kaliteli, doğal ve anlamlı cümleler üretilmektedir.

## Proje Açıklaması

Proje, 100 adet gerçek teknoloji haber başlığından yola çıkarak, Gemini API ile 3000 adet sentetik haber başlığı üretmektedir. Üretilen veriler, BERT tabanlı metriklerle kapsamlı bir şekilde değerlendirilmekte ve kalite kontrol yapılmaktadır.

## Ana Özellikler

### 1. Gemini 2.5 Flash API Entegrasyonu
- **Model**: `gemini-2.5-flash`
- **API Tabanlı**: Google Generative AI API kullanımı
- **Optimize Edilmiş Prompt**: Türkçe teknoloji haberleri için özel prompt tasarımı
- **Retry Mekanizması**: API hataları için otomatik yeniden deneme (max 3 kez)
- **Rate Limiting**: API kotası koruması için bekleme süreleri

### 2. Akıllı Prompt Mühendisliği

Gemini modeline verilen prompt şu özelliklere sahiptir:
- Orijinalden en az 2-3 kelime farklılık zorunluluğu
- Aynı anlam/konuyu koruma
- Doğal Türkçe ve dilbilgisi kontrolü
- Teknoloji terminolojisi korunumu
- Gerçekçi haber başlığı formatı (5-15 kelime arası)
- Numara ve özel karakterlerin otomatik temizlenmesi

### 3. Çok Katmanlı Kalite Kontrol

#### A. Gemini Çıktı Kontrolü
- Numara ve işaretlerin regex ile temizlenmesi
- Kelime sayısı kontrolü (3-20 kelime)
- Orijinal cümlelerle karşılaştırma
- Normalize edilmiş form kontrolü
- En az 2 kelime farklılık zorunluluğu

#### B. BERT Perplexity Filtreleme
- BERT MLM (Masked Language Model) ile doğallık ölçümü
- Eşik değer: 50.0 (düşük = doğal, yüksek = anlamsız)
- GPU hızlandırmalı hesaplama
- Anlamsız/yapay cümlelerin otomatik elenmesi

#### C. Çeşitlilik Kontrolü
- Tekrarlı cümlelerin temizlenmesi
- Orijinal veri seti ile çakışma kontrolü
- Set tabanlı hızlı arama

### 4. Kapsamlı 5 Temel Metrik Analizi

#### [1] Tekil Oran (Uniqueness)
- Her cümlenin benzersizliği kontrolü
- İdeal: ≥ %95
- Tekrar eden cümle tespiti

#### [2] BERTScore F1 (Anlamsal Benzerlik)
- BERT embeddings ile anlamsal benzerlik
- Precision, Recall ve F1 skorları
- TF-IDF ile en yakın orijinal cümle eşleştirmesi
- İdeal: ≥ 0.85

#### [3] Kelime Kapsama (Vocabulary Coverage)
- Orijinal kelimelerin korunma oranı
- Ortak kelime sayısı
- İdeal: ≥ %90

#### [4] Benzerlik Skoru (TF-IDF Cosine Similarity)
- Kelime tabanlı benzerlik analizi
- Dağılım analizi (yüksek/orta/düşük)
- İdeal: 0.5 - 0.7 (dengeli benzerlik)

#### [5] Perplexity Skoru (Anlamsal Doğallık)
- BERT MLM ile cümle doğallığı
- İstatistiksel analiz (ortalama, min/max, std)
- İdeal: ≤ 50

### 5. Görselleştirme ve Raporlama
- Kelime sayısı dağılımı karşılaştırması
- Benzerlik skoru dağılımı
- Perplexity dağılımı
- En sık kullanılan kelimeler (Orijinal vs Sentetik)
- 5 temel metrik özeti
- Yüksek çözünürlüklü grafik çıktısı (300 DPI)

## Dosya Yapısı

```
Gemini İle Sentetik Metin Üretimi/
│
├── gemini_sentetik_uretim.py                        # Ana Python scripti
├── tekonoloji-haber-baslıkları.csv                  # Orijinal veri (100 haber başlığı)
├── gemini_sentetik_teknoloji_haberleri_3000.csv     # Üretilen sentetik veri (3000 başlık)
├── gemini-cıktılar.txt                              # Üretim süreci log dosyası
└── gemini_sentetik_metrikler.png                    # Görselleştirme çıktısı
```

## Gereksinimler

```
google-generativeai
pandas
numpy
torch
tqdm
scikit-learn
matplotlib
transformers
bert-score
```

## Kurulum

### 1. Paketleri Yükleyin

```bash
pip install google-generativeai pandas numpy torch tqdm scikit-learn matplotlib transformers bert-score
```

### 2. Gemini API Key Alın

1. [Google AI Studio](https://makersuite.google.com/app/apikey) adresine gidin
2. API key oluşturun
3. API key'i kopyalayın

### 3. API Key'i Ayarlayın

**Seçenek 1: Kod içinde (önerilmez)**
```python
API_KEY = "your_api_key_here"
```

**Seçenek 2: Environment Variable (önerilen)**
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

```python
import os
API_KEY = os.getenv('GEMINI_API_KEY')
```

## Kullanım

```bash
python gemini_sentetik_uretim.py
```

### Çalışma Akışı

1. **API Bağlantısı**: Gemini API'ye bağlanır ve test eder
2. **BERT Yükleme**: Metrik hesaplamaları için BERT modelini yükler
3. **Veri Yükleme**: 100 orijinal cümle okunur
4. **Kullanıcı Onayı**: Enter tuşu ile üretim başlatılır
5. **Sentetik Üretim** (23-25 dakika):
   - Her cümleden ~30 varyant üretilir
   - Progress bar ile ilerleme gösterilir
   - Her 10 cümlede bir durum raporu
6. **Tekrar Temizleme**: Çift cümleler ve orijinallerle eşleşenler çıkarılır
7. **Eksik Kontrolü**: Hedef sayıya ulaşılmazsa ek üretim
8. **Perplexity Filtreleme**: BERT ile kalite kontrolü
9. **Kayıt**: CSV dosyasına kaydetme
10. **Metrik Analizi**: 5 temel metrik hesaplama
11. **Görselleştirme**: Grafiklerin oluşturulması

## Teknik Detaylar

### Gemini Prompt Yapısı

```python
prompt = f"""Sen bir Türkçe dil uzmanısın. Aşağıdaki teknoloji haber başlığından {n_varyant} adet
YENİ ve FARKLI haber başlığı üret.

KAYNAK BAŞLIK: "{cumle}"

KURALLAR:
1. Her başlık orijinalden FARKLI olmalı (en az 2-3 kelime değişmeli)
2. Aynı anlam/konuyu korumalı ama farklı kelimeler kullanmalı
3. Doğal Türkçe olmalı, dilbilgisi kurallarına uygun
4. Teknoloji terminolojisini koruyabilirsin ama çeşitlendir
5. Her başlık yeni satırda, numarasız, sadece başlık metni
6. Kısa ve öz (5-15 kelime arası)
7. Gerçekçi haber başlığı formatında

Şimdi {n_varyant} adet başlık üret:
"""
```

### API Çağrısı ve Hata Yönetimi

```python
# Retry mekanizması (max 3 deneme)
retry_count = 0
while retry_count < max_retry:
    try:
        response = model.generate_content(prompt)
        # İşleme...
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        retry_count += 1
        time.sleep(2)  # Hata durumunda daha uzun bekleme
```

### Çıktı Parse ve Temizleme

```python
# Numara temizleme: "1. ", "1) "
line = re.sub(r'^\d+[\.\)]\s*', '', line)

# İşaret temizleme: "- ", "* ", "• "
line = re.sub(r'^[-*•]\s*', '', line)
```

## Performans

| Metrik | Değer |
|--------|-------|
| Üretim Süresi | ~23-25 dakika (100 cümle → 3000 cümle) |
| API Çağrısı | ~100-150 request |
| GPU Kullanımı | BERT metrik hesaplamaları için |
| Başarı Oranı | 4-5/5 metrik geçilir |
| Perplexity Eşik | ≤ 50.0 |

## Örnek Çıktılar

### Orijinal Veri
```
Google şov yapacak Google I/O 2025 canlı yayını
12 taksitle alınabilecek en iyi akıllı telefonlar
Hastaneyi bileğe getiren saat Huawei Watch 5 çok farklı olmuş
```

### Üretilen Sentetik Örnekler (Log'dan)
```
[Script çalıştırıldığında 10 örnek görüntülenir]
```

## Değerlendirme Kriterleri

### İdeal Metrik Değerleri

| Metrik | İdeal Değer | Durum |
|--------|-------------|-------|
| Tekil Oran | ≥ %90 | ✓ Mükemmel |
| BERTScore F1 | ≥ 0.80 | ✓ Mükemmel |
| Kelime Kapsama | ≥ %85 | ✓ Mükemmel |
| Benzerlik Skoru | 0.45 - 0.75 | ✓ Mükemmel |
| Perplexity | ≤ 60 | ✓ Mükemmel |

### Başarı Değerlendirmesi

- **5/5 metrik**: ✓ MÜKEMMEL - Sentetik veri yüksek kalitede!
- **4/5 metrik**: ✓ MÜKEMMEL - Sentetik veri yüksek kalitede!
- **3/5 metrik**: ~ İYİ - Sentetik veri kullanılabilir durumdadır
- **<3 metrik**: ⚠ İYİLEŞTİRİLEBİLİR - Parametreleri ayarlayabilirsiniz

## Avantajlar

### Gemini API Kullanımının Avantajları

1. **Yüksek Kalite**: LLM tabanlı doğal dil üretimi
2. **Anlamsal Tutarlılık**: Konuyu ve bağlamı korur
3. **Dilbilgisi Doğruluğu**: Türkçe dilbilgisi kurallarına uygun
4. **Çeşitlilik**: Yaratıcı kelime seçimleri ve yapılar
5. **Kolay Kullanım**: API entegrasyonu basit
6. **Ölçeklenebilir**: Bulk işlemler için uygun

### BERT Modeli vs Gemini LLM

| Özellik | BERT MLM | Gemini API |
|---------|----------|------------|
| Üretim Yöntemi | Maskeleme + Tahmin | LLM Text Generation |
| Doğallık | Orta-Yüksek | Çok Yüksek |
| Anlamsal Tutarlılık | Orta | Yüksek |
| Çeşitlilik | Orta | Yüksek |
| Hız | Hızlı (GPU) | Orta (API) |
| Maliyet | Ücretsiz | API Kotası |
| Karmaşıklık | Yüksek (Kodlama) | Düşük (Prompt) |

## Sınırlamalar

1. **API Kotası**: Gemini API'nin günlük/dakikalık limitleri var
2. **Maliyet**: Ücretsiz katmanda sınırlı kullanım (15 RPM / 1500 RPD)
3. **Bağımlılık**: İnternet bağlantısı gerekli
4. **Süre**: API çağrıları zaman alıcı (~23 dakika)
5. **Rate Limiting**: API kotası dolduğunda bekleme gerekir

## İyileştirme Önerileri

### API Optimizasyonu
```python
# Batch processing (eğer API desteklerse)
# Async requests ile paralel çağrılar
# Daha uzun prompt ile tek seferde daha fazla varyant
```

### Perplexity Eşik Ayarı
```python
# Daha sıkı filtreleme için
perplexity_esik = 40.0  # (varsayılan: 50.0)

# Daha gevşek filtreleme için
perplexity_esik = 60.0
```

### Varyant Sayısı Ayarı
```python
# Her cümleden daha fazla varyant
n_varyant = 40  # (varsayılan: 30)
```

## Sorun Giderme

### API Bağlantı Hatası
```
[HATA] Gemini API baglantisi basarisiz
```
**Çözüm**: API key'inizi kontrol edin ve `API_KEY` değişkenini güncelleyin

### Rate Limit Hatası
```
[UYARI] Gemini API hatasi: 429 Resource exhausted
```
**Çözüm**:
- `time.sleep()` sürelerini artırın
- Günlük kotanızı kontrol edin
- API plan'ınızı yükseltin

### GPU Bulunamadı
```
[UYARI] GPU bulunamadi, CPU kullaniliyor
```
**Çözüm**: Bu bir uyarıdır, program CPU'da çalışmaya devam eder (daha yavaş)

### Perplexity Filtreleme Sonrası Eksik
```
[UYARI] Filtreleme sonrasi 2800 cumle kaldi
```
**Çözüm**: Program otomatik olarak eksik cümleleri üretir (normal davranış)

## API Kotaları (Ücretsiz Katman)

| Limit | Değer |
|-------|-------|
| RPM (Requests Per Minute) | 15 |
| RPD (Requests Per Day) | 1500 |
| TPM (Tokens Per Minute) | 1 milyon |

## Güvenlik

⚠️ **ÖNEMLİ**: API key'inizi asla public repository'lere yüklemeyin!

```python
# ✗ Yanlış
API_KEY = "AIzaSyCAX_9lbx3_7RmpD8MZL1BcY7o8RmNEEs8"

# ✓ Doğru
API_KEY = os.getenv('GEMINI_API_KEY')
```

## Referanslar

- **Gemini API**: [Google AI Studio](https://ai.google.dev/)
- **Gemini Docs**: [Documentation](https://ai.google.dev/docs)
- **BERTScore**: [Zhang et al., 2019](https://arxiv.org/abs/1904.09675)
- **Turkish BERT**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

---

**Not**: Bu proje araştırma ve eğitim amaçlıdır. Üretilen sentetik veriler gerçek haber içeriği değildir. Gemini API kullanımı Google'ın hizmet şartlarına tabidir.
