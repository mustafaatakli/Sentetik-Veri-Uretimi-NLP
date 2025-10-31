# Türkçe Duygu Analizi: LSTM Modeli Eğitimi ve Gemini ile Karşılaştırması

Bu proje, Türkçe metinler için duygu analizi (pozitif, negatif, nötr) yapmak üzere bir Bidirectional LSTM modeli eğitmek için geliştirilmiş bütüncül bir sistem sunar. Sistemin temel amacı, geleneksel bir derin öğrenme modeli olan LSTM'nin performansını, Google'ın gelişmiş büyük dil modeli Gemini tarafından üretilen etiketlerle karşılaştırmaktır.

Kod, bir eğitim veri setiyle modeli sıfırdan eğitir, eğitilmiş modeli kullanarak Gemini tarafından etiketlenmiş yeni bir veri setini tahmin eder ve son olarak iki modelin tahminlerini detaylı bir şekilde karşılaştırarak bir sonuç raporu oluşturur.

## 🚀 Temel Özellikler

- **Model Eğitimi:** `egitim-veriseti-5k.xlsx` dosyasını kullanarak sıfırdan bir Bidirectional LSTM modeli eğitir.
- **Performans Değerlendirme:** Eğitim sonrası modelin performansını test veri seti üzerinde `Accuracy`, `Precision`, `Recall`, `F1-Score` gibi metriklerle ölçer ve sınıflandırma raporu sunar.
- **Otomatik Etiketleme:** Eğitilmiş modeli kullanarak `etiketsiz-test-gemini-etiketlenmis-1k.xlsx` dosyasındaki metinler için duygu tahmini yapar.
- **🔍 LSTM vs. Gemini Karşılaştırması:** LSTM modelinin tahminlerini, Gemini'nin daha önceden yaptığı tahminlerle karşılaştırır.
- **💾 Model ve Tokenizer Kaydı:** Eğitim sonrası en iyi model (`best_lstm_model.h5`) ve metin ön işleme için kullanılan tokenizer (`tokenizer.pickle`) kaydedilir.

## 🛠️ Kullanılan Teknolojiler

- **Python 3**
- **TensorFlow / Keras:** Derin öğrenme modelini oluşturmak ve eğitmek için.
- **Pandas:** Veri setlerini okumak ve işlemek için.
- **Scikit-learn:** Model performansını değerlendirmek ve metrikleri hesaplamak için.
- **Numpy:** Sayısal işlemler için.
- **Openpyxl:** Excel dosyaları ile çalışmak için.

## ⚙️ Kurulum ve Çalıştırma

Bu kodun en verimli şekilde çalışması için **Kaggle** veya **Google Colab** gibi GPU destekli bir ortamda çalıştırılması önerilmektedir.

1.  **Ortamı Hazırlayın:**
    - Bir Kaggle Notebook veya Google Colab not defteri oluşturun.
    - `GPU T4 x2` gibi bir GPU hızlandırıcı seçeneğini aktif edin.

2.  **Dosyaları Yükleyin:**
    - Eğitim veri setini içeren `egitim-veriseti-5k.xlsx` dosyasını ortama yükleyin.
    - Gemini tarafından etiketlenmiş test verilerini içeren `etiketsiz-test-gemini-etiketlenmis-1k.xlsx` dosyasını yükleyin.

3.  **Dosya Yollarını Kontrol Edin:**
    `main.py` dosyası içindeki `AYARLAR` bölümünde dosya yollarının, yüklediğiniz dosyaların isimleriyle eşleştiğinden emin olun.

    ```python
    # 📁 Dosya yolları
    EGITIM_DOSYASI = 'egitim-veriseti-5k.xlsx'
    ETIKETSIZ_DOSYA = 'etiketsiz-test-gemini-etiketlenmis-1k.xlsx'
    CIKTI_DOSYASI = 'lstm_vs_gemini_sonuc_1k.xlsx'
    ```

4.  **Kodu Çalıştırın:**
    - `main.py` içerisindeki kodun tamamını not defterine yapıştırın ve çalıştırın.
    - Script, kütüphaneleri otomatik olarak kuracak, modeli eğitecek, tahminleri yapacak ve karşılaştırma sonuçlarını ekrana yazdıracaktır.

## 📊 Sonuçlar ve Analiz

### 1️⃣ LSTM Model Performansı (Test Seti)

```
[cite_start]🎯 Accuracy(Doğruluk): %86.8 [cite: 1]
   [cite_start]Precision: 0.8698 [cite: 1]
   [cite_start]Recall: 0.8680 [cite: 1]
   [cite_start]F1 Score: 0.8673 [cite: 1]
```

**Sınıf Bazında:**
- [cite_start]✅ Negatif: **%93** doğruluk (çok iyi!) [cite: 1]
- [cite_start]⚪ Nötr: **%75** doğruluk (zayıf nokta) [cite: 1]
- [cite_start]✅ Pozitif: **%90** doğruluk (iyi) [cite: 1]

---

### 2️⃣ Gemini ile Karşılaştırma (1000 örnek)

```
[cite_start]🎯 LSTM vs Gemini Accuracy: %87.5 [cite: 1]

📊 Uyuşma:
   [cite_start]✅ Aynı tahmin: 875 örnek (%87.50) [cite: 1]
   [cite_start]❌ Farklı tahmin: 125 örnek (%12.50) [cite: 1]
```

**Sınıf Bazında Uyuşma:**
- [cite_start]✅ Negatif: **%90.6** uyuşma [cite: 1]
- [cite_start]⚪ Nötr: **%81.4** uyuşma (en düşük) [cite: 1]
- [cite_start]✅ Pozitif: **%89.2** uyuşma [cite: 1]

---

## 🔥 BERT vs LSTM KARŞILAŞTIRMASI

| Metrik | BERT | LSTM | Fark |
|---|---|---|---|
| **Test Accuracy** | [cite_start]%92.6 [cite: 2] | [cite_start]%86.8 [cite: 2] | [cite_start]-5.8% [cite: 2] |
| **Gemini Uyuşma** | [cite_start]%92.3 [cite: 3] | [cite_start]%87.5 [cite: 3] | [cite_start]-4.8% [cite: 3] |
| **Güven Skoru** | %98.3 | %94.7 | -3.6% |
| **Eğitim Süresi** | [cite_start]~15-20 dk [cite: 4] | [cite_start]~10 dk [cite: 4] | [cite_start]✅ Daha hızlı [cite: 4] |
| **Bellek Kullanımı** | ~6-8 GB | [cite_start]~2-3 GB [cite: 5] | [cite_start]✅ Daha az [cite: 5] |
| **Model Boyutu** | 110M param | ~1-2M param | [cite_start]✅ Çok daha küçük [cite: 6] |

---

### 🎯 SINIF BAZINDA DETAYLI KARŞILAŞTIRMA

#### **Negatif Sınıf:**
- BERT: %98 → LSTM: %93 (-5%)
- İkisi de çok başarılı

#### **Pozitif Sınıf:**
- BERT: %94 → LSTM: %90 (-4%)
- LSTM yine iyi performans

#### **Nötr Sınıf:** ⚠️
- BERT: %84 → LSTM: %75 (-9%)
- **Her iki modelde de en zayıf nokta!**

---

### 🔍 İLGİNÇ BULGULAR

**1. [cite_start]Aynı Problem Patterni:** [cite: 7]
[cite_start]Her iki modelde de farklı tahminler çoğunlukla `Gemini: pozitif → Model: nötr` şeklinde gerçekleşmektedir. [cite: 7]

**Örnek:**
- [cite_start]Cümle: "BMW ve Mercedes lüks elektrikli modellerle pazara giriyor" [cite: 7]
  - [cite_start]Gemini: **pozitif** (olumlu gelişme) [cite: 7]
  - [cite_start]BERT: **nötr** (objektif bilgi) [cite: 7]
  - [cite_start]LSTM: **nötr** (objektif bilgi) [cite: 7]

**2. [cite_start]Early Stopping Çalıştı:** [cite: 8]
- [cite_start]Model, belirlenen 20 epoch yerine **10 epoch**'ta eğitimi durdurdu. [cite: 8]
- [cite_start]En iyi model **5. epoch**'ta kaydedildi. [cite: 8]
- [cite_start]Bu sayede modelin ezber yapması (overfitting) engellendi. [cite: 8]

**3. [cite_start]Learning Rate Otomatik Düştü:** [cite: 9]
[cite_start]`ReduceLROnPlateau` callback'i sayesinde modelin öğrenme oranı, performans artışı yavaşladığında otomatik olarak düşürüldü: [cite: 9]
- [cite_start]Başlangıç: 0.001 [cite: 9]
- [cite_start]Epoch 6'da: 0.0005 [cite: 9]
- [cite_start]Epoch 9'da: 0.00025 [cite: 9]

## 📈 Eğitim Grafikleri

Eğitim tamamlandıktan sonra, modelin öğrenme sürecini görselleştirmek için aşağıdaki kodu kullanabilirsiniz. Bu kod, eğitim ve doğrulama setleri için doğruluk (accuracy) ve kayıp (loss) metriklerinin grafiğini çizer.

---

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

### Yıldız Vermeyi Unutmayın! ⭐

Bu projeyi faydalı bulduysanız, GitHub'da yıldız vererek destek olabilirsiniz!
