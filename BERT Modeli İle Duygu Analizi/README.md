# Türkçe Duygu Analizi: BERT ile Model Eğitimi ve Gemini ile Karşılaştırma

Bu proje, Türkçe metinler için BERT tabanlı bir duygu analizi modelini eğitmek, test etmek ve bu modelin performansını Google Gemini tarafından üretilen etiketlerle karşılaştırmak için geliştirilmiş komple bir sistem sunar.

Proje, `dbmdz/bert-base-turkish-cased` modelini kullanarak üç sınıflı (pozitif, negatif, nötr) bir duygu sınıflandırması yapar. Eğitim ve değerlendirme süreçlerinin ardından, eğitilmiş modeli kullanarak yeni bir veri setini etiketler ve sonuçları Gemini etiketleriyle detaylı olarak karşılaştırır.

## 🚀 Projenin Temel Özellikleri

- **Eğitim:** Sıfırdan bir Türkçe BERT duygu analizi modeli eğitir.
- **Değerlendirme:** Modelin performansını test seti üzerinde `accuracy`, `precision`, `recall` ve `F1-score` metrikleriyle ölçer.
- **Tahmin:** Eğitilmiş modeli kullanarak daha önce etiketlenmemiş yeni veriler üzerinde duygu tahmini yapar.
- **Karşılaştırma:** BERT modelinin tahminlerini, aynı veri seti için Gemini tarafından üretilen etiketlerle karşılaştırarak model tutarlılığını analiz eder.
- **Raporlama:** Sınıf bazında performans raporları, karmaşıklık matrisi (confusion matrix) ve iki model arasındaki uyuşma oranlarını sunar.
- **Model Kaydı:** Eğitim sonrası en iyi performansı gösteren model, daha sonra kullanılmak üzere kaydedilir.

## 💻 Geliştirme Ortamı
Bu projenin eğitim ve test süreçleri aşağıdaki ortamda gerçekleştirilmiştir:
- **Platform:** Kaggle Notebooks
- **Hızlandırıcı:** GPU T4 x2

## 📁 Dosyalar
- main.py = Ana program dosyası.
- egitim-veriseti-5k.xlsx = 5000 adet 'elektrik araba' temalı (7-13) kelime aralığında etiketli(pozitif,n egatif, nötr) cümlelerden oluşturulmuş eğitim veriseti.
- bert_vs_gemini_sonuc_1k.xlsx = Eğitim sonucunda etiketlenmiş cümlelerin, gemini etiketlenme sonuçları ile detaylı karşılaştırma dosyası.
- etiketsiz-test-gemini-etiketlenmis-1k.xlsx = Sadece gemini api ile etiketlenmiş 1000 adet cümleden oluşan veriseti.

## 📊 Performans Sonuçları

Model, test verileri üzerinde ve Gemini ile karşılaştırmalı olarak aşağıdaki performansı göstermiştir.

### BERT Test Performansı

| Metrik | Değer |
| :--- | :--- |
| 🎯 **Accuracy (Doğruluk)** | **%92.60** |
| Precision | 0.9261 |
| Recall | 0.9260 |
| F1 Score | 0.9256 |

**Sınıf Bazında Doğruluk:**
- **Negatif:** `%98` (Mükemmel)
- **Pozitif:** `%94` (Çok İyi)
- **Nötr:** `%84` (Makul)

### Gemini ile Karşılaştırma (1000 Örnek)

Bu analizde, BERT modelinin tahminleri Gemini tarafından üretilen etiketlerle referans alınarak değerlendirilmiştir.

| Metrik | Değer |
| :--- | :--- |
| 🎯 **BERT vs Gemini Accuracy** | **%92.30** |
| ✅ **Aynı Tahmin (Uyuşma)** | **923 örnek (%92.3)** |
| ❌ **Farklı Tahmin** | **77 örnek (%7.7)** |

**Sınıf Bazında Uyuşma Oranları:**
- **Negatif:** `%96.1` (En yüksek uyuşma)
- **Pozitif:** `%90.8`
- **Nötr:** `%89.3`

> ### 🔍 Önemli Bulgular
> - **BERT Çok Başarılı:** Model, `%92.6`'lık test doğruluğu ve Gemini ile `%92.3`'lük tutarlılık oranıyla oldukça başarılı bir performans sergilemektedir.
> - **En Güçlü Sınıf "Negatif":** Model, hem kendi test setinde (`%98`) hem de Gemini karşılaştırmasında (`%96.1`) en iyi performansı negatif duyguları tespit etmede göstermiştir.
> - **İyileştirme Alanı "Nötr":** En düşük performans nötr sınıfta gözlemlenmiştir. Analizler, BERT'in bazı durumlarda **pozitif cümleleri nötr** olarak etiketleme eğiliminde olduğunu göstermektedir.

## 🛠️ Kurulum

Projeyi çalıştırmak için gerekli olan kütüphaneleri aşağıdaki komut ile kurabilirsiniz.

```bash
pip install torch transformers pandas numpy openpyxl scikit-learn
```

## ⚙️ Kullanım

Proje, `main.py` scripti üzerinden çalıştırılır. Script, Kaggle ortamında GPU ile çalışacak şekilde optimize edilmiştir ancak yerel makinenizde de çalışabilir.

1.  **Veri Setlerini Hazırlayın:**
    -   `egitim-veriseti-5k.xlsx`: `text` ve `sentiment` sütunlarını içeren 5000 örnekli eğitim verisi.
    -   `etiketsiz-test-gemini-etiketlenmis-1k.xlsx`: `text` ve Gemini tarafından etiketlenmiş `sentiment` sütunlarını içeren 1000 örnekli karşılaştırma verisi.

2.  **Script Ayarlarını Yapılandırın:**
    `main.py` dosyasının başındaki `AYARLAR` bölümünden dosya yollarını ve model hiperparametrelerini (epoch, batch size vb.) düzenleyebilirsiniz.

    ```python
    # Dosya yolları
    EGITIM_DOSYASI = 'path/to/egitim-veriseti-5k.xlsx'
    ETIKETSIZ_DOSYA = 'path/to/etiketsiz-test-gemini-etiketlenmis-1k.xlsx'
    CIKTI_DOSYASI = 'bert_vs_gemini_sonuc_1k.xlsx'

    # Model ayarları
    MODEL_ADI = 'dbmdz/bert-base-turkish-cased'
    EPOCHS = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    ```

3.  **Script'i Çalıştırın:**
    Terminal üzerinden aşağıdaki komutu çalıştırın.

    ```bash
    python main.py
    ```


---

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

### Yıldız Vermeyi Unutmayın! ⭐

Bu projeyi faydalı bulduysanız, GitHub'da yıldız vererek destek olabilirsiniz!