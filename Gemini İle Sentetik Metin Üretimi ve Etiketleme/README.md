# 🚗 Elektrikli Araba Türkçe Veri Seti Oluşturucu

Bu proje, Google Gemini 2.5 Flash modelini kullanarak "Elektrikli Arabalar" konusunda yüksek kaliteli ve özgün Türkçe cümlelerden oluşan bir veri seti oluşturmak için tasarlanmıştır. Proje, üretilen cümlelerin kalitesini, doğallığını ve benzersizliğini sağlamak için gelişmiş bir filtreleme mekanizması kullanır.

## ✨ Öne Çıkan Özellikler

- **Batch Generation Sistemi:** Tek bir API isteği ile 100 cümle üreterek API çağrılarını ve maliyetleri minimize eder.
- **Dual Hybrid Kalite Skoru:** Cümle kalitesini ölçmek için iki farklı metriği birleştiren hibrit bir yaklaşım kullanır:
    1.  **Faktörel Skor (%50):** Cümlenin kelime sayısı, dilbilgisi, anahtar kelime içeriği gibi yapısal özelliklerini analiz eder.
    2.  **Perplexity Skoru (%50):** Cümlenin ne kadar "doğal" ve "akıcı" olduğunu `ytu-ce-cosmos/turkish-gpt2` modeli ile ölçer.
- **Semantik Benzerlik Filtresi:** `sentence-transformers` kullanarak üretilen cümlelerin birbirine anlamsal olarak çok benzemesini engeller ve %100 özgün bir veri seti hedefler.
- **Detaylı Konfigürasyon:** Hedef cümle sayısı, kalite eşiği, benzerlik oranı gibi parametreler kolayca ayarlanabilir.
- **Çoklu Format Desteği:** Oluşturulan veri setini `.csv`, `.xlsx` ve `.json` formatlarında otomatik olarak kaydeder.

## 🛠️ Kullanılan Teknolojiler

- **Dil Modeli:** Google Gemini 2.5 Flash
- **Semantik Benzerlik:** Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Perplexity (Doğallık) Skoru:** Hugging Face Transformers (`ytu-ce-cosmos/turkish-gpt2`)
- **Veri İşleme:** Pandas, NumPy
- **Programlama Dili:** Python

## 📁 Dosyalar
- main10.py = Ana program dosyası.
- main10.pdf = Eğitim sonucunda oluşturulan detaylı rapor.
- elektrikli_araba_1000_batch.xlsx = Eğitim sonucunda oluşturulan 1000 adet 'elektrik araba' temalı cümlelerden oluşan veriseti.

## ⚙️ Nasıl Çalışır?

Projenin iş akışı dört ana adımdan oluşur:

1.  **Batch Üretimi:** Belirlenen konu başlıkları ve duygu dağılımına göre Gemini API'sine tek bir istek gönderilerek 100 cümlelik bir batch oluşturulur.
2.  **Kalite Skorlaması:** Her cümle, yapısal kalitesini ölçen **Faktörel Skor** ve doğallığını ölçen **Perplexity Skoru** ile değerlendirilir. Bu iki skorun ağırlıklı ortalamasıyla nihai kalite puanı hesaplanır.
3.  **Benzerlik Kontrolü:** Cümlenin, daha önce kabul edilmiş tüm cümlelere anlamsal olarak ne kadar benzediği ölçülür. Belirlenen eşiğin (`SIMILARITY_THRESHOLD`) üzerindeki cümleler elenir.
4.  **Filtreleme ve Kayıt:** Sadece belirlenen kalite eşiğini (`QUALITY_THRESHOLD`) geçen ve benzerlik testini başarıyla tamamlayan cümleler nihai veri setine eklenir. Bu işlem, hedeflenen cümle sayısına ulaşılana kadar tekrarlanır.

## 📊 Çalışma Sonuçları

Aşağıda, 1000 cümlelik bir veri seti oluşturma işleminin terminal çıktısı özetlenmiştir.

```bash
================================================================================
VERİ SETİ OLUŞTURULDU
================================================================================
Toplam cümle: 1000
Toplam batch: 42
Toplam API isteği: 42
Toplam süre: 50.5 dakika
Reddedilen: 3021

Sentiment Dağılımı:
pozitif : 400 (% 40.0)
negatif : 200 (% 20.0)
nötr    : 400 (% 40.0)

Ortalama Skorlar:
Quality:    0.688
Faktörel:   0.720
Perplexity: 0.658
Similarity: 0.770

Kelime İstatistikleri:
Ortalama:   7.8 kelime
Minimum:    4 kelime
Maksimum:   12 kelime
================================================================================
```

---

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

### Yıldız Vermeyi Unutmayın! ⭐

Bu projeyi faydalı bulduysanız, GitHub'da yıldız vererek destek olabilirsiniz!