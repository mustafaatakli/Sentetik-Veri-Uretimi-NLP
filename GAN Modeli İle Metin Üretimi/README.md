# PyTorch ile GAN Tabanlı Türkçe Cümle Üretimi

Bu proje, PyTorch kullanılarak geliştirilmiş bir Üretici Çekişmeli Ağ (Generative Adversarial Network - GAN) modelidir. Model, Türkçe Vikipedi verileriyle eğitilerek özgün ve anlamsal olarak tutarlı yeni cümleler üretmeyi amaçlamaktadır.

![Eğitim Grafiği](training_history.png)

##  Genel Bakış

Projenin temel amacı, sentetik metin verisi üretimi için bir GAN mimarisi kurmak ve bu mimariyi Türkçe dil yapısına uygun cümleler üretecek şekilde eğitmektir. Üretilen cümleler, çeşitli filtreleme ve kalite skorlama aşamalarından geçirilerek en başarılı olanları seçilir ve bir CSV dosyasına kaydedilir.

## Temel Özellikler

- **LSTM Tabanlı GAN Mimarisi:** Hem Üretici (Generator) hem de Ayırt Edici (Discriminator) model, metin verilerinin sıralı doğasını yakalamak için LSTM katmanları kullanır.
- **Gelişmiş Filtreleme:** Üretilen ham cümleler, sadece kelime sayısına göre değil, aynı zamanda orijinal veri setine olan anlamsal benzerliklerine (Cosine Similarity) göre de elenir.
- **Kalite Skorlama Modülü:** Filtrelenen cümleler, kelime çeşitliliği, ortalama kelime uzunluğu gibi metrikleri baz alan özel bir fonksiyon ile puanlanır.
- **Yapılandırılabilir Parametreler:** Eğitim süresi (epoch), üretilecek cümle sayısı, filtreleme eşikleri gibi tüm önemli parametreler `main.py` dosyasının başında kolayca değiştirilebilir.
- **Detaylı Raporlama:** Eğitim sonunda kayıp (loss) ve doğruluk (accuracy) grafiklerini içeren bir görsel (`training_history.png`) ve en kaliteli cümleleri içeren bir CSV dosyası (`uretilen_cumleler.csv`) oluşturulur.

## Veri Seti

Bu projede, temel kaynak olarak Nisan 2021 tarihli Türkçe Vikipedi dökümanlarından derlenmiş bir veri seti kullanılmıştır.

**Kaynak Veri Seti (`wiki.tr.txt`)**
- **İçerik:** Türkçe Vikipedi'den alınmış binlerce cümle.
- **Özellikler:** Cümle başına en fazla 14, en az 2 kelime. Özel sembol ve kısaltma içermez.
- **Kaynak Link:** [Turkish Sentences Dataset on Kaggle](https://www.kaggle.com/datasets/mahdinamidamirchi/turkish-sentences-dataset?select=wiki.tr.txt)

**Projede Kullanılan Dosyalar**
Bu projede geliştirme ve hızlı denemeler yapabilmek amacıyla iki farklı veri dosyası bulunmaktadır:
- `wiki.tr.txt`: Kaggle'dan indirilen **orijinal ve büyük** veri setidir.
- `sentences.txt`: `wiki.tr.txt` dosyasından rastgele seçilmiş **5000 cümleden oluşan daha küçük** bir alt kümedir. Varsayılan olarak kod bu dosya ile çalışır.

## 📁 Dosyalar

- sentences.txt = Birbirinden farklı 5000  adet 'elektrikli arabalar' ile ilgili cümleler.
- uretilen_cumleler.csv = Eğitim sonucu oluşturulan cümlelerin bulunduğu veriseti.
- wiki.tr.txt = Kaynak veri seti.

---

## Lisans

Bu projenin tüm hakları saklıdır © 2025 Mustafa Ataklı.
İzinsiz kullanımı, kopyalanması veya dağıtımı kesinlikle yasaktır.
Detaylı bilgi için lütfen LICENSE.md dosyasına bakınız.

### Yıldız Vermeyi Unutmayın! ⭐

Bu projeyi faydalı bulduysanız, GitHub'da yıldız vererek destek olabilirsiniz!