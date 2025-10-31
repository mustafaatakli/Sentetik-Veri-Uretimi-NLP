# Ses Veri Artırımı ve Sentezi Üzerine Literatür Derlemesi

Bu repo, ses sınıflandırma, konuşmacı doğrulama ve anahtar kelime tanıma gibi alanlarda model performansını iyileştirmek amacıyla sentetik ses verisi üretimi ve veri artırımı üzerine yapılmış akademik çalışmaları bir araya getirmektedir. Özellikle sınırlı veri kümeleriyle çalışırken karşılaşılan zorluklara çözüm olarak sunulan modern teknikler incelenmektedir.

## 📜 Genel Bakış

[cite_start]Derin öğrenme tabanlı ses işleme modelleri, yüksek performans elde etmek için genellikle büyük miktarda etiketli veriye ihtiyaç duyar[cite: 2129]. [cite_start]Ancak, gerçek dünya verisi toplamak ve etiketlemek maliyetli ve zaman alıcı bir süreçtir[cite: 2130]. Bu noktada, veri artırımı (data augmentation) ve sentetik veri üretimi (synthetic data generation) kritik bir rol oynamaktadır.

Bu derlemede incelenen makaleler, aşağıdaki temel sorulara odaklanmaktadır:

* **Veri Artırımı:** Mevcut ses verilerini kullanarak yeni ve çeşitli örnekler nasıl oluşturulabilir?
* **Sentetik Veri Üretimi:** Metinden sese (Text-to-Audio) veya ses dönüşümü (Voice Conversion) gibi tekniklerle sıfırdan gerçekçi ses verileri nasıl üretilebilir?
* **Performans Etkisi:** Bu teknikler, ses sınıflandırma ve doğrulama sistemlerinin doğruluğunu ve sağlamlığını ne ölçüde iyileştirmektedir?

---

## 🔬 İncelenen Teknikler ve Yöntemler

Bu derlemede öne çıkan ana yaklaşımlar aşağıda özetlenmiştir.

### 1. Ses Tabanlı Veri Artırımı (Audio-based Augmentation)

Bu teknikler doğrudan ses dalga formuna veya onun görsel temsillerine (spektrogram, skalogram) uygulanır.

* **Dalga Formu Manipülasyonları**:
    * [cite_start]**Gürültü Ekleme (Noise Addition):** Gerçek dünya koşullarını simüle etmek için sinyale çevresel veya rastgele gürültü eklenir[cite: 655, 1842].
    * [cite_start]**Zaman Esnetme (Time Stretching):** Sesin perdesini değiştirmeden hızını ayarlar[cite: 655, 2252].
    * [cite_start]**Perde Kaydırma (Pitch Shifting):** Sesin hızını değiştirmeden perdesini (frekansını) ayarlar[cite: 655, 1670, 1842, 2252].
    * [cite_start]**Ses Kontrolü (Volume Control):** Farklı kayıt seviyelerini taklit etmek için genliği değiştirir[cite: 655].
* **Görsel Temsil Manipülasyonları**:
    * Sesin **skalogram** gibi görsel temsillerine geometrik dönüşümler uygulanır:
        * [cite_start]Döndürme (Rotation) [cite: 656]
        * [cite_start]Ölçekleme (Scaling) [cite: 656]
        * [cite_start]Kırpma (Shearing) [cite: 656]
        * [cite_start]Öteleme (Translation) [cite: 656]

### 2. Üretken Modellerle Sentetik Veri Üretimi

Bu yaklaşım, mevcut verileri dönüştürmek yerine tamamen yeni ses örnekleri oluşturmaya odaklanır.

* **Metinden Sese (Text-to-Audio - TTA) Modelleri**:
    * [cite_start]**AudioGen** ve **AudioLDM2** gibi modeller, metinsel açıklamalardan (prompt) gerçekçi sesler üretebilir[cite: 657, 2124, 2125].
    * [cite_start]Bu yöntem, özellikle veri toplamanın zor olduğu nadir ses olayları için eğitim setlerini zenginleştirmede oldukça etkilidir[cite: 2128].
    * [cite_start]ChatGPT gibi büyük dil modelleri (LLM'ler) kullanılarak daha çeşitli ve bağlamsal olarak zengin prompt'lar oluşturulabilir[cite: 2191, 2194].
* **Ses Dönüşümü (Voice Conversion - VC)**:
    * [cite_start]Bir konuşmacının sesini (kaynak) başka bir konuşmacının sesine (hedef) dönüştürerek aynı içeriğe sahip yeni ses örnekleri oluşturur[cite: 1736, 1743].
    * [cite_start]Özellikle metne bağlı konuşmacı doğrulama (text-dependent speaker verification) gibi görevlerde, sınırlı sayıda konuşmacıdan daha fazla çeşitlilik elde etmek için kullanılır[cite: 1514, 1517, 1529].
    * [cite_start]**CycleGAN** ve **Autoencoder** tabanlı modeller bu alanda öne çıkan yaklaşımlardır[cite: 108, 112].

---

## 📊 Ana Bulgular ve Sonuçlar

İncelenen makalelerdeki temel bulgular şunlardır:

* [cite_start]**Veri Artırımı Performansı İyileştirir:** Hem ses tabanlı artırım teknikleri hem de sentetik veri üretimi, ses sınıflandırma modellerinin doğruluğunu genellikle artırmaktadır[cite: 662, 1409, 2131]. [cite_start]VGGish modelinde %9.05'e varan doğruluk artışları gözlemlenmiştir[cite: 664].
* [cite_start]**Aşırı Kullanım Riskleri:** Veri setini artırırken belirli bir eşiğin (%100-%200 artış) üzerine çıkmak, modelin aşırı öğrenmesine (overfitting) ve performans düşüşüne neden olabilir[cite: 962, 1337].
* **TTA ve VC'nin Gücü:**
    * [cite_start]TTA modelleri ile üretilen veriler, veri artırımı için kullanıldığında geleneksel sinyal işleme tabanlı yöntemlerden daha iyi sonuçlar vermiştir[cite: 2259, 2260].
    * [cite_start]Ancak, bir modeli **sadece** sentetik verilerle eğitmek, gerçek veriler üzerinde test edildiğinde genellikle daha düşük performansa yol açar[cite: 2118, 2278].
    * [cite_start]Gerçek verilerin bir kısmını (%20-%40) sentetik verilerle değiştirmek, performanstan önemli bir kayıp yaşamadan veri toplama maliyetini düşürme potansiyeli sunar[cite: 2332].
* [cite_start]**Gürültüye Karşı Dayanıklılık:** Gürültü ekleme gibi artırım teknikleri, modelleri çeşitli ve gürültülü ortamlara karşı daha sağlam hale getirir[cite: 1733, 1740].

---

## 🚀 Uygulama Alanları

Bu teknikler, aşağıdaki gibi birçok pratik alanda değerlidir:

* [cite_start]**Akıllı Şehirler:** Cam kırılması, insan çığlığı gibi nadir seslerin tespiti[cite: 681, 709].
* [cite_start]**Endüstriyel Bakım:** Makinelerdeki anormal sesleri tespit ederek kestirimci bakım yapma[cite: 681, 709].
* [cite_start]**Sağlık:** Öksürük, anormal nefes alma gibi sesleri analiz ederek teşhise yardımcı olma[cite: 681, 709].
* [cite_start]**Güvenlik:** Özelleştirilmiş uyandırma kelimeleri (wake-up words) ile konuşmacı doğrulama sistemleri geliştirme[cite: 1529].

---

## ⚠️ Sorumluluk Reddi ve Kullanım Amacı
Bu depoda özetlenen ve referans olarak gösterilen tüm akademik makaleler kamuya açık, çevrimiçi ve erişilebilir kaynaklardan temin edilmiştir.

Bu derlemenin ve ilgili materyallerin temel amacı, sentetik veri üretimi ve veri artırımı alanındaki mevcut bilimsel çalışmaları eğitim ve bilgilendirme hedefiyle bir araya getirmektir. Kullanıcıların, atıfta bulunulan her bir makalenin orijinal kaynağını incelemeleri ve o kaynağın belirttiği lisans koşullarına uymaları beklenmektedir. Tüm çalışmaların hakları orijinal yazarlarına ve yayıncılarına aittir.

---

## 📚 Makalelerin Kaynakları

[1] E. Tsalera, A. Papadakis, G. Pagiatakis, and M. Samarakou, "Impact Evaluation of Sound Dataset Augmentation and Synthetic Generation upon Classification Accuracy," J. Sens. Actuator Netw., vol. 14, no. 91, 2025.

[2] F. Ronchini, L. Comanducci, and F. Antonacci, "Synthetic Training Set Generation Using Text-to-Audio Models for Environmental Sound Classification," arXiv preprint arXiv:2403.17864, 2024.

[3] O. Slizovskaia, J. Janer, P. Chandna, and O. Mayor, "Voice Conversion with Limited Data and Limitless Data Augmentations," arXiv preprint arXiv:2212.13581, 2022.

[4] X. Qin, Y. Yang, L. Yang, X. Wang, J. Wang, and M. Li, "Exploring Voice Conversion Based Data Augmentation in Text-Dependent Speaker Verification," arXiv preprint arXiv:2011.10710, 2020.

[5] Y. A. Wubet and K.-Y. Lian, "Voice Conversion Based Augmentation and a Hybrid CNN-LSTM Model for Improving Speaker-Independent Keyword Recognition on Limited Datasets," IEEE Access, vol. 10, pp. 89170-89181, 2022.

[6] A. Tanna, M. Saxon, A. El Abbadi, and W. Y. Wang, "Data Augmentation for Diverse Voice Conversion in Noisy Environments," arXiv preprint arXiv:2305.10684, 2023.