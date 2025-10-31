# Yapay Zeka ile Sentetik Veri Üretimi: Literatür İncelemesi

Bu repo, özellikle Doğal Dil İşleme (NLP) alanında sentetik veri üretimi üzerine odaklanan akademik makalelerin bir derlemesini ve özetini içermektedir. İncelenen çalışmalar, Üretici Çekişmeli Ağlar (GAN'lar) gibi temel yöntemlerden ChatGPT gibi modern Büyük Dil Modelleri'ne (LLM'ler) kadar çeşitli teknikleri ele almaktadır. Amaç, yapay metin, haber ve kod üretimi için kullanılan yaklaşımları, karşılaşılan zorlukları ve değerlendirme metriklerini bütünsel bir bakış açısıyla sunmaktır.

## Genel Bakış

[cite_start]Sentetik veri, gerçek dünyadan toplanan verilerin yerine veya bu verileri tamamlamak amacıyla algoritmik olarak üretilen yapay verilerdir[cite: 1879]. [cite_start]Özellikle veri toplamanın maliyetli, zaman alıcı veya gizlilik endişeleri taşıdığı durumlarda büyük bir potansiyel sunmaktadır[cite: 1781, 1899]. Bu derleme, sentetik veri üretiminin aşağıdaki temel boyutlarını inceler:

* [cite_start]**Motivasyon:** Veri kıtlığı, maliyet, ölçeklenebilirlik, kontrol edilebilirlik ve gizliliğin korunması gibi sentetik veriye olan ihtiyacı artıran faktörler[cite: 1898, 1926].
* [cite_start]**Üretim Teknikleri:** GAN'lar, LLM'ler (örn. GPT-3.5/4o), SMOTE gibi veri dengeleme yöntemleri ve bu modelleri metin üretimi için uyarlamaya yönelik stratejiler[cite: 15, 1771, 5024].
* [cite_start]**Değerlendirme:** Üretilen sentetik verinin kalitesini, çeşitliliğini ve göreve özgü faydasını ölçmek için kullanılan hem otomatik hem de insan-merkezli metrikler[cite: 309, 2496].
* [cite_start]**Zorluklar:** Model çöküşü (model collapse), yanlılık (bias), doğruluk (factuality), öznellik (subjectivity) ve etik kaygılar gibi sentetik veri üretiminin getirdiği riskler[cite: 1784, 1816, 3826, 4616].

---

## İncelenen Makaleler ve Özetleri

Bu bölümde, repoda incelenen temel makalelerin özetleri sunulmaktadır.

### 1. A Survey on Text Generation using Generative Adversarial Networks
* **Yazarlar:** Gustavo H. de Rosa, João P. Papa
* [cite_start]**Odak:** Bu çalışma, metin üretimi için Üretici Çekişmeli Ağların (GAN) kullanımını kapsamlı bir şekilde incelemektedir[cite: 5029].
* **Temel Bulgular:**
    * [cite_start]GAN'ların temel olarak görseller gibi sürekli veriler için tasarlandığı, metin gibi ayrık veriler için doğrudan uygun olmadığı vurgulanmaktadır[cite: 5031].
    * [cite_start]Bu zorluğun üstesinden gelmek için literatürde üç ana yaklaşım geliştirilmiştir[cite: 5032]:
        1.  [cite_start]**Gumbel-Softmax:** Kategorik dağılımlar için sürekli bir yaklaşım sunarak gradyanların geri yayılımını mümkün kılar[cite: 5163, 5164].
        2.  [cite_start]**Pekiştirmeli Öğrenme (RL):** Üreticiyi bir "ajan" olarak modelleyerek, ayrıştırıcıdan gelen sinyalleri bir "ödül" olarak kullanır ve politika gradyanları ile günceller[cite: 5172, 5173].
        3.  [cite_start]**Değiştirilmiş Eğitim Hedefleri:** GAN'ların kayıp fonksiyonlarını ayrık verilerle daha uyumlu hale getirmek için çeşitli modifikasyonlar önerir[cite: 5418].
    * [cite_start]Makale, bu yaklaşımları kullanan çalışmaları (2016-2020 arası) metodoloji, kullanılan veri setleri ve sonuçlar açısından eleştirel bir şekilde analiz eder[cite: 5034, 5104].

### 2. A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models
* **Yazarlar:** Hanqing Zhang, Haolin Song, Shaoyu Li, Ming Zhou, Dawei Song
* [cite_start]**Odak:** Bu derleme, Transformer tabanlı Önceden Eğitilmiş Dil Modelleri (PLM'ler) kullanarak Kontrol Edilebilir Metin Üretimi (CTG) alanındaki güncel yöntemleri sistemli bir şekilde inceler[cite: 2504, 2506].
* **Temel Bulgular:**
    * [cite_start]CTG; duygu, konu, stil, anahtar kelime gibi belirli kısıtlamalara uygun metinler üretmeyi amaçlar[cite: 2598, 2609].
    * [cite_start]PLM tabanlı CTG yaklaşımları üç ana kategoride sınıflandırılmıştır[cite: 2737, 2756]:
        1.  **Fine-tuning (İnce Ayar):** PLM'nin bir kısmını veya tamamını hedef göreve göre ayarlama. [cite_start]Bu kategori altında `prompt-based` öğrenme, `RL-inspired` yaklaşımlar ve `instruction tuning` gibi yöntemler bulunur[cite: 2741, 2757].
        2.  [cite_start]**Retraining/Refactoring (Yeniden Eğitme/Yeniden Yapılandırma):** PLM'nin orijinal mimarisini değiştirme veya sıfırdan büyük bir koşullu dil modeli eğitme[cite: 2747, 2758].
        3.  [cite_start]**Post-Processing (Sonradan İşleme):** PLM parametrelerini sabit tutarak, kod çözme (decoding) aşamasında üretilen metni yönlendirme veya yeniden sıralama[cite: 2752, 2761].
    * [cite_start]Makale, bu yöntemlerin avantaj ve dezavantajlarını, uygulama alanlarını ve gelecekteki araştırma yönelimlerini tartışır[cite: 2507].

### 3. Evaluating Synthetic Data Generation from User Generated Text
* **Yazarlar:** Jenny Chim, Julia Ive, Maria Liakata
* [cite_start]**Odak:** Bu çalışma, özellikle kullanıcı tarafından oluşturulan içeriklerden (UGC) üretilen sentetik metin verilerinin kalitesini değerlendirmek için bir çerçeve sunar[cite: 309].
* **Temel Bulgular:**
    * [cite_start]Sentetik UGC kalitesini değerlendirmek için üç temel kriter (desiderata) tanımlanmıştır[cite: 341]:
        1.  [cite_start]**Anlamın Korunması (Meaning Preservation):** Üretilen metnin orijinal metnin anlamını ne kadar koruduğu[cite: 341].
        2.  [cite_start]**Stilin Korunması (Style Preservation):** Yazarın idiolect'i (kişisel dil stili) gibi stilistik özelliklerin korunması[cite: 341, 441].
        3.  [cite_start]**Ayrışma (Divergence):** Gizlilik için bir vekil olarak, sentetik metnin orijinal metinden ne kadar farklılaştığı[cite: 341, 451].
    * [cite_start]Hem içsel (intrinsic) metrikler (örn. BERTScore, POS JSD) hem de dışsal (extrinsic) değerlendirmeler (örn. alt görev performansı, yazar profilleme riski) içeren kapsamlı bir değerlendirme çerçevesi önerilmiştir[cite: 342].
    * [cite_start]Çerçeve, farklı sentetik veri üretim stratejilerinin (örn. geri çeviri, DP-BART, LLM prompting) fayda ve gizlilik riskleri arasındaki dengeyi nasıl etkilediğini göstermek için çeşitli görevlerde test edilmiştir[cite: 344].

### 4. Synthetic Data Generation Using Large Language Models: Advances in Text and Code
* **Yazarlar:** Mihai Nadăş, Laura Dioşan, Andreea Tomescu
* [cite_start]**Odak:** Bu makale, Büyük Dil Modelleri'nin (LLM'ler) hem metin hem de kod alanlarında sentetik veri üretimini nasıl dönüştürdüğünü inceler[cite: 1780].
* **Temel Bulgular:**
    * [cite_start]LLM'ler, veri kıtlığı olan, maliyetli veya hassas veriler içeren senaryolarda gerçek dünya veri setlerini artırabilir veya onların yerini alabilir[cite: 1781].
    * [cite_start]Temel teknikler arasında `prompt-based` üretim, `retrieval-augmented` (RAG) pijamaları ve `iterative self-refinement` bulunmaktadır[cite: 1782].
    * [cite_start]**Zorluklar:** Üretilen metinlerdeki olgusal hatalar (halüsinasyonlar), stilistik veya dağılımsal gerçekçilik eksikliği, yanlılıkların güçlendirilmesi riski ve **model çöküşü** (model collapse) gibi önemli sorunlar tartışılmaktadır[cite: 1784, 1816].
    * [cite_start]**Çözüm Önerileri:** Sentetik çıktıların filtrelenmesi, ağırlıklandırılması ve kod alanında yürütme geri bildiriminden (execution feedback) yararlanarak pekiştirmeli öğrenme gibi stratejiler sunulmaktadır[cite: 1785, 1817].

### 5. Yapay Zeka Destekli Haber Metni Üretimi ve Çevirilerinin Karşılaştırmalı Bir Analizi: ChatGPT-4o Örneği
* **Yazarlar:** Burcu Türkmen, Sema Koçlu Maman
* [cite_start]**Odak:** Bu çalışma, ChatGPT-4o yapay zeka aracının ideolojik olarak hassas bir konuda (Filistin-İsrail çatışması) haber metni üretme ve bu metni taraflı (pro-Filistin ve pro-İsrail) bir şekilde çevirme yeteneğini vaka analizi yoluyla inceler[cite: 4564, 4565].
* **Temel Bulgular:**
    * [cite_start]YZ ile haber üretimi; hız, verimlilik ve maliyet açısından önemli avantajlar sunmaktadır[cite: 4570, 4890].
    * [cite_start]ChatGPT-4o'dan tarafsız bir metin üretmesi istendiğinde, modelin bunu başarabildiği, ancak taraflı çeviri taleplerinde anlamı büyük ölçüde koruyarak sadece eş anlamlı kelimelerle küçük değişiklikler yaptığı gözlemlenmiştir[cite: 4853, 4859, 4860].
    * [cite_start]Model ile yapılan "röportajda", YZ'nın tarafsız kalmaya programlandığını ve hassas konularda dengeli bir dil kullanmaya çalıştığını belirttiği aktarılmaktadır[cite: 4874].
    * [cite_start]Çalışma, YZ tarafından üretilen içeriklerin doğruluğunun ve tarafsızlığının sürekli olarak denetlenmesi gerektiğini, özellikle etik konuların göz ardı edilmemesi gerektiğini vurgulamaktadır[cite: 4893, 4895].

---

## Öne Çıkan Temel Konular ve Zorluklar

İncelenen makalelerden elde edilen ortak temalar ve zorluklar aşağıda özetlenmiştir:

* [cite_start]**Kontrol Edilebilirlik vs. Kalite Dengesi:** Üretilen metnin belirli özelliklere (konu, stil, duygu) uyması istenirken, bu kısıtlamalar metnin akıcılığını ve doğallığını olumsuz etkileyebilir[cite: 3245]. [cite_start]Özellikle `Post-Processing` yöntemleri kontrolü artırsa da metin kalitesini düşürebilir[cite: 3067].
* [cite_start]**Değerlendirmenin Karmaşıklığı:** Sentetik verinin "kaliteli" olup olmadığını belirlemek tek bir metrikle mümkün değildir[cite: 387]. [cite_start]Anlam, stil, çeşitlilik, doğruluk ve alt görev performansı gibi çok boyutlu bir değerlendirme gereklidir[cite: 342, 3266].
* [cite_start]**Öznellik (Subjectivity) Problemi:** YZ modelleri, mizah, ironi veya duygu gibi yoruma açık ve öznel konularda insan benzeri nüansları yansıtan sentetik veriler üretmekte zorlanmaktadır[cite: 3843]. [cite_start]Model performansı, görevin öznelliği arttıkça düşme eğilimindedir[cite: 3827].
* [cite_start]**Model Çöküşü (Model Collapse):** Modellerin sürekli olarak kendi ürettikleri sentetik verilerle eğitilmesi, zamanla çeşitliliğin kaybolmasına ve model performansının düşmesine neden olan bir risktir[cite: 1816, 2237]. [cite_start]Bu riski azaltmak için gerçek veri ile sentetik verinin karıştırılması önerilmektedir[cite: 2241].
* [cite_start]**Etik ve Yanlılık (Bias):** YZ modelleri, eğitildikleri verilerdeki yanlılıkları sentetik verilere yansıtabilir ve hatta güçlendirebilir[cite: 2169, 2224]. [cite_start]Özellikle haber üretimi gibi hassas alanlarda, bu durum dezenformasyon riskini artırmaktadır[cite: 4677].

---

## ⚠️ Sorumluluk Reddi ve Kullanım Amacı
Bu depoda özetlenen ve referans olarak gösterilen tüm akademik makaleler kamuya açık, çevrimiçi ve erişilebilir kaynaklardan temin edilmiştir.

Bu derlemenin ve ilgili materyallerin temel amacı, sentetik veri üretimi ve veri artırımı alanındaki mevcut bilimsel çalışmaları eğitim ve bilgilendirme hedefiyle bir araya getirmektir. Kullanıcıların, atıfta bulunulan her bir makalenin orijinal kaynağını incelemeleri ve o kaynağın belirttiği lisans koşullarına uymaları beklenmektedir. Tüm çalışmaların hakları orijinal yazarlarına ve yayıncılarına aittir.

---

## 📚 Makalelerin Kaynakları

[1] B. Türkmen ve S. Koçlu Maman, "Yapay Zeka Destekli Haber Metni Üretimi Ve Çevirilerinin Karşılaştırmalı Bir Analizi: Chatgpt-40 Örneği," İstanbul Üniversitesi Çeviribilim Dergisi, no. 21, s. 212–229, 2024.

[2] J. Chim, J. Ive, ve M. Liakata, "Evaluating Synthetic Data Generation from User Generated Text," Computational Linguistics, c. 51, no. 1, 2024.

[3] A. F. Deveci ve M. F. Esen, "Medikal Sentetik Veri Üretimiyle Veri Dengelemesi," Journal of Statistics & Applied Science, no. 5, s. 17–27, 2022.

[4] M. Nadăş, L. Dioşan, ve A. Tomescu, "Synthetic Data Generation Using Large Language Models: Advances in Text and Code," arXiv preprint arXiv:2503.14023, 2025.

[5] Z. Li, H. Zhu, Z. Lu, ve M. Yin, "Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations," arXiv preprint arXiv:2310.07849, 2023.

[6] H. Zhang, H. Song, S. Li, M. Zhou, ve D. Song, "A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models," J. ACM, c. 37, no. 4, Makale 111, 2023.

[7] G. H. de Rosa ve J. P. Papa, "A Survey on Text Generation using Generative Adversarial Networks," arXiv preprint arXiv:2212.11119, 2022.