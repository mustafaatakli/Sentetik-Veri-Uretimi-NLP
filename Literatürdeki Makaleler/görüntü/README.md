# Sentetik Görüntü Üretimi: Literatür İncelemesi ve Temel Kavramlar

Bu repo, sentetik görüntü üretimi (Synthetic Image Generation) alanındaki temel ve güncel akademik makaleleri özetlemek ve bu alandaki anahtar kavramları bir araya getirmek amacıyla oluşturulmuştur. Özellikle Çekişmeli Üretken Ağlar (Generative Adversarial Networks - GANs), Difüzyon Modelleri ve bu modellerin pratik uygulamaları üzerine odaklanılmıştır.

## İçindekiler
1.  [Özetlenen Makaleler](#özetlenen-makaleler)
    * [Image Synthesis with Adversarial Networks: A Comprehensive Survey and Case Studies](#paper1)
    * [Çekişmeli Üretken Ağ Modellerinin Görüntü Üretme Performanslarının İncelenmesi](#paper2)
    * [A Comprehensive Review of Synthetic Image Generation Methods in Remote Sensing](#paper3)
    * [Comprehensive Exploration of Synthetic Data Generation: A Survey](#paper4)
    * [DCGAN ile Üretilen Sentetik Görüntülerin Veri Boyutuna ve Epoch Sayısına Göre İncelenmesi](#paper5)
    * ["Sentetik Büyük Veri" İnşasında Kullanılan Desen Yayma Yaklaşımlarının İncelenmesi](#paper6)
    * [SoK: Can Synthetic Images Replace Real Data? A Survey of Utility and Privacy](#paper7)
2.  [Genel Çıkarımlar ve Temel Kavramlar](#genel-çıkarımlar-ve-temel-kavramlar)
    * [Popüler Modeller ve Mimariler](#popüler-modeller-ve-mimariler)
    * [Temel Zorluklar ve Çözüm Yaklaşımları](#temel-zorluklar-ve-çözüm-yaklaşımları)
    * [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
    * [Uygulama Alanları](#uygulama-alanları)
3.  [Hangi Model Ne Zaman Kullanılmalı? (Model Seçim Rehberi)](#model-seçim-rehberi)
4.  [Katkıda Bulunma](#katkıda-bulunma)
5.  [Lisans](#lisans)

---

## Özetlenen Makaleler

Bu bölümde, repoda incelenen makalelerin kısa özetleri ve ana bulguları yer almaktadır.

<a id="paper1"></a>
### 1. Image Synthesis with Adversarial Networks: A Comprehensive Survey and Case Studies (2020)
Bu makale, Çekişmeli Üretken Ağlar (GAN'lar) üzerine kapsamlı bir literatür taraması sunmaktadır.
-   **Ana Konu:** GAN tabanlı görüntü üretimi yöntemleri, mimarileri, kayıp fonksiyonları ve değerlendirme metrikleri.
-   **Öne Çıkan Modeller:** Standart GAN'dan başlayarak DCGAN, Conditional GAN (cGAN, InfoGAN, ACGAN), Auto-Encoder GAN (BiGAN, BEGAN), CycleGAN ve StackGAN gibi birçok temel GAN mimarisini kronolojik olarak inceler.
-   **Temel Bulgular:** GAN'ların eğitiminde karşılaşılan temel zorluklar olan **mod çökmesi (mode collapse)**, **kaybolan gradyanlar (vanishing gradients)** ve **yakınsama sorunları** vurgulanmıştır.
-   **Katkısı:** GAN alanına yeni başlayanlar için temel bir kaynak niteliğindedir ve farklı GAN türlerinin evrimini ve uygulama alanlarını (örneğin, görüntüden görüntüye çeviri, metinden görüntüye üretim) detaylı bir şekilde ortaya koyar.

<a id="paper2"></a>
### 2. Çekişmeli Üretken Ağ Modellerinin Görüntü Üretme Performanslarının İncelenmesi (2020)
Bu çalışma, yaygın olarak kullanılan yedi farklı GAN modelinin sentetik görüntü üretme performansını pratik olarak karşılaştırır.
-   **Ana Konu:** CGAN, DCGAN, InfoGAN, SGAN, ACGAN, WGAN-GP ve LSGAN modellerinin MNIST ve Fashion-MNIST veri setleri üzerindeki performans analizi.
-   **Öne Çıkan Katkı:** cGAN ve DCGAN'in avantajlarını birleştiren hibrit bir model olan **cDCGAN** önerilmiştir.
-   **Temel Bulgular:** -   **LSGAN**, üretilen görüntülerin sınıflandırma başarımı açısından en iyi sonuçları vermiştir.
    -   **DCGAN** ve **WGAN-GP**, görsel olarak daha net ve gürültüsüz görüntüler üretmiştir.
    -   Bu durum, "görsel kalite" ile "istatistiksel benzerlik" arasında bir denge (trade-off) olduğunu göstermektedir.
-   **Değerlendirme:** Üretilen görüntülerin kalitesini ölçmek için **Fréchet Inception Distance (FID)** metriği ve bir CNN sınıflandırıcısı kullanılmıştır.

<a id="paper3"></a>
### 3. A Comprehensive Review of Synthetic Image Generation Methods in Remote Sensing (2025)
Bu derleme, sentetik görüntü üretim tekniklerinin uzaktan algılama (uydu görüntüleri) gibi özel bir alandaki uygulamalarını inceler.
-   **Ana Konu:** Uydu görüntüleri için sentetik veri üretimi.
-   **Öne Çıkan Modeller:** GAN'ların (CycleGAN, Pix2Pix, StyleGAN2) yanı sıra, bu alandaki en yeni ve güçlü yaklaşım olan **Difüzyon Modelleri** (LDM, ControlNet, DALL-E 2) de incelenmiştir.
-   **Temel Bulgular:** Sentetik verilerle zenginleştirilmiş veri setleri, uydu görüntülerinde segmentasyon (IoU skorları) ve sınıflandırma doğruluğunu önemli ölçüde artırmaktadır. Difüzyon modellerinin, GAN'lara kıyasla daha kaliteli ve kontrol edilebilir sonuçlar sunduğu belirtilmiştir.
-   **Katkısı:** Üretken modellerin, kendine özgü zorlukları (yüksek çözünürlük, çoklu spektral bantlar, küçük nesneler) olan özel bir alana nasıl uyarlandığını gösterir.

<a id="paper4"></a>
### 4. Comprehensive Exploration of Synthetic Data Generation: A Survey (2024)
Bu çalışma, son on yılda yayınlanmış 417 sentetik veri üretimi modelini inceleyen devasa bir "Sistematizasyon Bilgisi" (SoK) makalesidir.
-   **Ana Konu:** Sadece GAN'ları değil, VAEs, Difüzyon Modelleri, Transformer'lar, RNN'ler gibi tüm sentetik veri üretimi paradigmalarını kapsar.
-   **Öne Çıkan Katkı:** Modelleri; veri tipi, performans, gizlilik ve eğitim süreci gibi çok sayıda kritere göre sınıflandırır. Hangi senaryoda hangi modelin seçilmesi gerektiğine dair pratik bir **karar ağacı (guideline)** sunar.
-   **Temel Bulgular:** -   Bilgisayarla görü, en baskın uygulama alanıdır.
    -   GAN'lar en popüler modeller olsa da, difüzyon modelleri ve transformer'lar hızla yükselmektedir.
    -   Modelleri karşılaştırmada standart metriklerin ve veri setlerinin eksikliği büyük bir sorundur.
    -   Gizlilik korumalı veri üretiminde genellikle daha basit modeller (Markov Zincirleri, Bayesian Ağları) veya özel olarak tasarlanmış GAN'lar tercih edilir.

<a id="paper5"></a>
### 5. DCGAN ile Üretilen Sentetik Görüntülerin Veri Boyutuna ve Epoch Sayısına Göre İncelenmesi (2023)
Bu makale, DCGAN modelinin performansını etkileyen iki temel hiperparametre olan veri seti büyüklüğü ve eğitim süresi (epoch sayısı) üzerine odaklanır.
-   **Ana Konu:** Eğitim verisi miktarının ve epoch sayısının üretilen görüntü kalitesine etkisi.
-   **Metodoloji:** CelebA yüz veri seti kullanılarak, 5.000 ve 10.000 görüntü ile farklı epoch sayılarında (10, 20, 30, 40) eğitimler yapılmış ve sonuçlar görsel olarak karşılaştırılmıştır.
-   **Temel Bulgular:** Üretilen sentetik görüntülerin kalitesi (netlik ve gerçekçilik), hem eğitimdeki veri miktarıyla hem de epoch sayısıyla **doğru orantılıdır**. Daha fazla veri ve daha uzun eğitim, daha iyi sonuçlar vermektedir.

<a id="paper6"></a>
### 6. "Sentetik Büyük Veri" İnşasında Kullanılan Desen Yayma Yaklaşımlarının İncelenmesi (2018)
Bu çalışma, derin öğrenme tabanlı üretken modellerden önce kullanılan geleneksel desen sentezleme (texture synthesis) yöntemlerini inceler.
-   **Ana Konu:** Piksel tabanlı, parça tabanlı (patch-based) ve piramit tabanlı desen yayma yaklaşımları.
-   **Metodoloji:** Küçük bir desen parçasından yola çıkarak daha büyük doku görüntüleri üretme performansı; hız, doğruluk (SSIM, MSE) ve gürültüye dayanıklılık açısından karşılaştırılmıştır.
-   **Temel Bulgular:** **Parça tabanlı yöntem**, hız ve doğruluk açısından en elverişli yöntem olarak öne çıkmıştır. Piksel tabanlı yöntemler ise aşırı yavaştır.
-   **Katkısı:** Modern üretken modellere tarihsel bir bağlam sunar ve doku üretimi probleminin temellerini açıklar.

<a id="paper7"></a>
### 7. SoK: Can Synthetic Images Replace Real Data? A Survey of Utility and Privacy (2025)
Bu makale, sentetik verilerin pratik faydası (utility) ile gizlilik (privacy) riskleri arasındaki dengeyi sistematik olarak inceler.
-   **Ana Konu:** "Sentetik veri, gerçek verinin yerini alabilir mi?" sorusuna fayda-gizlilik ekseninde yanıt arar.
-   **Öne Çıkan Katkı:** Sentetik veri paylaşımı için "üretim-örnekleme-sınıflandırma" (generation-sampling-classification) boru hattını tanımlar ve her aşamadaki gizlilik risklerini analiz eder.
-   **Temel Bulgular:** -   Sentetik veriyle eğitilmiş bir sınıflandırıcının yayınlanması, sentetik görüntülerin doğrudan yayınlanmasından daha güvenli olabilir, ancak bu durum veri setine bağlıdır.
    -   Yüksek kaliteli sentetik veriyle (özellikle difüzyon modellerinden elde edilen) eğitilmiş sınıflandırıcılar, gerçek veri üzerinde DP-SGD gibi gizlilik koruma yöntemleriyle eğitilmiş sınıflandırıcılardan daha iyi bir fayda-gizlilik dengesi sunabilir.
    -   Difüzyon modelleri, bu dengeyi kurmada genellikle GAN ve VAE'lerden daha başarılıdır.

---

## Genel Çıkarımlar ve Temel Kavramlar

İncelenen makalelerden elde edilen ortak sonuçlar aşağıda özetlenmiştir.

### Popüler Modeller ve Mimariler
-   **Çekişmeli Üretken Ağlar (GANs):** Alandaki en temel ve yaygın model ailesidir. `Generator` ve `Discriminator` arasındaki çekişmeli oyuna dayanır.
    -   **DCGAN:** Evrişimli katmanlar kullanarak görüntü üretiminde bir devrim yaratmıştır.
    -   **Conditional GANs (cGAN):** Üretim sürecini etiketler veya metin gibi ek bilgilerle yönlendirmeyi sağlar.
    -   **CycleGAN:** Eşleştirilmemiş veri setleri arasında (örneğin, yaz fotoğraflarını kışa çevirme) görüntüden görüntüye çeviri yapar.
    -   **StyleGAN:** Çok yüksek çözünürlüklü ve gerçekçi görüntüler üretir; stil transferi ve özellik ayrıştırma (disentanglement) konularında çok başarılıdır.
-   **Difüzyon Modelleri:** Görüntüye aşamalı olarak gürültü ekleme (forward process) ve bu süreci tersine çevirerek gürültüden görüntü üretme (reverse process) mantığına dayanır. Günümüzde en yüksek kalitede görüntüleri üreten state-of-the-art yaklaşımdır.
-   **Varyasyonel Otomatik Kodlayıcılar (VAEs):** Verinin olasılıksal bir gizli uzay (latent space) temsilini öğrenir. Özellikle özellik ayrıştırma ve yeni varyasyonlar üretme konusunda kullanışlıdır.
-   **Geleneksel Yöntemler:** Parça tabanlı desen sentezleme gibi yöntemler, derin öğrenme öncesi dönemin temelini oluşturur ve özellikle doku üretimi gibi sınırlı görevlerde hala geçerlidir.

### Temel Zorluklar ve Çözüm Yaklaşımları
-   **Eğitim Kararsızlığı:** GAN'ların eğitimindeki en büyük sorundur. WGAN ve LSGAN gibi farklı kayıp fonksiyonları bu sorunu hafifletmeyi amaçlar.
-   **Mod Çökmesi (Mode Collapse):** Üreticinin, ayrıştırıcıyı kandırmanın kolay birkaç yolunu bularak sürekli benzer örnekler üretmesidir. Farklı mimariler (örn. StyleGAN) ve eğitim teknikleri ile aşılmaya çalışılır.
-   **Fayda-Gizlilik Dengesi (Utility-Privacy Trade-off):** Üretilen sentetik verinin kullanışlı olması (yüksek fayda) ile orijinal verideki bireylerin gizliliğini ihlal etmemesi (yüksek gizlilik) arasındaki dengedir. Diferansiyel Gizlilik (DP) gibi teknikler bu dengeyi sağlamak için kullanılır.

### Değerlendirme Metrikleri
-   **Görsel Kalite ve Benzerlik:**
    -   **Fréchet Inception Distance (FID):** Üretilen görüntülerin dağılımının gerçek görüntülerin dağılımına ne kadar benzediğini ölçen en popüler metriktir. Düşük FID skoru daha iyidir.
    -   **Inception Score (IS):** Üretilen görüntülerin hem çeşitli (diverse) hem de tanınabilir (kaliteli) olup olmadığını ölçer. Yüksek IS skoru daha iyidir.
    -   **SSIM & MSE:** Daha geleneksel, piksel bazlı benzerlik metrikleridir.
-   **Gizlilik:**
    -   **Membership Inference Attack (MIA) Başarı Oranı:** Bir saldırganın, belirli bir verinin modelin eğitim setinde olup olmadığını ne kadar başarıyla tahmin edebildiğini ölçer. Düşük başarı oranı daha iyi gizlilik anlamına gelir.

### Uygulama Alanları
-   **Veri Artırma (Data Augmentation):** Özellikle tıp gibi az verinin olduğu alanlarda, mevcut veri setini sentetik örneklerle büyüterek modellerin performansını artırmak.
-   **Görüntüden Görüntüye Çeviri:** Stiller arası geçiş (Style Transfer), mevsim değiştirme, segmentasyon haritasından gerçekçi görüntü oluşturma.
-   **Metinden Görüntü Üretimi:** Verilen metinsel bir açıklamaya uygun görseller oluşturma.
-   **Gizlilik Korumalı Veri Paylaşımı:** Orijinal hassas veriyi (örn. hasta verileri) paylaşmak yerine, istatistiksel özelliklerini koruyan sentetik bir versiyonunu paylaşmak.

---

## Hangi Model Ne Zaman Kullanılmalı? (Model Seçim Rehberi)

Makalelerden çıkarılan sonuçlara göre, projenizin ihtiyacına yönelik model seçimi için aşağıdaki rehberi kullanabilirsiniz:

-   **En Yüksek Kalitede ve Gerçekçi Görüntüler Gerekiyorsa:**
    -   **Difüzyon Modelleri (Stable Diffusion, DALL-E):** Mevcut en iyi seçenektir. Özellikle metin veya başka koşullarla yönlendirilebilen, çok yüksek kaliteli sonuçlar sunar.

-   **Hızlı ve Göreceli Olarak Yüksek Kaliteli Görüntüler Gerekiyorsa:**
    -   **GAN'lar (örn. StyleGAN, BigGAN):** Difüzyon modellerine göre daha hızlı üretim (inference) yapabilirler ve hala çok güçlü sonuçlar verirler.

-   **Veri Setindeki Özellikleri (Stil, İçerik) Ayrıştırmak ve Kontrol Etmek Önemliyse:**
    -   **VAEs:** Gizli uzay (latent space) üzerinde manipülasyon yapmaya çok uygundur.
    -   **StyleGAN:** Stil katmanları sayesinde özellik ayrıştırmada (disentanglement) çok başarılıdır.

-   **Sıralı (Sequential) Veri Üretimi (Metin, Müzik) Gerekiyorsa:**
    -   **Transformer'lar** ve **RNN'ler:** Bu modeller sıralı veri üretimi için tasarlanmıştır.

-   **Hassas Verilerin Gizliliğini Koruyarak Sentetik Veri Üretmek Gerekiyorsa:**
    -   **Basit Modeller (Bayesian Networks, Markov Chains):** Düşük karmaşıklıktaki veriler için yorumlanabilir ve güvenli bir seçenektir.
    -   **Özelleştirilmiş GAN'lar (örn. DP-CGAN, PATE-GAN):** Diferansiyel Gizlilik gibi tekniklerle birleştirilmiş GAN'lar, karmaşık veriler için daha iyi bir fayda-gizlilik dengesi sunabilir.

-   **Çok Doğru ve Detaylı Etiketlere Sahip Sentetik Veri Gerekiyorsa:**
    -   **Sanal Ortamlar (Virtual Environments - örn. Unreal Engine, Unity):** Özellikle otonom sürüş gibi alanlarda, mükemmel etiketli (segmentasyon, derinlik vb.) sentetik veriler üretmek için kullanılır.

---

## ⚠️ Sorumluluk Reddi ve Kullanım Amacı
Bu depoda özetlenen ve referans olarak gösterilen tüm akademik makaleler kamuya açık, çevrimiçi ve erişilebilir kaynaklardan temin edilmiştir.

Bu derlemenin ve ilgili materyallerin temel amacı, sentetik veri üretimi ve veri artırımı alanındaki mevcut bilimsel çalışmaları eğitim ve bilgilendirme hedefiyle bir araya getirmektir. Kullanıcıların, atıfta bulunulan her bir makalenin orijinal kaynağını incelemeleri ve o kaynağın belirttiği lisans koşullarına uymaları beklenmektedir. Tüm çalışmaların hakları orijinal yazarlarına ve yayıncılarına aittir.

---

## 📚 Makalelerin Kaynakları

[1] P. Shamsolmoali, M. Zareapoor, E. Granger, H. Zhou, R. Wang, M. E. Celebi, and J. Yang, "Image Synthesis with Adversarial Networks: a Comprehensive Survey and Case Studies," arXiv preprint arXiv:2012.13736, 2020.

[2] J. Castro Lopes, J. L. Oliveira, and R. P. Lopes, "A comprehensive review of synthetic image generation methods in remote sensing," International Journal of Remote Sensing, vol. 46, no. 15, pp. 5773-5801, 2025.

[3] A. Bauer, M. Leznik, S. Trapp, M. Stenger, R. Leppich, S. Kounev, K. Chard, and I. Foster, "Comprehensive Exploration of Synthetic Data Generation: A Survey," arXiv preprint arXiv:2401.02524, 2024.

[4] Y. Chung, Y. Zhang, N. Marrouche, and J. Hamm, "SoK: Can Synthetic Images Replace Real Data? A Survey of Utility and Privacy of Synthetic Image Generation," arXiv preprint arXiv:2506.19360, 2025.

[5] G. Çelik and M. F. Talu, "Çekişmeli üretken ağ modellerinin görüntü üretme performanslarının incelenmesi," Balıkesir Üniversitesi Fen Bilimleri Enstitüsü Dergisi, vol. 22, no. 1, pp. 181-192, 2020.

[6] G. Çelik and M. F. Talu, "“Sentetik Büyük Veri” İnşasında Kullanılan Desen Yayma Yaklaşımlarının İncelenmesi," Anatolian Journal of Computer Sciences, vol. 3, no. 2, pp. 24-34, 2018.

[7] C. Koç and F. Özyurt, "DCGAN ile üretilen sentetik görüntülerin veri boyutuna ve epoch sayısına göre incelenmesi," Firat University Journal of Experimental and Computational Engineering, vol. 2, no. 1, pp. 32-37, 2023.