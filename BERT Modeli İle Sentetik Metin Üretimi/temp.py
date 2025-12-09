from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import random
import pandas as pd
from tqdm import tqdm
import sys
import io
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score as bert_score

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# ========== GPU AYARLARI (YENİ!) ==========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[OK] GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[UYARI] GPU bulunamadi, CPU kullaniliyor")

# ========== MODEL YÜKLE ==========
print("\n[INFO] Model yukleniyor...")
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# YENİ: Modeli GPU'ya taşı
model = model.to(device)
print(f"[OK] Model {device} uzerinde hazir!\n")


# ========== YARDIMCI FONKSİYON ==========
def cumleleri_normalize(cumle):
    """
    Cümleyi karşılaştırma için normalize et (küçük harf, noktalama temizle)
    """
    import string
    cumle = cumle.lower()
    cumle = cumle.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(cumle.split())  # Fazla boşlukları temizle


# ========== PERPLEXITY FONKSİYONLARI ==========
def hesapla_perplexity(cumle, model, tokenizer, device):
    """
    Cümlenin perplexity değerini hesapla.
    Düşük perplexity = doğal, yüksek perplexity = anlamsız
    """
    try:
        inputs = tokenizer(cumle, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity
    except:
        return float('inf')  # Hata durumunda sonsuz perplexity


def filtrele_perplexity(cumleler, model, tokenizer, device, esik=50.0):
    """
    Cümleleri perplexity eşiğine göre filtrele.
    esik: 50.0 önerilen (düşük=doğal, yüksek=anlamsız)
    """
    print(f"\n[INFO] Perplexity filtreleme basliyor (esik: {esik})...")

    filtreli_cumleler = []
    perplexity_skorlari = []

    for cumle in tqdm(cumleler, desc="Perplexity hesaplaniyor"):
        ppl = hesapla_perplexity(cumle, model, tokenizer, device)
        perplexity_skorlari.append(ppl)

        if ppl <= esik:
            filtreli_cumleler.append(cumle)

    # İstatistikler
    print(f"\n[SONUC] Perplexity Filtreleme:")
    print(f"  - Baslangic: {len(cumleler)} cumle")
    print(f"  - Filtrelenen: {len(filtreli_cumleler)} cumle")
    print(f"  - Elenen: {len(cumleler) - len(filtreli_cumleler)} cumle (%{(1 - len(filtreli_cumleler)/len(cumleler))*100:.1f})")
    print(f"  - Ortalama perplexity: {np.mean(perplexity_skorlari):.2f}")
    print(f"  - Min/Max: {np.min(perplexity_skorlari):.2f} / {np.max(perplexity_skorlari):.2f}")

    return filtreli_cumleler, perplexity_skorlari


# ========== İYİLEŞTİRİLMİŞ ANA FONKSİYON ==========
def bert_sentetik_uret(cumle, n_varyant=30, orijinal_cumleler=None, kullanilan_trigramlar=None):
    """
    Gelişmiş BERT sentetik veri üretimi:
    - Konservatif maskeleme (1-3 kelime) - daha doğal cümleler
    - Trigram çeşitlilik kontrolü (max 8 tekrar)
    - Temperature sampling (1.2) - çeşitlilik
    - Yapısal çeşitlilik kontrolü
    """
    kelimeler = cumle.split()
    varyantlar = set()

    if len(kelimeler) < 2:
        return []

    # Orijinal cümleleri normalize et (karşılaştırma için)
    if orijinal_cumleler:
        orijinal_normalize_set = {cumleleri_normalize(c) for c in orijinal_cumleler}
    else:
        orijinal_normalize_set = set()

    # Kullanılan trigramları takip et (çeşitlilik için)
    if kullanilan_trigramlar is None:
        kullanilan_trigramlar = Counter()

    denemeler = n_varyant * 8  # Daha fazla deneme

    for _ in range(denemeler):
        # YENİ: Daha konservatif maskeleme (1-3 kelime) - anlamsız cümle riski azalır
        if len(kelimeler) <= 4:
            max_mask = 1  # Kısa cümleler: sadece 1 kelime
        elif len(kelimeler) <= 7:
            max_mask = 2  # Orta cümleler: max 2 kelime
        else:
            max_mask = 3  # Uzun cümleler: max 3 kelime

        min_mask = 1  # En az 1 kelime maskele
        n_mask = random.randint(min_mask, max_mask)
        
        try:
            mask_pozisyonlar = random.sample(range(len(kelimeler)), n_mask)
        except:
            continue
        
        masked_kelimeler = kelimeler.copy()
        for poz in mask_pozisyonlar:
            masked_kelimeler[poz] = "[MASK]"
        
        masked_cumle = " ".join(masked_kelimeler)
        
        try:
            inputs = tokenizer(masked_cumle, return_tensors="pt", 
                             truncation=True, max_length=128)
            
            # YENİ: Input'ları da GPU'ya taşı
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            yeni_kelimeler = masked_kelimeler.copy()
            mask_token_indeksler = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            for mask_idx in mask_token_indeksler:
                # YENİ: Temperature sampling (daha çeşitli seçim)
                logits = outputs.logits[0, mask_idx]
                temperature = 1.2  # Yüksek temperature = daha çeşitli

                # Softmax ile olasılıkları hesapla
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

                # Top-K filtreleme (en iyi 15 kelime)
                top_k = 15
                top_probs, top_indices = probs.topk(top_k)

                # Olasılığa göre ağırlıklı seçim
                secilen_idx = torch.multinomial(top_probs, 1).item()
                secilen_id = top_indices[secilen_idx].item()

                tahmin_kelime = tokenizer.decode([secilen_id]).strip().replace('##', '').replace('#', '')

                if "[MASK]" in yeni_kelimeler:
                    idx = yeni_kelimeler.index("[MASK]")
                    yeni_kelimeler[idx] = tahmin_kelime
            
            yeni_cumle = " ".join(yeni_kelimeler)

            if "[MASK]" not in yeni_cumle and yeni_cumle != cumle:
                kelime_sayisi = len(yeni_cumle.split())
                if 2 <= kelime_sayisi <= 50:
                    # Normalize edilmiş haliyle orijinallerle karşılaştır
                    yeni_normalized = cumleleri_normalize(yeni_cumle)
                    cumle_normalized = cumleleri_normalize(cumle)

                    # Orijinal cümlelerden farklı olmalı
                    if yeni_normalized not in orijinal_normalize_set:
                        # Ek kontrol: Orijinal cümleden en az 2 kelime farklı olmalı
                        yeni_kelimeler_set = set(yeni_normalized.split())
                        cumle_kelimeler_set = set(cumle_normalized.split())
                        farkli_kelime_sayisi = len(yeni_kelimeler_set.symmetric_difference(cumle_kelimeler_set))

                        if farkli_kelime_sayisi >= 2:  # En az 2 kelime farklı olmalı

                            # YENİ: Trigram çeşitlilik kontrolü
                            yeni_cumle_kelimeleri = yeni_cumle.split()
                            trigram_kontrolu_basarili = True

                            # 3-kelimelik kalıpları kontrol et
                            if len(yeni_cumle_kelimeleri) >= 3:
                                for i in range(len(yeni_cumle_kelimeleri) - 2):
                                    trigram = ' '.join(yeni_cumle_kelimeleri[i:i+3])

                                    # Bu trigram çok fazla kullanıldı mı?
                                    if kullanilan_trigramlar[trigram] >= 8:  # Max 8 kez
                                        trigram_kontrolu_basarili = False
                                        break

                            if trigram_kontrolu_basarili:
                                varyantlar.add(yeni_cumle)

                                # Trigramları kaydet
                                if len(yeni_cumle_kelimeleri) >= 3:
                                    for i in range(len(yeni_cumle_kelimeleri) - 2):
                                        trigram = ' '.join(yeni_cumle_kelimeleri[i:i+3])
                                        kullanilan_trigramlar[trigram] += 1
        
        except:
            continue

        if len(varyantlar) >= n_varyant:
            break

    return list(varyantlar)[:n_varyant], kullanilan_trigramlar


# ========== İYİLEŞTİRİLMİŞ 100→3000 PİPELINE ==========
def uret_3000_cumle(seed_cumleler, hedef_sayi=3000):
    """
    100 cümleden 3000 cümle üret - İyileştirilmiş versiyon
    """
    print(f"\n{'='*60}")
    print(f"[BASLIYOR] Sentetik Veri Uretimi - OPTIMIZE EDILMIS")
    print(f"[BASLANGIC] {len(seed_cumleler)} cumle")
    print(f"[HEDEF] {hedef_sayi} cumle")
    print(f"[CIHAZ] {device}")
    print(f"[OZELLIK] Konservatif maskeleme (1-3 kelime)")
    print(f"[OZELLIK] Trigram cesitlilik kontrolu (max 8)")
    print(f"[OZELLIK] Temperature sampling (1.2)")
    print(f"[OZELLIK] Perplexity filtreleme (esik: 50.0)")
    print(f"{'='*60}\n")

    tum_sentetik = []
    her_cumleden = hedef_sayi // len(seed_cumleler)

    print(f"[AYAR] Her cumleden ~{her_cumleden} varyant uretiliyor...\n")

    # Orijinal cümleleri set olarak sakla (hızlı arama için)
    orijinal_cumle_set = set(seed_cumleler)

    # Global trigram tracker
    global_trigramlar = Counter()

    for cumle in tqdm(seed_cumleler, desc="Isleniyor"):
        varyantlar, global_trigramlar = bert_sentetik_uret(
            cumle,
            n_varyant=her_cumleden,
            orijinal_cumleler=seed_cumleler,
            kullanilan_trigramlar=global_trigramlar
        )
        # Sadece orijinallerden farklı olanları ekle
        for varyant in varyantlar:
            if varyant not in orijinal_cumle_set:
                tum_sentetik.append(varyant)

    print("\n[INFO] Tekrarlar ve orijinal cumleler temizleniyor...")
    # Tekrarları temizle ama orijinalleri kontrol et
    tum_sentetik = list(set(tum_sentetik))

    # Çift kontrol: orijinallerle aynı olanları çıkar
    tum_sentetik = [c for c in tum_sentetik if c not in orijinal_cumle_set]

    if len(tum_sentetik) < hedef_sayi:
        print(f"[UYARI] Eksik var, {hedef_sayi - len(tum_sentetik)} cumle daha uretiliyor...")

        while len(tum_sentetik) < hedef_sayi:
            rastgele_cumle = random.choice(seed_cumleler)
            ek_varyantlar, global_trigramlar = bert_sentetik_uret(
                rastgele_cumle,
                n_varyant=10,
                orijinal_cumleler=seed_cumleler,
                kullanilan_trigramlar=global_trigramlar
            )
            # Sadece orijinallerden farklı olanları ekle
            for varyant in ek_varyantlar:
                if varyant not in orijinal_cumle_set and varyant not in tum_sentetik:
                    tum_sentetik.append(varyant)
            # Hedef sayıya ulaştıysak dur
            if len(tum_sentetik) >= hedef_sayi:
                break

    # Trigram istatistikleri
    en_sik_trigramlar = global_trigramlar.most_common(10)
    print(f"\n[INFO] En sik kullanilan 10 trigram:")
    for trigram, sayi in en_sik_trigramlar:
        print(f"  '{trigram}': {sayi} kez")

    final_veri = tum_sentetik[:hedef_sayi]

    # YENİ: Perplexity ile kalite filtreleme
    print(f"\n{'='*60}")
    print(f"[ADIM 2] PERPLEXITY ILE KALITE KONTROLU")
    print(f"{'='*60}")

    filtreli_veri, perplexity_skorlari = filtrele_perplexity(
        final_veri,
        model,
        tokenizer,
        device,
        esik=50.0
    )

    # Eğer filtreleme sonucu hedef sayıdan az kaldıysa, ek üretim yap
    if len(filtreli_veri) < hedef_sayi:
        print(f"\n[UYARI] Filtreleme sonrasi {len(filtreli_veri)} cumle kaldi, {hedef_sayi - len(filtreli_veri)} cumle daha uretiliyor...")

        while len(filtreli_veri) < hedef_sayi:
            rastgele_cumle = random.choice(seed_cumleler)
            ek_varyantlar, global_trigramlar = bert_sentetik_uret(
                rastgele_cumle,
                n_varyant=10,
                orijinal_cumleler=seed_cumleler,
                kullanilan_trigramlar=global_trigramlar
            )

            # Perplexity kontrolü ile ekle
            for varyant in ek_varyantlar:
                if varyant not in orijinal_cumle_set and varyant not in filtreli_veri:
                    ppl = hesapla_perplexity(varyant, model, tokenizer, device)
                    if ppl <= 50.0:
                        filtreli_veri.append(varyant)
                        perplexity_skorlari.append(ppl)

            if len(filtreli_veri) >= hedef_sayi:
                break

    final_veri = filtreli_veri[:hedef_sayi]
    final_perplexity = perplexity_skorlari[:hedef_sayi]

    print(f"\n{'='*60}")
    print(f"[OK] TAMAMLANDI!")
    print(f"[SONUC] Toplam uretilen: {len(final_veri)} cumle")
    print(f"{'='*60}\n")

    return final_veri, final_perplexity


# ========== VERİ YÜKLE ==========
print("[INFO] Veri yukleniyor...")
with open('tekonoloji-haber-baslıkları.csv', 'r', encoding='utf-8') as f:
    tum_satirlar = f.readlines()

seed_cumleler = [line.strip() for line in tum_satirlar[0:] if line.strip()][:100]
print(f"[OK] {len(seed_cumleler)} cumle yuklendi!\n")


# ========== ÇALIŞTIR ==========
sentetik_veri, perplexity_skorlari = uret_3000_cumle(seed_cumleler, hedef_sayi=3000)


# ========== KAYDET ==========
df_sonuc = pd.DataFrame({'haber_basligi': sentetik_veri})
dosya_adi = 'sentetik_teknoloji_haberleri_3000.csv'
df_sonuc.to_csv(dosya_adi, index=False, encoding='utf-8-sig')

print(f"[KAYIT] Dosya kaydedildi: {dosya_adi}")

print("\n[ORNEK] Ilk 10 sentetik haber basligi:")
print("-" * 60)
for i, cumle in enumerate(sentetik_veri[:10], 1):
    print(f"{i}. {cumle}")


# ========== METRIK ANALIZI ==========
print("\n\n" + "="*70)
print("METRIK ANALIZI BASLIYOR...")
print("="*70)

def metrik_analizi(orijinal_cumleler, sentetik_cumleler, perplexity_skorlari=None):
    """
    Orijinal ve sentetik cümleler arasında detaylı metrik analizi yapar
    perplexity_skorlari: Her cümlenin perplexity değeri (opsiyonel)
    """

    # ========== 1. TEMEL İSTATİSTİKLER ==========
    print("\n" + "="*70)
    print("[1] TEMEL ISTATISTIKLER")
    print("="*70)

    # Orijinal istatistikler
    orijinal_kelime_sayilari = [len(c.split()) for c in orijinal_cumleler]
    orijinal_karakter_sayilari = [len(c) for c in orijinal_cumleler]

    # Sentetik istatistikler
    sentetik_kelime_sayilari = [len(c.split()) for c in sentetik_cumleler]
    sentetik_karakter_sayilari = [len(c) for c in sentetik_cumleler]

    print(f"\nORIJINAL VERI:")
    print(f"  Cumle sayisi: {len(orijinal_cumleler)}")
    print(f"  Ortalama kelime sayisi: {np.mean(orijinal_kelime_sayilari):.2f}")
    print(f"  Medyan kelime sayisi: {np.median(orijinal_kelime_sayilari):.2f}")
    print(f"  Min/Max kelime: {min(orijinal_kelime_sayilari)} / {max(orijinal_kelime_sayilari)}")
    print(f"  Ortalama karakter: {np.mean(orijinal_karakter_sayilari):.2f}")
    print(f"  Tekil cumle: {len(set(orijinal_cumleler))}")
    print(f"  Tekil oran: %{(len(set(orijinal_cumleler)) / len(orijinal_cumleler) * 100):.2f}")

    print(f"\nSENTETIK VERI:")
    print(f"  Cumle sayisi: {len(sentetik_cumleler)}")
    print(f"  Ortalama kelime sayisi: {np.mean(sentetik_kelime_sayilari):.2f}")
    print(f"  Medyan kelime sayisi: {np.median(sentetik_kelime_sayilari):.2f}")
    print(f"  Min/Max kelime: {min(sentetik_kelime_sayilari)} / {max(sentetik_kelime_sayilari)}")
    print(f"  Ortalama karakter: {np.mean(sentetik_karakter_sayilari):.2f}")
    print(f"  Tekil cumle: {len(set(sentetik_cumleler))}")
    print(f"  Tekrarli cumle: {len(sentetik_cumleler) - len(set(sentetik_cumleler))}")
    print(f"  Tekil oran: %{(len(set(sentetik_cumleler)) / len(sentetik_cumleler) * 100):.2f}")
    print(f"  Uretim carpani: {len(sentetik_cumleler) / len(orijinal_cumleler):.1f}x")


    # ========== 2. BENZERLIK ANALIZI ==========
    print("\n" + "="*70)
    print("[2] BENZERLIK ANALIZI (Cosine Similarity)")
    print("="*70)

    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        tum_cumleler = orijinal_cumleler + sentetik_cumleler
        tfidf_matrix = vectorizer.fit_transform(tum_cumleler)

        orijinal_vektorler = tfidf_matrix[:len(orijinal_cumleler)]
        sentetik_vektorler = tfidf_matrix[len(orijinal_cumleler):]

        benzerlikler = cosine_similarity(sentetik_vektorler, orijinal_vektorler)
        max_benzerlikler = benzerlikler.max(axis=1)
        ort_benzerlikler = benzerlikler.mean(axis=1)

        print(f"\nBenzerlik Skorlari (0=farkli, 1=ayni):")
        print(f"  Ortalama maksimum benzerlik: {max_benzerlikler.mean():.4f}")
        print(f"  Medyan benzerlik: {np.median(max_benzerlikler):.4f}")
        print(f"  Standart sapma: {max_benzerlikler.std():.4f}")
        print(f"  Min/Max benzerlik: {max_benzerlikler.min():.4f} / {max_benzerlikler.max():.4f}")

        yuksek_benzerlik = (max_benzerlikler > 0.8).sum()
        orta_benzerlik = ((max_benzerlikler > 0.5) & (max_benzerlikler <= 0.8)).sum()
        dusuk_benzerlik = (max_benzerlikler <= 0.5).sum()

        print(f"\nBenzerlik Dagilimi:")
        print(f"  Yuksek benzerlik (>0.8): {yuksek_benzerlik} cumle (%{yuksek_benzerlik/len(max_benzerlikler)*100:.2f})")
        print(f"  Orta benzerlik (0.5-0.8): {orta_benzerlik} cumle (%{orta_benzerlik/len(max_benzerlikler)*100:.2f})")
        print(f"  Dusuk benzerlik (<0.5): {dusuk_benzerlik} cumle (%{dusuk_benzerlik/len(max_benzerlikler)*100:.2f})")

        # En benzer cümleler
        print(f"\n[ORNEK] En benzer 3 sentetik cumle:")
        en_benzer_idx = max_benzerlikler.argsort()[-3:][::-1]
        for idx in en_benzer_idx:
            en_benzer_orijinal = benzerlikler[idx].argmax()
            print(f"\n  Benzerlik: {max_benzerlikler[idx]:.4f}")
            print(f"  Sentetik: {sentetik_cumleler[idx]}")
            print(f"  Orijinal: {orijinal_cumleler[en_benzer_orijinal]}")

        # En farklı cümleler
        print(f"\n[ORNEK] En farkli 3 sentetik cumle:")
        en_farkli_idx = max_benzerlikler.argsort()[:3]
        for idx in en_farkli_idx:
            print(f"\n  Benzerlik: {max_benzerlikler[idx]:.4f}")
            print(f"  Sentetik: {sentetik_cumleler[idx]}")

    except Exception as e:
        print(f"[HATA] Benzerlik analizi yapilamadi: {e}")
        max_benzerlikler = None


    # ========== 3. KELIME DAĞILIMI ==========
    print("\n" + "="*70)
    print("[3] KELIME DAGILIMI ANALIZI")
    print("="*70)

    # Orijinal kelime dağılımı
    orijinal_tum_kelimeler = []
    for cumle in orijinal_cumleler:
        kelimeler = re.findall(r'\b\w+\b', cumle.lower())
        orijinal_tum_kelimeler.extend(kelimeler)
    orijinal_kelime_frekansi = Counter(orijinal_tum_kelimeler)

    # Sentetik kelime dağılımı
    sentetik_tum_kelimeler = []
    for cumle in sentetik_cumleler:
        kelimeler = re.findall(r'\b\w+\b', cumle.lower())
        sentetik_tum_kelimeler.extend(kelimeler)
    sentetik_kelime_frekansi = Counter(sentetik_tum_kelimeler)

    print(f"\nORIJINAL VERI:")
    print(f"  Toplam kelime: {len(orijinal_tum_kelimeler)}")
    print(f"  Tekil kelime: {len(orijinal_kelime_frekansi)}")
    print(f"  Type-Token Ratio: {len(orijinal_kelime_frekansi)/len(orijinal_tum_kelimeler):.4f}")

    print(f"\nSENTETIK VERI:")
    print(f"  Toplam kelime: {len(sentetik_tum_kelimeler)}")
    print(f"  Tekil kelime: {len(sentetik_kelime_frekansi)}")
    print(f"  Type-Token Ratio: {len(sentetik_kelime_frekansi)/len(sentetik_tum_kelimeler):.4f}")

    print(f"\nEn sik 15 kelime (ORIJINAL):")
    for kelime, sayi in orijinal_kelime_frekansi.most_common(15):
        print(f"  {kelime}: {sayi}")

    print(f"\nEn sik 15 kelime (SENTETIK):")
    for kelime, sayi in sentetik_kelime_frekansi.most_common(15):
        print(f"  {kelime}: {sayi}")


    # ========== 4. KELIME ÇEŞİTLİLİĞİ ==========
    print("\n" + "="*70)
    print("[4] KELIME CESITLILIGI KARSILASTIRMASI")
    print("="*70)

    orijinal_set = set(orijinal_kelime_frekansi.keys())
    sentetik_set = set(sentetik_kelime_frekansi.keys())

    ortak_kelimeler = orijinal_set & sentetik_set
    sadece_orijinal = orijinal_set - sentetik_set
    sadece_sentetik = sentetik_set - orijinal_set

    print(f"\nKelime Karsilastirmasi:")
    print(f"  Orijinal tekil kelime: {len(orijinal_set)}")
    print(f"  Sentetik tekil kelime: {len(sentetik_set)}")
    print(f"  Ortak kelimeler: {len(ortak_kelimeler)}")
    print(f"  Sadece orijinalde: {len(sadece_orijinal)}")
    print(f"  Sadece sentetikte: {len(sadece_sentetik)}")
    print(f"  Kelime kapsama orani: %{(len(ortak_kelimeler) / len(orijinal_set) * 100):.2f}")
    print(f"  Kelime zenginlestirme: {len(sentetik_set) / len(orijinal_set):.2f}x")

    if len(sadece_sentetik) > 0:
        print(f"\n[ORNEK] Yeni eklenen kelimeler (ilk 30):")
        yeni_kelimeler = list(sadece_sentetik)[:30]
        print(f"  {', '.join(yeni_kelimeler)}")


    # ========== 5. SENTETIK VERI KALITE RAPORU ==========
    print("\n" + "="*70)
    print("[5] SENTETIK VERI KALITE RAPORU")
    print("="*70)

    # Metrikleri hesapla
    tekil_oran = len(set(sentetik_cumleler)) / len(sentetik_cumleler)
    kelime_kapsama = len(ortak_kelimeler) / len(orijinal_set)

    # Type-Token Ratio
    ttr_orijinal = len(orijinal_kelime_frekansi) / len(orijinal_tum_kelimeler)
    ttr_sentetik = len(sentetik_kelime_frekansi) / len(sentetik_tum_kelimeler)

    # [A] ÇEŞİTLİLİK METRİKLERİ
    print("\n[A] CESITLILIK METRIKLERI")
    print("-" * 70)

    # A1. Tekil Oran
    if tekil_oran >= 0.95:
        durum_a1 = "✓"
    elif tekil_oran >= 0.85:
        durum_a1 = "~"
    else:
        durum_a1 = "⚠"
    print(f"\nA1. Tekil Oran (Uniqueness): %{tekil_oran * 100:.2f} {durum_a1}")
    print(f"    → Her cumle benzersiz mi?")
    print(f"    → {len(set(sentetik_cumleler))}/{len(sentetik_cumleler)} cumle tekil")

    # A2. Type-Token Ratio
    if ttr_sentetik >= ttr_orijinal * 0.9:
        durum_a2 = "✓"
    elif ttr_sentetik >= ttr_orijinal * 0.7:
        durum_a2 = "~"
    else:
        durum_a2 = "⚠"
    print(f"\nA2. Type-Token Ratio: {ttr_sentetik:.4f} {durum_a2}")
    print(f"    → Kelime cesitliligi orani")
    print(f"    → Orijinal: {ttr_orijinal:.4f}, Sentetik: {ttr_sentetik:.4f}")

    # [B] ANLAMSAL KALITE METRIKLERI
    print("\n\n[B] ANLAMSAL KALITE METRIKLERI")
    print("-" * 70)

    # B1. BERTScore (Zhang et al., 2019)
    print(f"\nB1. BERTScore")
    print(f"    → BERT embeddings ile anlamsal benzerlik (Precision/Recall/F1)")
    print(f"    → Hesaplaniyor (bu biraz zaman alabilir)...")

    try:
        # YENI YAKLASIM: Her sentetik cümle için en yakın orijinal cümleyi bul
        # TF-IDF ile en yakın orijinal cümleyi belirle
        print(f"    → TF-IDF ile en yakin orijinal cumleler bulunuyor...")

        vectorizer_bert = TfidfVectorizer(max_features=500)
        tum_cumleler_bert = orijinal_cumleler + sentetik_cumleler
        tfidf_matrix_bert = vectorizer_bert.fit_transform(tum_cumleler_bert)

        orijinal_vektorler_bert = tfidf_matrix_bert[:len(orijinal_cumleler)]
        sentetik_vektorler_bert = tfidf_matrix_bert[len(orijinal_cumleler):]

        # Her sentetik cümle için en yakın orijinal cümleyi bul
        benzerlikler_bert = cosine_similarity(sentetik_vektorler_bert, orijinal_vektorler_bert)
        en_yakin_orijinal_indeksler = benzerlikler_bert.argmax(axis=1)

        # Referans cümleleri: Her sentetik için en yakın orijinal
        reference_cumleler = [orijinal_cumleler[idx] for idx in en_yakin_orijinal_indeksler]

        print(f"    → BERTScore hesaplaniyor ({len(sentetik_cumleler)} cumle)...")

        P, R, F1 = bert_score(
            sentetik_cumleler,
            reference_cumleler,
            lang='tr',
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        bertscore_f1 = F1.mean().item()
        bertscore_precision = P.mean().item()
        bertscore_recall = R.mean().item()

        # Değerlendirme (F1 skoru baz alınarak)
        if bertscore_f1 >= 0.85:
            durum_b1 = "✓"
        elif bertscore_f1 >= 0.75:
            durum_b1 = "~"
        else:
            durum_b1 = "⚠"

        print(f"\n    Precision: {bertscore_precision:.4f}")
        print(f"    Recall:    {bertscore_recall:.4f}")
        print(f"    F1 Score:  {bertscore_f1:.4f} {durum_b1}")
        print(f"    → Ideal: F1 ≥ 0.85 (yuksek anlamsal benzerlik)")
        print(f"    → Her sentetik cumle en yakin orijinal ile karsilastirildi")

    except Exception as e:
        print(f"    [UYARI] BERTScore hesaplanamadi: {str(e)}")
        bertscore_f1 = None

    # B2. TF-IDF Cosine Similarity
    if max_benzerlikler is not None:
        ortalama_benzerlik = max_benzerlikler.mean()
        if 0.5 <= ortalama_benzerlik <= 0.7:
            durum_b2 = "✓"
        elif 0.4 <= ortalama_benzerlik <= 0.8:
            durum_b2 = "~"
        else:
            durum_b2 = "⚠"
        print(f"\nB2. TF-IDF Cosine Similarity: {ortalama_benzerlik:.2f} {durum_b2}")
        print(f"    → Anlamsal benzerlik dengesi")
        print(f"    → Ideal aralik: 0.5-0.7 (ne cok benzer, ne cok farkli)")

    # B3. Perplexity
    if perplexity_skorlari is not None and len(perplexity_skorlari) > 0:
        ortalama_perplexity = np.mean(perplexity_skorlari)
        if ortalama_perplexity <= 50:
            durum_b3 = "✓"
        elif ortalama_perplexity <= 70:
            durum_b3 = "~"
        else:
            durum_b3 = "⚠"
        print(f"\nB3. Perplexity (BERT MLM): {ortalama_perplexity:.2f} {durum_b3}")
        print(f"    → Cumleler dogal/anlamli mi?")
        print(f"    → Ideal: ≤50 (dusuk = dogal, yuksek = anlamsiz)")
        print(f"    → Min/Max: {np.min(perplexity_skorlari):.2f} / {np.max(perplexity_skorlari):.2f}")

    # [C] KAPSAM METRIKLERI
    print("\n\n[C] KAPSAM METRIKLERI")
    print("-" * 70)

    # C1. Vocabulary Coverage
    if kelime_kapsama >= 0.90:
        durum_c1 = "✓"
    elif kelime_kapsama >= 0.75:
        durum_c1 = "~"
    else:
        durum_c1 = "⚠"
    print(f"\nC1. Vocabulary Coverage: %{kelime_kapsama * 100:.2f} {durum_c1}")
    print(f"    → Orijinal kelimelerin ne kadari korundu?")
    print(f"    → {len(ortak_kelimeler)}/{len(orijinal_set)} kelime ortak")

    # C2. Vocabulary Enrichment
    zenginlestirme = len(sentetik_set) / len(orijinal_set)
    if zenginlestirme >= 1.1:
        durum_c2 = "✓"
    elif zenginlestirme >= 0.95:
        durum_c2 = "~"
    else:
        durum_c2 = "⚠"
    print(f"\nC2. Vocabulary Enrichment: {zenginlestirme:.2f}x {durum_c2}")
    print(f"    → Yeni kelimeler eklendi mi?")
    print(f"    → Orijinal: {len(orijinal_set)}, Sentetik: {len(sentetik_set)}")

    # [D] GENEL DEGERLENDIRME
    print("\n\n" + "="*70)
    print("GENEL DEGERLENDIRME")
    print("="*70)

    # Güçlü yönleri topla
    guclu_yonler = []
    if tekil_oran >= 0.95:
        guclu_yonler.append("Tum cumleler benzersiz (%{:.2f})".format(tekil_oran * 100))
    if kelime_kapsama >= 0.90:
        guclu_yonler.append("Vocabulary coverage mukemmel (%{:.2f})".format(kelime_kapsama * 100))
    if max_benzerlikler is not None and 0.5 <= ortalama_benzerlik <= 0.7:
        guclu_yonler.append("Anlamsal benzerlik dengeli ({:.2f})".format(ortalama_benzerlik))
    if ttr_sentetik >= ttr_orijinal * 0.9:
        guclu_yonler.append("Kelime cesitliligi korunmus")
    if bertscore_f1 is not None and bertscore_f1 >= 0.85:
        guclu_yonler.append("BERTScore F1 yuksek ({:.4f}) - anlamsal benzerlik mukemmel".format(bertscore_f1))

    # Zayıf yönleri topla
    zayif_yonler = []
    if perplexity_skorlari is not None and ortalama_perplexity > 70:
        zayif_yonler.append("Perplexity yuksek ({:.2f} > 50) - bazi cumleler dogal degil".format(ortalama_perplexity))
    if max_benzerlikler is not None and ortalama_benzerlik > 0.8:
        zayif_yonler.append("Cosine similarity cok yuksek ({:.2f}) - orijinallere cok benziyor".format(ortalama_benzerlik))
    if tekil_oran < 0.95:
        zayif_yonler.append("Tekrar eden cumleler var (%{:.2f})".format((1-tekil_oran)*100))
    if kelime_kapsama < 0.90:
        zayif_yonler.append("Vocabulary coverage dusuk (%{:.2f})".format(kelime_kapsama * 100))
    if bertscore_f1 is not None and bertscore_f1 < 0.75:
        zayif_yonler.append("BERTScore F1 dusuk ({:.4f}) - anlamsal benzerlik zayif".format(bertscore_f1))

    print("\nGUCLU YONLER:")
    if guclu_yonler:
        for yön in guclu_yonler:
            print(f"  ✓ {yön}")
    else:
        print("  - Tespit edilemedi")

    print("\nIYILESTIRILEBILIR ALANLAR:")
    if zayif_yonler:
        for yön in zayif_yonler:
            print(f"  ⚠ {yön}")
    else:
        print("  - Tespit edilemedi")

    # Öneriler
    if perplexity_skorlari is not None and ortalama_perplexity > 70:
        print("\nONERILER:")
        print("  1. Maskeleme sayisini azalt (1-2 kelime)")
        print("  2. Temperature sampling'i duzenle (1.0-1.1)")
        print("  3. Perplexity esigini 40-45'e dusurebilirsiniz")
        print("  4. Daha fazla seed cumle kullanin")

    print("\nSONUC:")
    if len(guclu_yonler) >= 3 and len(zayif_yonler) <= 1:
        print("  Veri cesitlilik ve kapsam acisindan MUKEMMEL,")
        print("  BERT MLM yontemi basarili bir sekilde uygulanmis.")
    elif len(guclu_yonler) >= 2:
        print("  Veri IYI kalitede ve kullanilabilir durumdadir.")
        print("  Bazi iyilestirmelerle kalite artirilebilir.")
    else:
        print("  Veri KULLANILABILIR ancak onemli iyilestirme gereklidir.")

    print("\n" + "="*70 + "\n")


    # ========== 6. GÖRSELLEŞTIRME ==========
    print("[6] Gorsellestirme olusturuluyor...")

    try:
        fig = plt.figure(figsize=(16, 10))

        # 1. Kelime sayısı dağılımı
        plt.subplot(2, 3, 1)
        plt.hist(orijinal_kelime_sayilari, bins=20, alpha=0.6, label='Orijinal', color='blue', edgecolor='black')
        plt.hist(sentetik_kelime_sayilari, bins=20, alpha=0.6, label='Sentetik', color='orange', edgecolor='black')
        plt.xlabel('Kelime Sayisi')
        plt.ylabel('Frekans')
        plt.title('Kelime Sayisi Dagilimi')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Benzerlik dağılımı
        if max_benzerlikler is not None:
            plt.subplot(2, 3, 2)
            plt.hist(max_benzerlikler, bins=30, color='green', alpha=0.7, edgecolor='black')
            plt.xlabel('Maksimum Benzerlik Skoru')
            plt.ylabel('Cumle Sayisi')
            plt.title('Benzerlik Dagilimi')
            plt.axvline(max_benzerlikler.mean(), color='red', linestyle='--',
                       label=f'Ort: {max_benzerlikler.mean():.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 3. En sık kelimeler (Orijinal)
        plt.subplot(2, 3, 3)
        top_10_orijinal = dict(orijinal_kelime_frekansi.most_common(10))
        plt.bar(range(len(top_10_orijinal)), list(top_10_orijinal.values()),
                alpha=0.8, color='blue', edgecolor='black')
        plt.xlabel('Kelimeler')
        plt.ylabel('Frekans')
        plt.title('En Sik 10 Kelime (Orijinal)')
        plt.xticks(range(len(top_10_orijinal)), list(top_10_orijinal.keys()),
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 4. En sık kelimeler (Sentetik)
        plt.subplot(2, 3, 4)
        top_10_sentetik = dict(sentetik_kelime_frekansi.most_common(10))
        plt.bar(range(len(top_10_sentetik)), list(top_10_sentetik.values()),
                alpha=0.8, color='orange', edgecolor='black')
        plt.xlabel('Kelimeler')
        plt.ylabel('Frekans')
        plt.title('En Sik 10 Kelime (Sentetik)')
        plt.xticks(range(len(top_10_sentetik)), list(top_10_sentetik.keys()),
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 5. Kelime çeşitliliği
        plt.subplot(2, 3, 5)
        kategoriler = ['Ortak', 'Sadece\nOrijinal', 'Sadece\nSentetik']
        degerler = [len(ortak_kelimeler), len(sadece_orijinal), len(sadece_sentetik)]
        colors = ['green', 'blue', 'orange']
        plt.bar(kategoriler, degerler, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('Kelime Sayisi')
        plt.title('Kelime Cesitliligi')
        plt.grid(True, alpha=0.3, axis='y')

        # 6. Özet
        plt.subplot(2, 3, 6)
        plt.axis('off')
        ozet = f"""
OZET METRIKLER

Veri:
  Orijinal: {len(orijinal_cumleler)} cumle
  Sentetik: {len(sentetik_cumleler)} cumle
  Carpan: {len(sentetik_cumleler)/len(orijinal_cumleler):.1f}x

Istatistikler:
  Ort kelime (O): {np.mean(orijinal_kelime_sayilari):.1f}
  Ort kelime (S): {np.mean(sentetik_kelime_sayilari):.1f}
  Tekil oran: %{tekil_oran*100:.1f}

"""
        if max_benzerlikler is not None:
            ozet += f"Benzerlik: {max_benzerlikler.mean():.3f}\n"

        ozet += f"""
Kelime:
  Kapsama: %{kelime_kapsama*100:.1f}
  Zenginlik: {len(sentetik_set)/len(orijinal_set):.1f}x

Metrikler: Kategorize Edildi
(Cesitlilik, Anlamsal, Kapsam)
"""

        plt.text(0.1, 0.5, ozet, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('sentetik_veri_metrikleri.png', dpi=300, bbox_inches='tight')
        print("[OK] Grafik kaydedildi: sentetik_veri_metrikleri.png\n")

    except Exception as e:
        print(f"[HATA] Gorsellestirme yapilamadi: {e}\n")

# Metrik analizini çalıştır (perplexity skorları ile)
metrik_analizi(seed_cumleler, sentetik_veri, perplexity_skorlari)