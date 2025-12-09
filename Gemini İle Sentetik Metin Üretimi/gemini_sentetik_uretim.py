import google.generativeai as genai
import pandas as pd
import numpy as np
import torch
import time
import random
import sys
import io
from tqdm import tqdm
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from bert_score import score as bert_score

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# ========== GEMINI API AYARLARI ==========
print("="*70)
print("GEMINI API ILE SENTETIK VERI URETIMI")
print("="*70)

# API Key'inizi buraya girin (veya environment variable kullanın)
API_KEY = ""  # TODO: API key'inizi buraya yazın

# Alternatif: Environment variable'dan okuma
# import os
# API_KEY = os.getenv('GEMINI_API_KEY')

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("\n[OK] Gemini API baglantisi basarili!")
except Exception as e:
    print(f"\n[HATA] Gemini API baglantisi basarisiz: {e}")
    print("Lutfen API key'inizi kontrol edin: https://makersuite.google.com/app/apikey")
    sys.exit(1)

# ========== BERT MODELI (METRIKLER ICIN) ==========
print("\n[INFO] BERT modeli yukleniyor (metrik hesaplamalari icin)...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[OK] GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[UYARI] GPU bulunamadi, CPU kullaniliyor")

model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained(model_name)
bert_model = bert_model.to(device)
print(f"[OK] BERT modeli {device} uzerinde hazir!\n")


# ========== YARDIMCI FONKSIYONLAR ==========
def cumleleri_normalize(cumle):
    """Cümleyi karşılaştırma için normalize et"""
    import string
    cumle = cumle.lower()
    cumle = cumle.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(cumle.split())


def hesapla_perplexity(cumle, model, tokenizer, device):
    """Perplexity hesapla (BERT MLM ile)"""
    try:
        inputs = tokenizer(cumle, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity
    except:
        return float('inf')


# ========== GEMINI ILE SENTETIK VERI URETIMI ==========
def gemini_sentetik_uret(cumle, n_varyant=30, orijinal_cumleler=None, max_retry=3):
    """
    Gemini API kullanarak sentetik cümle üretimi

    Args:
        cumle: Kaynak cümle
        n_varyant: Üretilecek varyant sayısı
        orijinal_cumleler: Orijinal cümle listesi (tekrar kontrolü için)
        max_retry: API hata durumunda max deneme sayısı

    Returns:
        list: Üretilen sentetik cümleler
    """

    # Orijinal cümleleri normalize et
    if orijinal_cumleler:
        orijinal_normalize_set = {cumleleri_normalize(c) for c in orijinal_cumleler}
    else:
        orijinal_normalize_set = set()

    varyantlar = set()

    # Gemini promptu - Türkçe teknoloji haberleri için optimize edilmiş
    prompt = f"""Sen bir Türkçe dil uzmanısın. Aşağıdaki teknoloji haber başlığından {n_varyant} adet YENİ ve FARKLI haber başlığı üret.

KAYNAK BAŞLIK:
"{cumle}"

KURALLAR:
1. Her başlık orijinalden FARKLI olmalı (en az 2-3 kelime değişmeli)
2. Aynı anlam/konuyu korumalı ama farklı kelimeler kullanmalı
3. Doğal Türkçe olmalı, dilbilgisi kurallarına uygun
4. Teknoloji terminolojisini koruyabilirsin ama çeşitlendir
5. Her başlık yeni satırda, numarasız, sadece başlık metni
6. Kısa ve öz (5-15 kelime arası)
7. Gerçekçi haber başlığı formatında

ÖRNEKLERİ TAKLIT ETME, sadece mantığı anla:
Eğer kaynak "iPhone 15 Türkiye'de satışa çıktı" ise:
- "Apple'ın yeni telefonu Türk pazarında" ✓
- "iPhone 15 ülkemizde piyasaya sürüldü" ✓
- "iPhone 15 Türkiye'de satışa sunuldu" ✗ (çok benzer)

Şimdi {n_varyant} adet başlık üret (sadece başlıkları yaz, açıklama yapma):
"""

    retry_count = 0
    while retry_count < max_retry:
        try:
            # Gemini API çağrısı
            response = model.generate_content(prompt)

            # Yanıtı parse et
            if response and response.text:
                lines = response.text.strip().split('\n')

                for line in lines:
                    # Temizlik: numara, tire, nokta vb. temizle
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)  # "1. " veya "1) " gibi
                    line = re.sub(r'^[-*•]\s*', '', line)  # "- " veya "* " gibi
                    line = line.strip()

                    # Geçerlilik kontrolleri
                    if not line or len(line) < 10:
                        continue

                    kelime_sayisi = len(line.split())
                    if kelime_sayisi < 3 or kelime_sayisi > 20:
                        continue

                    # Orijinalden farklı olmalı
                    line_normalized = cumleleri_normalize(line)
                    cumle_normalized = cumleleri_normalize(cumle)

                    if line_normalized == cumle_normalized:
                        continue

                    # Orijinal cümlelerden farklı olmalı
                    if line_normalized in orijinal_normalize_set:
                        continue

                    # En az 2 kelime farklı olmalı
                    line_kelimeler = set(line_normalized.split())
                    cumle_kelimeler = set(cumle_normalized.split())
                    farkli_kelime_sayisi = len(line_kelimeler.symmetric_difference(cumle_kelimeler))

                    if farkli_kelime_sayisi >= 2:
                        varyantlar.add(line)

                # Yeterli varyant üretildiyse döngüden çık
                if len(varyantlar) >= n_varyant:
                    break

            # API rate limit için kısa bekleme
            time.sleep(0.5)

        except Exception as e:
            print(f"[UYARI] Gemini API hatasi (deneme {retry_count+1}/{max_retry}): {e}")
            retry_count += 1
            time.sleep(2)  # Hata durumunda daha uzun bekle

            if retry_count >= max_retry:
                print(f"[HATA] '{cumle}' icin varyant uretilemedi")
                break

    return list(varyantlar)[:n_varyant]


# ========== PERPLEXITY FILTRELEME ==========
def filtrele_perplexity(cumleler, model, tokenizer, device, esik=50.0):
    """Perplexity eşiğine göre filtrele"""
    print(f"\n[INFO] Perplexity filtreleme basliyor (esik: {esik})...")

    filtreli_cumleler = []
    perplexity_skorlari = []

    for cumle in tqdm(cumleler, desc="Perplexity hesaplaniyor"):
        ppl = hesapla_perplexity(cumle, model, tokenizer, device)
        perplexity_skorlari.append(ppl)

        if ppl <= esik:
            filtreli_cumleler.append(cumle)

    print(f"\n[SONUC] Perplexity Filtreleme:")
    print(f"  - Baslangic: {len(cumleler)} cumle")
    print(f"  - Filtrelenen: {len(filtreli_cumleler)} cumle")
    print(f"  - Elenen: {len(cumleler) - len(filtreli_cumleler)} cumle (%{(1 - len(filtreli_cumleler)/len(cumleler))*100:.1f})")
    if perplexity_skorlari:
        print(f"  - Ortalama perplexity: {np.mean(perplexity_skorlari):.2f}")
        print(f"  - Min/Max: {np.min(perplexity_skorlari):.2f} / {np.max(perplexity_skorlari):.2f}")

    return filtreli_cumleler, perplexity_skorlari


# ========== ANA URETIM PIPELINE ==========
def uret_3000_cumle_gemini(seed_cumleler, hedef_sayi=3000, perplexity_esik=50.0):
    """
    Gemini API ile 100 cümleden 3000 cümle üret
    """
    print(f"\n{'='*70}")
    print(f"[BASLIYOR] GEMINI ILE SENTETIK VERI URETIMI")
    print(f"[BASLANGIC] {len(seed_cumleler)} cumle")
    print(f"[HEDEF] {hedef_sayi} cumle")
    print(f"[MODEL] Gemini 1.5 Flash")
    print(f"[METRIK HESAPLAMA] BERT + GPU")
    print(f"[PERPLEXITY ESIK] {perplexity_esik}")
    print(f"{'='*70}\n")

    tum_sentetik = []
    her_cumleden = hedef_sayi // len(seed_cumleler)

    print(f"[AYAR] Her cumleden ~{her_cumleden} varyant uretiliyor...\n")

    orijinal_cumle_set = set(seed_cumleler)

    # Her seed cümle için varyant üret
    for idx, cumle in enumerate(tqdm(seed_cumleler, desc="Gemini ile uretiliyor"), 1):
        varyantlar = gemini_sentetik_uret(
            cumle,
            n_varyant=her_cumleden,
            orijinal_cumleler=seed_cumleler
        )

        # Sadece orijinallerden farklı olanları ekle
        for varyant in varyantlar:
            if varyant not in orijinal_cumle_set:
                tum_sentetik.append(varyant)

        # Progress bilgisi
        if idx % 10 == 0:
            print(f"\n[PROGRESS] {idx}/{len(seed_cumleler)} cumle islendi, {len(tum_sentetik)} varyant uretildi")

    print(f"\n[INFO] Ilk asamada {len(tum_sentetik)} cumle uretildi")

    # Tekrarları temizle
    print("[INFO] Tekrarlar temizleniyor...")
    tum_sentetik = list(set(tum_sentetik))
    tum_sentetik = [c for c in tum_sentetik if c not in orijinal_cumle_set]

    print(f"[INFO] Tekrar temizleme sonrasi: {len(tum_sentetik)} cumle")

    # Eksik varsa ek üretim
    if len(tum_sentetik) < hedef_sayi:
        eksik = hedef_sayi - len(tum_sentetik)
        print(f"\n[UYARI] {eksik} cumle eksik, ek uretim yapiliyor...")

        ek_uretim_count = 0
        while len(tum_sentetik) < hedef_sayi and ek_uretim_count < 50:
            rastgele_cumle = random.choice(seed_cumleler)
            ek_varyantlar = gemini_sentetik_uret(
                rastgele_cumle,
                n_varyant=20,
                orijinal_cumleler=seed_cumleler
            )

            for varyant in ek_varyantlar:
                if varyant not in orijinal_cumle_set and varyant not in tum_sentetik:
                    tum_sentetik.append(varyant)

            ek_uretim_count += 1

            if len(tum_sentetik) >= hedef_sayi:
                break

    final_veri = tum_sentetik[:hedef_sayi]

    # PERPLEXITY FILTRELEME
    print(f"\n{'='*70}")
    print(f"[ADIM 2] PERPLEXITY ILE KALITE KONTROLU")
    print(f"{'='*70}")

    filtreli_veri, perplexity_skorlari = filtrele_perplexity(
        final_veri,
        bert_model,
        tokenizer,
        device,
        esik=perplexity_esik
    )

    # Filtreleme sonrası eksik varsa ek üretim
    if len(filtreli_veri) < hedef_sayi:
        eksik = hedef_sayi - len(filtreli_veri)
        print(f"\n[UYARI] Filtreleme sonrasi {len(filtreli_veri)} cumle kaldi")
        print(f"[INFO] {eksik} cumle daha uretiliyor (perplexity kontrollü)...")

        ek_uretim_count = 0
        while len(filtreli_veri) < hedef_sayi and ek_uretim_count < 100:
            rastgele_cumle = random.choice(seed_cumleler)
            ek_varyantlar = gemini_sentetik_uret(
                rastgele_cumle,
                n_varyant=10,
                orijinal_cumleler=seed_cumleler
            )

            for varyant in ek_varyantlar:
                if varyant not in orijinal_cumle_set and varyant not in filtreli_veri:
                    ppl = hesapla_perplexity(varyant, bert_model, tokenizer, device)
                    if ppl <= perplexity_esik:
                        filtreli_veri.append(varyant)
                        perplexity_skorlari.append(ppl)

            ek_uretim_count += 1

            if len(filtreli_veri) >= hedef_sayi:
                break

    final_veri = filtreli_veri[:hedef_sayi]
    final_perplexity = perplexity_skorlari[:hedef_sayi]

    print(f"\n{'='*70}")
    print(f"[OK] TAMAMLANDI!")
    print(f"[SONUC] Toplam uretilen: {len(final_veri)} cumle")
    print(f"{'='*70}\n")

    return final_veri, final_perplexity


# ========== METRIK ANALIZI ==========
def metrik_analizi(orijinal_cumleler, sentetik_cumleler, perplexity_skorlari=None):
    """
    5 temel metrik + görselleştirme
    """
    print("\n\n" + "="*70)
    print("5 TEMEL METRIK ANALIZI")
    print("="*70)

    # ========== 1. TEKIL ORAN ==========
    print("\n[1] TEKIL ORAN (Uniqueness)")
    print("-"*70)

    tekil_sayisi = len(set(sentetik_cumleler))
    tekil_oran = tekil_sayisi / len(sentetik_cumleler)

    print(f"  Toplam cumle: {len(sentetik_cumleler)}")
    print(f"  Tekil cumle: {tekil_sayisi}")
    print(f"  Tekrarli cumle: {len(sentetik_cumleler) - tekil_sayisi}")
    print(f"  TEKIL ORAN: %{tekil_oran * 100:.2f}")

    if tekil_oran >= 0.95:
        print(f"  Durum: ✓ MUKEMMEL")
    elif tekil_oran >= 0.85:
        print(f"  Durum: ~ IYI")
    else:
        print(f"  Durum: ⚠ IYILESTIRILEBILIR")


    # ========== 2. BERTSCORE F1 ==========
    print("\n[2] BERTSCORE F1 (Anlamsal Benzerlik)")
    print("-"*70)

    try:
        print("  Hesaplaniyor (bu 1-2 dakika surebilir)...")

        # TF-IDF ile en yakın orijinal cümleleri bul
        vectorizer = TfidfVectorizer(max_features=500)
        tum_cumleler = orijinal_cumleler + sentetik_cumleler
        tfidf_matrix = vectorizer.fit_transform(tum_cumleler)

        orijinal_vektorler = tfidf_matrix[:len(orijinal_cumleler)]
        sentetik_vektorler = tfidf_matrix[len(orijinal_cumleler):]

        benzerlikler = cosine_similarity(sentetik_vektorler, orijinal_vektorler)
        en_yakin_indeksler = benzerlikler.argmax(axis=1)

        reference_cumleler = [orijinal_cumleler[idx] for idx in en_yakin_indeksler]

        # BERTScore hesapla
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

        print(f"  Precision: {bertscore_precision:.4f}")
        print(f"  Recall: {bertscore_recall:.4f}")
        print(f"  BERTSCORE F1: {bertscore_f1:.4f}")

        if bertscore_f1 >= 0.85:
            print(f"  Durum: ✓ MUKEMMEL")
        elif bertscore_f1 >= 0.75:
            print(f"  Durum: ~ IYI")
        else:
            print(f"  Durum: ⚠ IYILESTIRILEBILIR")

    except Exception as e:
        print(f"  [HATA] BERTScore hesaplanamadi: {e}")
        bertscore_f1 = None


    # ========== 3. KELIME KAPSAMA ==========
    print("\n[3] KELIME KAPSAMA (Vocabulary Coverage)")
    print("-"*70)

    # Orijinal kelimeler
    orijinal_kelimeler = []
    for cumle in orijinal_cumleler:
        kelimeler = re.findall(r'\b\w+\b', cumle.lower())
        orijinal_kelimeler.extend(kelimeler)
    orijinal_set = set(orijinal_kelimeler)

    # Sentetik kelimeler
    sentetik_kelimeler = []
    for cumle in sentetik_cumleler:
        kelimeler = re.findall(r'\b\w+\b', cumle.lower())
        sentetik_kelimeler.extend(kelimeler)
    sentetik_set = set(sentetik_kelimeler)

    ortak_kelimeler = orijinal_set & sentetik_set
    kelime_kapsama = len(ortak_kelimeler) / len(orijinal_set)

    print(f"  Orijinal tekil kelime: {len(orijinal_set)}")
    print(f"  Sentetik tekil kelime: {len(sentetik_set)}")
    print(f"  Ortak kelimeler: {len(ortak_kelimeler)}")
    print(f"  KELIME KAPSAMA: %{kelime_kapsama * 100:.2f}")

    if kelime_kapsama >= 0.90:
        print(f"  Durum: ✓ MUKEMMEL")
    elif kelime_kapsama >= 0.75:
        print(f"  Durum: ~ IYI")
    else:
        print(f"  Durum: ⚠ IYILESTIRILEBILIR")


    # ========== 4. BENZERLIK SKORU ==========
    print("\n[4] BENZERLIK SKORU (TF-IDF Cosine Similarity)")
    print("-"*70)

    try:
        max_benzerlikler = benzerlikler.max(axis=1)
        ortalama_benzerlik = max_benzerlikler.mean()

        print(f"  Ortalama benzerlik: {ortalama_benzerlik:.4f}")
        print(f"  Min/Max: {max_benzerlikler.min():.4f} / {max_benzerlikler.max():.4f}")
        print(f"  Std sapma: {max_benzerlikler.std():.4f}")
        print(f"  BENZERLIK SKORU: {ortalama_benzerlik:.4f}")

        if 0.5 <= ortalama_benzerlik <= 0.7:
            print(f"  Durum: ✓ MUKEMMEL (ideal denge)")
        elif 0.4 <= ortalama_benzerlik <= 0.8:
            print(f"  Durum: ~ IYI")
        else:
            print(f"  Durum: ⚠ IYILESTIRILEBILIR")

        # Dağılım analizi
        yuksek = (max_benzerlikler > 0.8).sum()
        orta = ((max_benzerlikler > 0.5) & (max_benzerlikler <= 0.8)).sum()
        dusuk = (max_benzerlikler <= 0.5).sum()

        print(f"\n  Dagilim:")
        print(f"    Yuksek benzerlik (>0.8): {yuksek} (%{yuksek/len(max_benzerlikler)*100:.1f})")
        print(f"    Orta benzerlik (0.5-0.8): {orta} (%{orta/len(max_benzerlikler)*100:.1f})")
        print(f"    Dusuk benzerlik (<0.5): {dusuk} (%{dusuk/len(max_benzerlikler)*100:.1f})")

    except Exception as e:
        print(f"  [HATA] Benzerlik skoru hesaplanamadi: {e}")
        ortalama_benzerlik = None


    # ========== 5. PERPLEXITY SKORU ==========
    print("\n[5] PERPLEXITY SKORU (Anlamsal Dogallik)")
    print("-"*70)

    if perplexity_skorlari and len(perplexity_skorlari) > 0:
        ortalama_perplexity = np.mean(perplexity_skorlari)

        print(f"  Ortalama perplexity: {ortalama_perplexity:.2f}")
        print(f"  Min/Max: {np.min(perplexity_skorlari):.2f} / {np.max(perplexity_skorlari):.2f}")
        print(f"  Std sapma: {np.std(perplexity_skorlari):.2f}")
        print(f"  PERPLEXITY SKORU: {ortalama_perplexity:.2f}")

        if ortalama_perplexity <= 50:
            print(f"  Durum: ✓ MUKEMMEL (dogal cumleler)")
        elif ortalama_perplexity <= 70:
            print(f"  Durum: ~ IYI")
        else:
            print(f"  Durum: ⚠ IYILESTIRILEBILIR")
    else:
        print(f"  [UYARI] Perplexity skorlari bulunamadi")
        ortalama_perplexity = None


    # ========== GENEL DEGERLENDIRME ==========
    print("\n" + "="*70)
    print("GENEL DEGERLENDIRME")
    print("="*70)

    skorlar = {
        "Tekil Oran": tekil_oran >= 0.90,
        "BERTScore F1": bertscore_f1 is not None and bertscore_f1 >= 0.80,
        "Kelime Kapsama": kelime_kapsama >= 0.85,
        "Benzerlik Skoru": ortalama_benzerlik is not None and 0.45 <= ortalama_benzerlik <= 0.75,
        "Perplexity": ortalama_perplexity is not None and ortalama_perplexity <= 60
    }

    basarili = sum(skorlar.values())
    toplam = len(skorlar)

    print(f"\nBASARI ORANI: {basarili}/{toplam} metrik gecildi")

    print("\nMETRIK DURUMU:")
    for metrik, durum in skorlar.items():
        simge = "✓" if durum else "⚠"
        print(f"  {simge} {metrik}")

    if basarili >= 4:
        print(f"\nSONUC: ✓ MUKEMMEL - Sentetik veri yuksek kalitede!")
    elif basarili >= 3:
        print(f"\nSONUC: ~ IYI - Sentetik veri kullanilabilir durumdadir")
    else:
        print(f"\nSONUC: ⚠ IYILESTIRILEBILIR - Parametreleri ayarlayabilirsiniz")

    print("\n" + "="*70)


    # ========== GORSELESTIRME ==========
    print("\n[6] Gorsellestirme olusturuluyor...")

    try:
        fig = plt.figure(figsize=(16, 12))

        # 1. Kelime sayısı dağılımı
        plt.subplot(2, 3, 1)
        orijinal_kelime_sayilari = [len(c.split()) for c in orijinal_cumleler]
        sentetik_kelime_sayilari = [len(c.split()) for c in sentetik_cumleler]
        plt.hist(orijinal_kelime_sayilari, bins=20, alpha=0.6, label='Orijinal', color='blue', edgecolor='black')
        plt.hist(sentetik_kelime_sayilari, bins=20, alpha=0.6, label='Sentetik', color='orange', edgecolor='black')
        plt.xlabel('Kelime Sayisi')
        plt.ylabel('Frekans')
        plt.title('Kelime Sayisi Dagilimi')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Benzerlik dağılımı
        if ortalama_benzerlik is not None:
            plt.subplot(2, 3, 2)
            plt.hist(max_benzerlikler, bins=30, color='green', alpha=0.7, edgecolor='black')
            plt.xlabel('Benzerlik Skoru')
            plt.ylabel('Cumle Sayisi')
            plt.title('Benzerlik Dagilimi')
            plt.axvline(ortalama_benzerlik, color='red', linestyle='--',
                       label=f'Ort: {ortalama_benzerlik:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 3. Perplexity dağılımı
        if perplexity_skorlari and len(perplexity_skorlari) > 0:
            plt.subplot(2, 3, 3)
            plt.hist(perplexity_skorlari, bins=30, color='purple', alpha=0.7, edgecolor='black')
            plt.xlabel('Perplexity Skoru')
            plt.ylabel('Cumle Sayisi')
            plt.title('Perplexity Dagilimi')
            plt.axvline(np.mean(perplexity_skorlari), color='red', linestyle='--',
                       label=f'Ort: {np.mean(perplexity_skorlari):.2f}')
            plt.axvline(50, color='green', linestyle=':', label='Esik: 50')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 4. En sık kelimeler (Orijinal)
        plt.subplot(2, 3, 4)
        orijinal_kelime_frekansi = Counter(orijinal_kelimeler)
        top_10_orijinal = dict(orijinal_kelime_frekansi.most_common(10))
        plt.bar(range(len(top_10_orijinal)), list(top_10_orijinal.values()),
                alpha=0.8, color='blue', edgecolor='black')
        plt.xlabel('Kelimeler')
        plt.ylabel('Frekans')
        plt.title('En Sik 10 Kelime (Orijinal)')
        plt.xticks(range(len(top_10_orijinal)), list(top_10_orijinal.keys()),
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 5. En sık kelimeler (Sentetik)
        plt.subplot(2, 3, 5)
        sentetik_kelime_frekansi = Counter(sentetik_kelimeler)
        top_10_sentetik = dict(sentetik_kelime_frekansi.most_common(10))
        plt.bar(range(len(top_10_sentetik)), list(top_10_sentetik.values()),
                alpha=0.8, color='orange', edgecolor='black')
        plt.xlabel('Kelimeler')
        plt.ylabel('Frekans')
        plt.title('En Sik 10 Kelime (Sentetik)')
        plt.xticks(range(len(top_10_sentetik)), list(top_10_sentetik.keys()),
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 6. Metrik özeti
        plt.subplot(2, 3, 6)
        plt.axis('off')
        ozet = f"""
5 TEMEL METRIK OZETI

[1] Tekil Oran: %{tekil_oran*100:.1f}

[2] BERTScore F1: {bertscore_f1:.3f if bertscore_f1 else 'N/A'}

[3] Kelime Kapsama: %{kelime_kapsama*100:.1f}

[4] Benzerlik: {ortalama_benzerlik:.3f if ortalama_benzerlik else 'N/A'}

[5] Perplexity: {ortalama_perplexity:.1f if ortalama_perplexity else 'N/A'}

Veri:
  Orijinal: {len(orijinal_cumleler)}
  Sentetik: {len(sentetik_cumleler)}
  Carpan: {len(sentetik_cumleler)/len(orijinal_cumleler):.1f}x

Basari: {basarili}/{toplam} metrik
"""

        plt.text(0.1, 0.5, ozet, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.savefig('gemini_sentetik_metrikler.png', dpi=300, bbox_inches='tight')
        print("[OK] Grafik kaydedildi: gemini_sentetik_metrikler.png\n")

    except Exception as e:
        print(f"[HATA] Gorsellestirme yapilamadi: {e}\n")


# ========== ANA PROGRAM ==========
if __name__ == "__main__":

    # VERİ YÜKLE
    print("\n[INFO] Veri yukleniyor...")
    try:
        with open('tekonoloji-haber-baslıkları.csv', 'r', encoding='utf-8') as f:
            tum_satirlar = f.readlines()

        seed_cumleler = [line.strip() for line in tum_satirlar if line.strip()][:100]
        print(f"[OK] {len(seed_cumleler)} cumle yuklendi!")

        # İlk 5 örnek göster
        print("\n[ORNEK] Ilk 5 haber basligi:")
        for i, cumle in enumerate(seed_cumleler[:5], 1):
            print(f"  {i}. {cumle}")

    except FileNotFoundError:
        print("[HATA] 'tekonoloji-haber-baslıkları.csv' dosyasi bulunamadi!")
        sys.exit(1)


    # SENTETİK VERİ ÜRET
    print("\n" + "="*70)
    input("Gemini API ile uretim baslasin mi? (Enter'a basin)")

    sentetik_veri, perplexity_skorlari = uret_3000_cumle_gemini(
        seed_cumleler,
        hedef_sayi=3000,
        perplexity_esik=50.0
    )


    # KAYDET
    df_sonuc = pd.DataFrame({'haber_basligi': sentetik_veri})
    dosya_adi = 'gemini_sentetik_teknoloji_haberleri_3000.csv'
    df_sonuc.to_csv(dosya_adi, index=False, encoding='utf-8-sig')

    print(f"\n[KAYIT] Dosya kaydedildi: {dosya_adi}")

    print("\n[ORNEK] Ilk 10 sentetik haber basligi:")
    print("-" * 70)
    for i, cumle in enumerate(sentetik_veri[:10], 1):
        print(f"{i}. {cumle}")


    # METRİK ANALİZİ
    metrik_analizi(seed_cumleler, sentetik_veri, perplexity_skorlari)


    print("\n" + "="*70)
    print("TAMAMLANDI!")
    print("="*70)
    print(f"Cikti dosyasi: {dosya_adi}")
    print(f"Grafik dosyasi: gemini_sentetik_metrikler.png")
    print("="*70)
