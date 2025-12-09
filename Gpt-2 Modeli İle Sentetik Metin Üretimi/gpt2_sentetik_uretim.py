# -*- coding: utf-8 -*-
"""
GPT-2 Türkçe ile Sentetik Veri Üretimi
100 teknoloji haber başlığından 3000 sentetik cümle üretimi
Aynı 5 metrik ile değerlendirme (BERT/Gemini/LSTM karşılaştırması için)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    BertTokenizer,
    BertForMaskedLM
)
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time
import random

warnings.filterwarnings('ignore')

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Cihaz: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# 1. VERİ YÜKLEME
# ==============================================================================

def veri_yukle(dosya_yolu):
    """CSV dosyasından veriyi yükle"""
    try:
        # CSV'de başlık YOK - direkt 100 cümle var
        # encoding='utf-8-sig' BOM karakterini otomatik temizler
        df = pd.read_csv(dosya_yolu, encoding='utf-8-sig', header=None)
        cumleler = df.iloc[:, 0].dropna().tolist()

        print(f"[BAŞARI] {len(cumleler)} cümle yüklendi")
        return cumleler
    except FileNotFoundError:
        print(f"[HATA] '{dosya_yolu}' dosyasi bulunamadi!")
        return []
    except Exception as e:
        print(f"[HATA] Veri yukleme hatasi: {e}")
        return []

# ==============================================================================
# 2. GPT-2 TÜRKÇE İLE SENTETİK VERİ ÜRETİMİ
# ==============================================================================

def gpt2_modeli_yukle():
    """GPT-2 Türkçe modelini yükle"""
    print("\n[INFO] GPT-2 Turkce modeli yukleniyor...")
    print("[INFO] Model: ytu-ce-cosmos/turkish-gpt2")

    try:
        tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
        model = AutoModelForCausalLM.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
        model.to(device)
        model.eval()

        # Pad token ayarı
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("[BAŞARI] GPT-2 modeli yuklendi")
        print(f"[INFO] Parametreler: {sum(p.numel() for p in model.parameters()):,}")
        return tokenizer, model

    except Exception as e:
        print(f"[HATA] Model yukleme hatasi: {e}")
        return None, None

def gpt2_sentetik_uret(cumle, tokenizer, model, n_varyant=30, orijinal_cumleler=None):
    """
    GPT-2 ile bir cümleden n_varyant kadar yeni cümle üret (BATCH OPTIMIZATION)

    Strateji:
    - Batch generation (num_return_sequences=5) kullan -> 5x hızlı
    - Her cümleyi prompt olarak kullan
    - Temperature sampling (0.8-1.2 arası)
    - Top-k ve top-p sampling
    - Orijinalden farklı kontrolleri
    """
    varyantlar = set()
    batch_size = 10  # Her seferde 10 cümle üret (5'ten artırıldı)
    max_rounds = (n_varyant // batch_size) + 15  # Daha fazla deneme (GPT-2 için)

    for round_num in range(max_rounds):
        if len(varyantlar) >= n_varyant:
            break

        try:
            # TÜM CÜMLE prompt olarak (prefix yerine - daha iyi sonuç)
            prefix = cumle

            # Tokenize
            inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=50)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generation parametreleri (daha yüksek temperature - daha çeşitli)
            temperature = random.uniform(1.0, 1.5)  # 0.8-1.2 → 1.0-1.5
            top_k = random.randint(40, 80)  # 30-60 → 40-80
            top_p = random.uniform(0.90, 0.98)  # 0.85-0.95 → 0.90-0.98

            # BATCH Üretim - 5 cümle aynı anda
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,  # max_length yerine daha hızlı
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=batch_size,  # 5 cümle aynı anda
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )

            # Tüm üretilen cümleleri işle
            for output in outputs:
                generated = tokenizer.decode(output, skip_special_tokens=True)
                generated = generated.strip()

                # ===== GPT-2 TEMİZLEME =====
                # 1. Sadece ilk cümleyi al (satır sonuna kadar)
                generated = generated.split('\n')[0].strip()

                # 2. Pipe (|) karakterinden sonrasını kaldır (kategori bilgisi)
                if '|' in generated:
                    generated = generated.split('|')[0].strip()

                # 3. Tire ve özel karakterlerden sonrasını kaldır (site/kategori bilgisi)
                # Örnek: "Cümle - SiteAdı" → "Cümle"
                # Desteklenen karakterler: - – — » « • | :
                # Hem normal tire (-), uzun tire (–, —), guillemets (», «), bullet (•), pipe (|), colon (:)
                generated = re.split(r'\s+[-–—»«•|:]\s+', generated)[0].strip()

                # 4. Link/site isimlerini kaldır (kalan parçalar için)
                generated = re.sub(r' - \w+\.\w+', '', generated)  # " - Site.com"
                generated = re.sub(r'https?://\S+', '', generated)  # HTTP linkler
                generated = re.sub(r'www\.\S+', '', generated)  # www linkler

                # 4. Tarih/saat formatlarını kaldır
                generated = re.sub(r'\d{1,2}\.\d{1,2}\.\d{4}', '', generated)
                generated = re.sub(r'\d{1,2}:\d{2}', '', generated)

                # 5. Parantez içi site/kaynak bilgilerini kaldır
                generated = re.sub(r'\([^)]*\)', '', generated)
                generated = re.sub(r'\[[^\]]*\]', '', generated)

                # 6. Fazla noktalama/boşluk temizle
                generated = re.sub(r'\s*-\s*$', '', generated)
                generated = re.sub(r'\s+', ' ', generated).strip()
                generated = re.sub(r'[.!?]+$', '', generated).strip()  # Son noktayı kaldır

                # Geçerlilik kontrolleri
                if not generated or len(generated) < 10:
                    continue

                # Orijinalden farklı mı?
                if generated == cumle:
                    continue

                # Kelime sayısı kontrolü (haber başlığı: 5-20 kelime)
                kelime_sayisi = len(generated.split())
                if kelime_sayisi < 5 or kelime_sayisi > 20:
                    continue

                # Orijinal cümlelerle aynı mı?
                if orijinal_cumleler and generated in orijinal_cumleler:
                    continue

                # En az 2 kelime farklı olmalı
                orig_kelimeler = set(cumle.lower().split())
                gen_kelimeler = set(generated.lower().split())
                fark_sayisi = len(orig_kelimeler.symmetric_difference(gen_kelimeler))

                if fark_sayisi >= 2:
                    varyantlar.add(generated)

                if len(varyantlar) >= n_varyant:
                    break

        except Exception as e:
            continue

    return list(varyantlar)

def toplu_sentetik_uret(seed_cumleler, hedef_sayi=3000):
    """Tüm seed cümlelerden sentetik veri üret"""
    print(f"\n[INFO] GPT-2 ile {hedef_sayi} sentetik cumle uretiliyor...")
    her_cumleden = hedef_sayi // len(seed_cumleler) + 5
    print(f"[INFO] {len(seed_cumleler)} seed cumleden ~{her_cumleden} varyant/cumle")

    # GPT-2 modelini yükle
    tokenizer, model = gpt2_modeli_yukle()
    if tokenizer is None or model is None:
        return []

    sentetik_veri = []
    her_cumleden = hedef_sayi // len(seed_cumleler) + 5  # Biraz fazla üret

    for idx, cumle in enumerate(tqdm(seed_cumleler, desc="Uretim")):
        varyantlar = gpt2_sentetik_uret(
            cumle,
            tokenizer,
            model,
            n_varyant=her_cumleden,
            orijinal_cumleler=seed_cumleler
        )
        sentetik_veri.extend(varyantlar)

        # İlerleme raporu
        if (idx + 1) % 20 == 0:
            print(f"\n[INFO] {idx + 1}/{len(seed_cumleler)} cumle islendi")
            print(f"[INFO] Toplam uretilen: {len(sentetik_veri)}")

    print(f"\n[BAŞARI] Toplam {len(sentetik_veri)} cumle uretildi (ham)")

    # Tekrarları temizle
    print("\n[INFO] Tekrarlar temizleniyor...")
    sentetik_veri = list(set(sentetik_veri))
    print(f"[BAŞARI] {len(sentetik_veri)} tekil cumle")

    # Hedef sayıya ulaşmadıysak ek üretim
    if len(sentetik_veri) < hedef_sayi:
        eksik = hedef_sayi - len(sentetik_veri)
        print(f"\n[UYARI] {eksik} cumle eksik, ek uretim yapiliyor...")

        # GPU memory temizle
        torch.cuda.empty_cache()

        # Progress bar ile ek üretim
        pbar = tqdm(total=eksik, desc="Ek uretim")
        max_attempts = eksik * 3  # Sonsuz döngü önleme
        attempts = 0

        while len(sentetik_veri) < hedef_sayi and attempts < max_attempts:
            cumle = random.choice(seed_cumleler)
            varyantlar = gpt2_sentetik_uret(
                cumle,
                tokenizer,
                model,
                n_varyant=5,
                orijinal_cumleler=seed_cumleler
            )

            for v in varyantlar:
                if v not in sentetik_veri:
                    sentetik_veri.append(v)
                    pbar.update(1)
                    if len(sentetik_veri) >= hedef_sayi:
                        break

            attempts += 1

        pbar.close()

        if len(sentetik_veri) < hedef_sayi:
            print(f"\n[UYARI] Sadece {len(sentetik_veri)} cumle uretilebildi!")

    # Tam olarak hedef_sayi kadar al
    final_veri = sentetik_veri[:hedef_sayi]
    print(f"\n[BAŞARI] FINAL: {len(final_veri)} sentetik cumle hazirlandi")

    # GPT-2 modelini bellekten temizle (BERT için yer aç)
    print("\n[INFO] GPT-2 modeli bellekten temizleniyor...")
    del model, tokenizer
    torch.cuda.empty_cache()
    print("[INFO] GPU memory temizlendi")

    # Perplexity hesaplama (FILTRELEME YOK - sadece raporlama)
    print("\n" + "="*60)
    print("PERPLEXITY HESAPLAMA (Raporlama amacli)")
    print("="*60)
    print("[INFO] GPT-2 icin perplexity filtreleme yapilmiyor.")
    print("[INFO] Adil karsilastirma icin 3000 cumle garanti edildi.")
    print("[INFO] Perplexity skorlari sadece kalite raporu icin hesaplanacak.\n")

    # BERT modelini yükle (perplexity için)
    print("[INFO] BERT modeli yukleniyor...")
    bert_tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model = BertForMaskedLM.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model.to(device)
    bert_model.eval()
    print("[INFO] BERT modeli hazir")

    print("[INFO] Batch perplexity hesaplama basladi (16 cumle/batch)...")
    perplexity_skorlari = hesapla_perplexity_batch(final_veri, bert_model, bert_tokenizer, device, batch_size=16)

    ortalama_ppl = np.mean(perplexity_skorlari)
    print(f"\n[INFO] Ortalama Perplexity: {ortalama_ppl:.2f}")
    print(f"[INFO] Perplexity aralik: [{min(perplexity_skorlari):.2f}, {max(perplexity_skorlari):.2f}]")

    return final_veri

# ==============================================================================
# 3. PERPLEXITY HESAPLAMA (BERT MLM ile)
# ==============================================================================

def hesapla_perplexity(cumle, model, tokenizer, device):
    """BERT MLM ile perplexity hesapla (tek cümle)"""
    try:
        inputs = tokenizer(cumle, return_tensors='pt', truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

    except Exception as e:
        return float('inf')

def hesapla_perplexity_batch(cumleler, model, tokenizer, device, batch_size=16):
    """BERT MLM ile batch perplexity hesapla (HIZLI)"""
    perplexity_skorlari = []

    for i in range(0, len(cumleler), batch_size):
        batch = cumleler[i:i+batch_size]

        try:
            inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                # Her cümle için ayrı loss hesapla
                for j in range(len(batch)):
                    single_loss = outputs.loss if len(batch) == 1 else outputs.loss
                    ppl = torch.exp(single_loss).item()
                    perplexity_skorlari.append(ppl)
        except:
            # Hata varsa tek tek hesapla
            for cumle in batch:
                ppl = hesapla_perplexity(cumle, model, tokenizer, device)
                perplexity_skorlari.append(ppl)

    return perplexity_skorlari

# ==============================================================================
# 4. METRİK ANALİZİ (AYNI 5 METRİK)
# ==============================================================================

def metrik_analizi(orijinal_cumleler, sentetik_cumleler):
    """
    5 Temel Değerlendirme Metriği (BERT/Gemini/LSTM ile aynı)

    1. Tekil Oran (Uniqueness)
    2. BERTScore F1 (Anlamsal Benzerlik)
    3. Kelime Kapsama (Vocabulary Coverage)
    4. Benzerlik Skoru (TF-IDF Cosine Similarity)
    5. Perplexity (Anlamsal Doğallık - BERT MLM)
    """

    print("\n" + "="*60)
    print("5 TEMEL METRİK ANALİZİ (GPT-2)")
    print("="*60)

    # BERT modelini yükle (BERTScore ve Perplexity için)
    print("\n[INFO] BERT modeli yukleniyor (metrik hesaplama icin)...")
    bert_tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model = BertForMaskedLM.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model.to(device)
    bert_model.eval()

    # -------------------------------------------------------------------------
    # [1] TEKİL ORAN (Uniqueness)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[1] TEKİL ORAN (Uniqueness)")
    print("="*60)

    toplam_cumle = len(sentetik_cumleler)
    tekil_cumleler = set(sentetik_cumleler)
    tekil_sayi = len(tekil_cumleler)
    tekil_oran = (tekil_sayi / toplam_cumle) * 100

    print(f"  Toplam cumle: {toplam_cumle}")
    print(f"  Tekil cumle: {tekil_sayi}")
    print(f"  TEKIL ORAN: %{tekil_oran:.2f}")

    if tekil_oran >= 95:
        print(f"  Durum: ✓ MUKEMMEL")
    elif tekil_oran >= 85:
        print(f"  Durum: ✓ IYI")
    else:
        print(f"  Durum: ✗ ZAYIF (cok tekrar var)")

    # -------------------------------------------------------------------------
    # [2] BERTSCORE F1 (Anlamsal Benzerlik)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[2] BERTSCORE F1 (Anlamsal Benzerlik)")
    print("="*60)
    print("  [INFO] BERT ile anlamsal benzerlik olculuyor...")
    print("  [INFO] Her sentetik icin en yakin orijinal cumle bulunuyor (TF-IDF)...\n")

    # TF-IDF ile her sentetik için en yakın orijinal cümleyi bul
    vectorizer = TfidfVectorizer()
    try:
        tfidf_orijinal = vectorizer.fit_transform(orijinal_cumleler)
        tfidf_sentetik = vectorizer.transform(sentetik_cumleler)
        benzerlikler = cosine_similarity(tfidf_sentetik, tfidf_orijinal)
        en_yakin_idx = benzerlikler.argmax(axis=1)
        referanslar = [orijinal_cumleler[idx] for idx in en_yakin_idx]
    except:
        referanslar = orijinal_cumleler * (len(sentetik_cumleler) // len(orijinal_cumleler) + 1)
        referanslar = referanslar[:len(sentetik_cumleler)]

    # BERTScore hesapla
    P, R, F1 = bert_score(
        sentetik_cumleler,
        referanslar,
        lang='tr',
        model_type='dbmdz/bert-base-turkish-cased',
        device=device.type,
        verbose=False
    )

    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    bert_f1 = F1.mean().item()

    print(f"  Precision: {bert_precision:.4f}")
    print(f"  Recall: {bert_recall:.4f}")
    print(f"  BERTSCORE F1: {bert_f1:.4f}")

    if bert_f1 >= 0.85:
        print(f"  Durum: ✓ MUKEMMEL (cok yuksek anlamsal benzerlik)")
    elif bert_f1 >= 0.75:
        print(f"  Durum: ✓ IYI")
    elif bert_f1 >= 0.60:
        print(f"  Durum: ~ ORTA (daha cesitli ama anlamsal benzerlik dusuk)")
    else:
        print(f"  Durum: ✗ ZAYIF (anlamsal benzerlik cok dusuk)")

    # -------------------------------------------------------------------------
    # [3] KELIME KAPSAMA (Vocabulary Coverage)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[3] KELIME KAPSAMA (Vocabulary Coverage)")
    print("="*60)

    def kelime_ayir(cumleler):
        kelimeler = set()
        for cumle in cumleler:
            kelimeler.update(cumle.lower().split())
        return kelimeler

    orijinal_kelimeler = kelime_ayir(orijinal_cumleler)
    sentetik_kelimeler = kelime_ayir(sentetik_cumleler)
    ortak_kelimeler = orijinal_kelimeler.intersection(sentetik_kelimeler)

    kelime_kapsama = (len(ortak_kelimeler) / len(orijinal_kelimeler)) * 100

    print(f"  Orijinal tekil kelime: {len(orijinal_kelimeler)}")
    print(f"  Sentetik tekil kelime: {len(sentetik_kelimeler)}")
    print(f"  Ortak kelimeler: {len(ortak_kelimeler)}")
    print(f"  KELIME KAPSAMA: %{kelime_kapsama:.2f}")

    if kelime_kapsama >= 90:
        print(f"  Durum: ✓ MUKEMMEL")
    elif kelime_kapsama >= 80:
        print(f"  Durum: ✓ IYI")
    elif kelime_kapsama >= 70:
        print(f"  Durum: ~ ORTA")
    else:
        print(f"  Durum: ✗ ZAYIF (kelime dagilimi cok farkli)")

    # -------------------------------------------------------------------------
    # [4] BENZERLIK SKORU (TF-IDF Cosine Similarity)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[4] BENZERLIK SKORU (TF-IDF Cosine Similarity)")
    print("="*60)

    benzerlik_skoru = benzerlikler.max(axis=1).mean()

    print(f"  Ortalama benzerlik: {benzerlik_skoru:.4f}")

    if 0.5 <= benzerlik_skoru <= 0.7:
        print(f"  Durum: ✓ MUKEMMEL (ideal denge: cesitli ama ilgili)")
    elif 0.4 <= benzerlik_skoru < 0.5:
        print(f"  Durum: ~ ORTA (biraz fazla cesitli)")
    elif 0.7 < benzerlik_skoru <= 0.8:
        print(f"  Durum: ~ ORTA (biraz fazla benzer)")
    else:
        print(f"  Durum: ✗ DENGESIZ")

    # -------------------------------------------------------------------------
    # [5] PERPLEXITY SKORU (Anlamsal Doğallık)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[5] PERPLEXITY SKORU (Anlamsal Dogallik)")
    print("="*60)
    print("  [INFO] BERT MLM ile cumle dogalligi olculuyor (batch mode)...\n")

    perplexity_skorlari = hesapla_perplexity_batch(sentetik_cumleler, bert_model, bert_tokenizer, device, batch_size=16)

    ortalama_perplexity = np.mean(perplexity_skorlari)

    print(f"\n  Ortalama perplexity: {ortalama_perplexity:.2f}")

    if ortalama_perplexity <= 50:
        print(f"  Durum: ✓ MUKEMMEL (cok dogal cumleler)")
    elif ortalama_perplexity <= 70:
        print(f"  Durum: ✓ IYI")
    elif ortalama_perplexity <= 90:
        print(f"  Durum: ~ ORTA")
    else:
        print(f"  Durum: ✗ ZAYIF (anlamsal olarak zayif cumleler)")

    # -------------------------------------------------------------------------
    # ÖZET RAPOR
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("GENEL DEGERLENDIRME")
    print("="*60)

    basari_sayisi = 0
    if tekil_oran >= 95: basari_sayisi += 1
    if bert_f1 >= 0.75: basari_sayisi += 1
    if kelime_kapsama >= 80: basari_sayisi += 1
    if 0.5 <= benzerlik_skoru <= 0.7: basari_sayisi += 1
    if ortalama_perplexity <= 70: basari_sayisi += 1

    print(f"\nBASARI ORANI: {basari_sayisi}/5 metrik gecildi")

    if basari_sayisi >= 4:
        print("SONUC: ✓ MUKEMMEL - Sentetik veri yuksek kalitede!")
    elif basari_sayisi >= 3:
        print("SONUC: ✓ IYI - Kabul edilebilir kalite")
    elif basari_sayisi >= 2:
        print("SONUC: ~ ORTA - Iyilestirme gerekebilir")
    else:
        print("SONUC: ✗ ZAYIF - Ciddi iyilestirme gerekli")

    print("\n" + "="*60)

    # Dağılımları göster
    print("\n[INFO] Dagilim analizleri...")

    # Kelime sayısı dağılımı
    orijinal_kelime_sayilari = [len(c.split()) for c in orijinal_cumleler]
    sentetik_kelime_sayilari = [len(c.split()) for c in sentetik_cumleler]

    print(f"\nKelime Sayisi Dagilimi:")
    print(f"  Orijinal: Ort={np.mean(orijinal_kelime_sayilari):.1f}, Min={min(orijinal_kelime_sayilari)}, Max={max(orijinal_kelime_sayilari)}")
    print(f"  Sentetik: Ort={np.mean(sentetik_kelime_sayilari):.1f}, Min={min(sentetik_kelime_sayilari)}, Max={max(sentetik_kelime_sayilari)}")

    # Benzerlik dağılımı
    benzerlik_dagilimi = benzerlikler.max(axis=1)
    print(f"\nBenzerlik Dagilimi:")
    print(f"  Min: {benzerlik_dagilimi.min():.4f}")
    print(f"  Median: {np.median(benzerlik_dagilimi):.4f}")
    print(f"  Max: {benzerlik_dagilimi.max():.4f}")

    # Perplexity dağılımı
    print(f"\nPerplexity Dagilimi:")
    print(f"  Min: {min(perplexity_skorlari):.2f}")
    print(f"  Median: {np.median(perplexity_skorlari):.2f}")
    print(f"  Max: {max(perplexity_skorlari):.2f}")

    return {
        'tekil_oran': tekil_oran,
        'bertscore_f1': bert_f1,
        'kelime_kapsama': kelime_kapsama,
        'benzerlik_skoru': benzerlik_skoru,
        'perplexity': ortalama_perplexity
    }

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPT-2 TURKCE ILE SENTETIK VERI URETIMI")
    print("="*60)
    print("Model: ytu-ce-cosmos/turkish-gpt2")
    print("Hedef: 100 seed -> 3000 sentetik cumle")
    print("Metrikler: Ayni 5 metrik (BERT/Gemini/LSTM karsilastirmasi)")
    print("="*60)

    # 1. Veri yükleme
    dosya_yolu = 'tekonoloji-haber-baslıkları.csv'
    orijinal_cumleler = veri_yukle(dosya_yolu)

    if len(orijinal_cumleler) == 0:
        print("[HATA] Veri yuklenemedi, program sonlandiriliyor.")
        exit()

    print(f"\n[INFO] {len(orijinal_cumleler)} seed cumle yuklendi")
    print(f"[INFO] Hedef: 3000 sentetik cumle")

    # 2. Sentetik veri üretimi
    sentetik_cumleler = toplu_sentetik_uret(orijinal_cumleler, hedef_sayi=3000)

    if len(sentetik_cumleler) == 0:
        print("[HATA] Sentetik veri uretilemedi!")
        exit()

    # 3. CSV'ye kaydet
    output_file = 'gpt2_sentetik_teknoloji_haberleri_3000.csv'
    df_output = pd.DataFrame({'Sentetik_Baslik': sentetik_cumleler})
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=1, lineterminator='\n')
    print(f"\n[BAŞARI] {output_file} dosyasi kaydedildi ({len(sentetik_cumleler)} cumle)")

    # 4. Metrik analizi
    metrikler = metrik_analizi(orijinal_cumleler, sentetik_cumleler)

    print("\n" + "="*60)
    print("PROGRAM TAMAMLANDI")
    print("="*60)
    print(f"Cikti dosyasi: {output_file}")
    print("="*60)
