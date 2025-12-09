"""
T5 (Turkish-NLP) ile Sentetik Teknoloji Haber Başlığı Üretimi
- Model: Turkish-NLP/t5-efficient-base-turkish (Türkçe'ye özel)
- Input: 100 teknoloji haber başlığı
- Output: 3000 yeni sentetik cümle
- Metrikler: Tekil Oran, BERTScore F1, Kelime Kapsama, Benzerlik, Perplexity
- Karşılaştırma: google/mt5-base vs Turkish-NLP T5 ablation study
"""

import pandas as pd
import numpy as np
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# YAPILANDIRMA
# ============================================================

CSV_INPUT = 'tekonoloji-haber-baslıkları.csv'
CSV_OUTPUT = 't5_turkish_sentetik_teknoloji_haberleri_3000.csv'  # Farklı dosya adı
TARGET_SENTENCES = 3000
MODEL_NAME = 'Turkish-NLP/t5-efficient-base-turkish'  # Türkçe'ye özel T5

# GPU varsa kullan
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Cihaz: {device}")

# ============================================================
# 1. VERİ YÜKLEME (100 CÜMLE)
# ============================================================

def veri_yukle(csv_path):
    """CSV'den 100 cümle yükle"""
    print(f"[INFO] Veri yukleniyor: {csv_path}")

    # Başlık satırı YOK, header=None kullan
    df = pd.read_csv(csv_path, encoding='utf-8-sig', header=None)

    # İlk sütunu al
    sentences = df.iloc[:, 0].astype(str).tolist()

    # Boş/NaN değerleri filtrele
    sentences = [s.strip() for s in sentences if isinstance(s, str) and len(s.strip()) > 0]

    print(f"[INFO] Yuklenen cumle sayisi: {len(sentences)}")
    return sentences

# ============================================================
# 2. T5 MODEL YÜKLEME
# ============================================================

print(f"[INFO] Turkish-NLP T5 modeli yukleniyor: {MODEL_NAME}")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("[INFO] Turkish-NLP T5 modeli hazir")

# ============================================================
# 3. SENTETİK CÜMLE ÜRETİMİ (3000 ADET)
# ============================================================

def t5_uret(prompt_text, num_sequences=5):
    """
    T5 ile sentetik cümle üret

    Args:
        prompt_text: Orijinal başlık
        num_sequences: Kaç varyasyon üretilecek

    Returns:
        List[str]: Üretilen cümleler
    """
    # T5 için Türkçe prompt - paraphrase task kullan
    # mT5 "paraphrase:" prefix'i için eğitilmiş

    # Farklı prompt stratejileri dene
    strategies = [
        f"paraphrase: {prompt_text}",
        f"generate similar: {prompt_text}",
        f"rewrite: {prompt_text}",
        f"{prompt_text}",  # Prefix olmadan
    ]

    # Rastgele strateji seç
    input_text = random.choice(strategies)

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding=True
    ).to(device)

    # Rastgele parametreler (daha kontrollü)
    temperature = random.uniform(0.7, 1.1)  # Daha düşük = daha anlamlı
    top_k = random.randint(30, 60)  # Daha sınırlı kelime havuzu
    top_p = random.uniform(0.85, 0.95)  # Daha konsantre

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # max_length yerine max_new_tokens
            num_return_sequences=num_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,  # 2'den 3'e çıkar
            repetition_penalty=1.2,  # Tekrar cezası ekle
            early_stopping=True
        )

    # Decode
    generated_texts = []
    for output in outputs:
        generated = tokenizer.decode(output, skip_special_tokens=True)
        generated = generated.strip()

        # ===== T5 AGRESIF TEMİZLEME =====
        # 1. Sadece ilk satırı al
        generated = generated.split('\n')[0].strip()

        # 2. T5 özel token'larını kaldır (<extra_id_X>, <unk>, vb.)
        generated = re.sub(r'<[^>]+>', '', generated)  # <extra_id_0> gibi
        generated = re.sub(r'\[UNK\]', '', generated)

        # 3. Sadece Türkçe karakterleri tut (diğer dilleri filtrele)
        # Türkçe: a-zA-ZçÇğĞıİöÖşŞüÜ ve sayılar, noktalama
        # Eğer cümlede Kiril (Rusça) veya diğer alfabeler varsa atla
        if re.search(r'[а-яА-ЯёЁ]', generated):  # Kiril alfabesi
            continue
        if re.search(r'[α-ωΑ-Ω]', generated):  # Yunanca
            continue
        if re.search(r'[\u4e00-\u9fff]', generated):  # Çince
            continue

        # 4. Özel ayırıcı karakterlerden sonrasını kaldır
        # Karakterler: - – — » « • | :
        generated = re.split(r'\s+[-–—»«•|:]\s+', generated)[0].strip()

        # 5. Link/site isimlerini kaldır
        generated = re.sub(r'https?://\S+', '', generated)  # HTTP linkler
        generated = re.sub(r'www\.\S+', '', generated)  # www linkler
        generated = re.sub(r'\w+\.com', '', generated)  # .com siteleri
        generated = re.sub(r'\w+\.net', '', generated)  # .net siteleri

        # 6. Tarih/saat formatlarını kaldır
        generated = re.sub(r'\d{1,2}\.\d{1,2}\.\d{4}', '', generated)
        generated = re.sub(r'\d{1,2}:\d{2}', '', generated)

        # 7. Parantez içi site/kaynak bilgilerini kaldır
        generated = re.sub(r'\([^)]*\)', '', generated)
        generated = re.sub(r'\[[^\]]*\]', '', generated)

        # 8. Emoji ve özel semboller kaldır
        generated = re.sub(r'[😀-🙏🌀-🗿🚀-🛿]', '', generated)  # Emoji
        generated = re.sub(r'[™®©]', '', generated)  # Trademark

        # 9. Fazla noktalama/boşluk temizle
        generated = re.sub(r'\s+', ' ', generated).strip()
        generated = re.sub(r'[.!?]+$', '', generated).strip()
        generated = re.sub(r'^[.!?,;:]+', '', generated).strip()  # Başındaki noktalama

        # 10. Sadece Türkçe harf kontrolü (en az %70 Türkçe karakter)
        turkce_karakterler = len(re.findall(r'[a-zA-ZçÇğĞıİöÖşŞüÜ]', generated))
        toplam_karakterler = len(re.findall(r'\S', generated))

        if toplam_karakterler > 0:
            turkce_orani = turkce_karakterler / toplam_karakterler
            if turkce_orani < 0.7:  # %70'den az Türkçe karakter varsa atla
                continue

        # 11. Geçerlilik kontrolleri
        words = generated.split()
        word_count = len(words)

        # En az 5 kelime, en fazla 20 kelime
        if not (5 <= word_count <= 20 and len(generated) >= 15):
            continue

        # 12. Her kelime en az 2 harf içermeli
        if not all(len(word) >= 2 for word in words):
            continue

        # 13. Anlamsız tekrar kontrolü (aynı kelime 3+ kez)
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            if word_counts[word_lower] >= 3:  # Aynı kelime 3 kez tekrarlanıyorsa
                break
        else:
            # 14. "lar lar lar" gibi anlamsız tekrarları yakala
            # Ardışık aynı kelime varsa atla
            duplicate_found = False
            for i in range(len(words) - 1):
                if words[i].lower() == words[i+1].lower():
                    duplicate_found = True
                    break

            if not duplicate_found:
                # 15. Noktalama işareti ile başlayan kelime varsa atla
                if not any(word[0] in '.,;:!?' for word in words if len(word) > 0):
                    generated_texts.append(generated)

    return generated_texts


def uret_3000_cumle(orijinal_cumleler, target=3000):
    """3000 sentetik cümle üret"""
    print(f"\n[INFO] {target} adet sentetik cumle uretiliyor (T5)...")

    sentetik_cumleler = []
    tekil_set = set()

    batch_size = 10  # Her seferinde 10 cümle üret
    pbar = tqdm(total=target, desc="T5 Uretim")

    attempts = 0
    max_attempts = target * 20  # Sonsuz döngü önleme

    while len(sentetik_cumleler) < target and attempts < max_attempts:
        # Rastgele orijinal cümle seç
        orijinal = random.choice(orijinal_cumleler)

        # T5 ile üret
        try:
            generated_list = t5_uret(orijinal, num_sequences=batch_size)

            for generated in generated_list:
                # Tekil kontrolü
                if generated not in tekil_set:
                    sentetik_cumleler.append(generated)
                    tekil_set.add(generated)
                    pbar.update(1)

                    if len(sentetik_cumleler) >= target:
                        break

        except Exception as e:
            print(f"\n[UYARI] Uretim hatasi: {e}")
            attempts += 1
            continue

        attempts += 1

    pbar.close()

    if len(sentetik_cumleler) < target:
        print(f"[UYARI] Sadece {len(sentetik_cumleler)} tekil cumle uretilebildi")

    return sentetik_cumleler[:target]


# ============================================================
# 4. CSV KAYDETME
# ============================================================

def csv_kaydet(sentences, output_path):
    """Sentetik cümleleri CSV'ye kaydet"""
    df = pd.DataFrame({'Sentetik_Cumle': sentences})
    df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=1)
    print(f"[BAŞARI] {output_path} dosyasi kaydedildi ({len(sentences)} cumle)")


# ============================================================
# 5. METRİK ANALİZİ (5 TEMEL METRİK)
# ============================================================

def metrik_analizi(orijinal_cumleler, sentetik_cumleler):
    """5 temel metrik hesapla"""
    print("\n" + "="*60)
    print("5 TEMEL METRİK ANALİZİ (Turkish-NLP T5)")
    print("="*60)

    # BERT modeli yükle (metrik hesaplama için)
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    print("\n[INFO] BERT modeli yukleniyor (metrik hesaplama icin)...")
    bert_tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-turkish-cased').to(device)
    bert_model.eval()

    # ===== 1. TEKİL ORAN =====
    print("\n" + "="*60)
    print("[1] TEKİL ORAN (Uniqueness)")
    print("="*60)

    tekil_oran = len(set(sentetik_cumleler)) / len(sentetik_cumleler) * 100

    print(f"  Toplam cumle: {len(sentetik_cumleler)}")
    print(f"  Tekil cumle: {len(set(sentetik_cumleler))}")
    print(f"  TEKIL ORAN: %{tekil_oran:.2f}")
    print(f"  Durum: {'✓ MUKEMMEL' if tekil_oran >= 95 else '✗ DUSUK'}")

    # ===== 2. BERTSCORE F1 =====
    print("\n" + "="*60)
    print("[2] BERTSCORE F1 (Anlamsal Benzerlik)")
    print("="*60)
    print("  [INFO] BERT ile anlamsal benzerlik olculuyor...")

    from bert_score import score as bert_score

    # Her sentetik için en yakın orijinal cümleyi bul
    print("  [INFO] Her sentetik icin en yakin orijinal cumle bulunuyor (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    orijinal_vecs = vectorizer.fit_transform(orijinal_cumleler)
    sentetik_vecs = vectorizer.transform(sentetik_cumleler)
    similarities = cosine_similarity(sentetik_vecs, orijinal_vecs)
    en_yakin_indices = similarities.argmax(axis=1)
    referans_cumleler = [orijinal_cumleler[i] for i in en_yakin_indices]

    # BERTScore hesapla
    P, R, F1 = bert_score(
        sentetik_cumleler,
        referans_cumleler,
        lang='tr',
        model_type='dbmdz/bert-base-turkish-cased',
        device=device,
        verbose=False
    )

    precision_mean = P.mean().item()
    recall_mean = R.mean().item()
    f1_mean = F1.mean().item()

    print(f"\n  Precision: {precision_mean:.4f}")
    print(f"  Recall: {recall_mean:.4f}")
    print(f"  BERTSCORE F1: {f1_mean:.4f}")
    print(f"  Durum: {'✓ MUKEMMEL' if f1_mean >= 0.85 else '✓ IYI' if f1_mean >= 0.70 else '✗ DUSUK'}")

    # ===== 3. KELIME KAPSAMA =====
    print("\n" + "="*60)
    print("[3] KELIME KAPSAMA (Vocabulary Coverage)")
    print("="*60)

    orijinal_kelimeler = set(' '.join(orijinal_cumleler).lower().split())
    sentetik_kelimeler = set(' '.join(sentetik_cumleler).lower().split())
    ortak_kelimeler = orijinal_kelimeler & sentetik_kelimeler

    kapsama_orani = len(ortak_kelimeler) / len(orijinal_kelimeler) * 100

    print(f"  Orijinal tekil kelime: {len(orijinal_kelimeler)}")
    print(f"  Sentetik tekil kelime: {len(sentetik_kelimeler)}")
    print(f"  Ortak kelimeler: {len(ortak_kelimeler)}")
    print(f"  KELIME KAPSAMA: %{kapsama_orani:.2f}")
    print(f"  Durum: {'✓ MUKEMMEL' if kapsama_orani >= 95 else '✓ IYI' if kapsama_orani >= 80 else '✗ DUSUK'}")

    # ===== 4. BENZERLIK SKORU =====
    print("\n" + "="*60)
    print("[4] BENZERLIK SKORU (TF-IDF Cosine Similarity)")
    print("="*60)

    avg_similarity = similarities.max(axis=1).mean()

    print(f"  Ortalama benzerlik: {avg_similarity:.4f}")
    print(f"  Durum: {'✓ DENGELI' if 0.50 <= avg_similarity <= 0.75 else '✗ DENGESIZ'}")

    # ===== 5. PERPLEXITY =====
    print("\n" + "="*60)
    print("[5] PERPLEXITY SKORU (Anlamsal Dogallik)")
    print("="*60)
    print("  [INFO] BERT MLM ile cumle dogalligi olculuyor (batch mode)...")

    def bert_perplexity_batch(sentences, batch_size=16):
        perplexities = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = bert_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

            with torch.no_grad():
                outputs = bert_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

        return np.mean(perplexities)

    avg_perplexity = bert_perplexity_batch(sentetik_cumleler)

    print(f"\n  Ortalama perplexity: {avg_perplexity:.2f}")
    print(f"  Durum: {'✓ MUKEMMEL' if avg_perplexity <= 50 else '✓ IYI' if avg_perplexity <= 100 else '✗ ZAYIF (anlamsal olarak zayif cumleler)'}")

    # ===== GENEL DEĞERLENDİRME =====
    print("\n" + "="*60)
    print("GENEL DEGERLENDIRME")
    print("="*60)

    basari_sayisi = sum([
        tekil_oran >= 95,
        f1_mean >= 0.70,
        kapsama_orani >= 80,
        0.50 <= avg_similarity <= 0.75,
        avg_perplexity <= 100
    ])

    print(f"\nBASARI ORANI: {basari_sayisi}/5 metrik gecildi")
    if basari_sayisi >= 4:
        print("SONUC: ✓ MUKEMMEL - Yuksek kalite sentetik veri")
    elif basari_sayisi >= 3:
        print("SONUC: ✓ IYI - Kabul edilebilir kalite")
    else:
        print("SONUC: ✗ ZAYIF - Iyilestirme gerekli")

    # Detaylı istatistikler
    print("\n" + "="*60)
    print("[INFO] Dagilim analizleri...")
    print("\nKelime Sayisi Dagilimi:")
    orijinal_word_counts = [len(s.split()) for s in orijinal_cumleler]
    sentetik_word_counts = [len(s.split()) for s in sentetik_cumleler]
    print(f"  Orijinal: Ort={np.mean(orijinal_word_counts):.1f}, Min={min(orijinal_word_counts)}, Max={max(orijinal_word_counts)}")
    print(f"  Sentetik: Ort={np.mean(sentetik_word_counts):.1f}, Min={min(sentetik_word_counts)}, Max={max(sentetik_word_counts)}")

    print("\nBenzerlik Dagilimi:")
    sim_values = similarities.max(axis=1)
    print(f"  Min: {sim_values.min():.4f}")
    print(f"  Median: {np.median(sim_values):.4f}")
    print(f"  Max: {sim_values.max():.4f}")


# ============================================================
# ANA PROGRAM
# ============================================================

if __name__ == "__main__":
    # 1. Veri yükle
    orijinal_cumleler = veri_yukle(CSV_INPUT)

    # 2. 3000 sentetik cümle üret
    sentetik_cumleler = uret_3000_cumle(orijinal_cumleler, target=TARGET_SENTENCES)

    # 3. CSV'ye kaydet
    csv_kaydet(sentetik_cumleler, CSV_OUTPUT)

    # 4. Metrik analizi
    metrik_analizi(orijinal_cumleler, sentetik_cumleler)

    print("\n" + "="*60)
    print("PROGRAM TAMAMLANDI")
    print("="*60)
    print(f"Cikti dosyasi: {CSV_OUTPUT}")
    print("="*60)
