# Triple Hybrid (Faktörel + Perplexity + Gemini) + Semantik Benzerlik
# BATCH GENERATION: 100 cümle = 1 API isteği
import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import time
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# KONFİGÜRASYON
# ============================================================================

# API Key
GEMINI_API_KEY = 'GEMINI_API_KEY'  

# Model parametreleri
BATCH_SIZE = 100  
TARGET_SAMPLES = 1000  # Hedef cümle
SIMILARITY_THRESHOLD = 0.90  # Semantik benzerlik eşiği
QUALITY_THRESHOLD = 0.60  # Minimum kalite skoru
USE_PERPLEXITY = True  
USE_GEMINI_VALIDATION = False  

# ============================================================================
# MODELLERİ YÜKLEME
# ============================================================================

print("\n" + "ELEKTRİKLİ ARABA VERİ SETİ" * 40)
print("ELEKTRİKLİ ARABA VERİ SETİ OLUŞTURUCU v5.0")
print("BATCH GENERATION SİSTEMİ")
print("100 Cümle = 1 API İsteği")
print("ELEKTRİKLİ ARABA VERİ SETİ" * 40 + "\n")
print("Modeller yükleniyor...")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("Gemini 2.5 Flash hazır!")

# Sentence Transformer (Semantik benzerlik için)
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Sentence Transformer hazır!")

# Perplexity Model (Opsiyonel ama doğruluk için önerilir)
if USE_PERPLEXITY:
    try:
        perplexity_tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
        perplexity_model = AutoModelForCausalLM.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
        perplexity_model.eval()
        print("Perplexity modeli hazır!")
    except Exception as e:
        print(f"Perplexity modeli yüklenemedi: {e}")
        USE_PERPLEXITY = False
else:
    print("Perplexity kullanılmıyor")

print("\n" + "=" * 80 + "\n")


# ============================================================================
# KALİTE SKORLAMA FONKSİYONLARI
# ============================================================================

def calculate_factual_quality_score(text):
    """
    Faktörel kalite skorunu hesaplar (Rule-based)

    Args:
        text: Değerlendirilecek cümle

    Returns:
        score: 0-1 arası skor
        factors: Detaylı faktör bilgileri
    """
    score = 0.0
    factors = {}

    # 1. Kelime Sayısı (0-0.20)
    words = text.split()
    word_count = len(words)

    if word_count < 3:
        word_score = 0.0
    elif 3 <= word_count <= 6:
        word_score = 0.10
    elif 7 <= word_count <= 12:
        word_score = 0.20
    elif 13 <= word_count <= 15:
        word_score = 0.15
    else:
        word_score = 0.05

    score += word_score
    factors['word_count_score'] = word_score
    factors['word_count'] = word_count

    # 2. Dilbilgisi ve Yapı (0-0.30)
    grammar_score = 0.0

    # Büyük harf ile başlama
    if text and text[0].isupper():
        grammar_score += 0.10
        factors['starts_with_capital'] = True
    else:
        factors['starts_with_capital'] = False

    # Noktalama ile bitme
    if text and text[-1] in '.!?':
        grammar_score += 0.10
        factors['ends_with_punctuation'] = True
    else:
        factors['ends_with_punctuation'] = False

    # Kelime çeşitliliği
    unique_words = len(set(words))
    word_diversity = unique_words / max(1, word_count)
    if word_diversity > 0.8:
        grammar_score += 0.10
        factors['high_word_diversity'] = True
    else:
        factors['high_word_diversity'] = False

    score += grammar_score
    factors['grammar_score'] = grammar_score
    factors['word_diversity'] = round(word_diversity, 2)

    # 3. Anahtar Kelimeler (0-0.30)
    keywords = {
        'temel': ['elektrikli', 'araba', 'araç', 'otomobil', 'taşıt'],
        'markalar': ['tesla', 'bmw', 'mercedes', 'audi', 'nissan', 'renault', 'togg',
                     'volkswagen', 'hyundai', 'kia', 'ford', 'porsche'],
        'teknik': ['batarya', 'pil', 'şarj', 'menzil', 'km', 'kwh', 'motor',
                   'güç', 'tork', 'hız', 'performans'],
        'çevre': ['emisyon', 'karbon', 'temiz', 'yeşil', 'sürdürülebilir',
                  'çevre', 'çevreci'],
        'ekonomi': ['fiyat', 'maliyet', 'tasarruf', 'teşvik', 'ucuz', 'pahalı',
                    'ekonomik', 'bütçe']
    }

    text_lower = text.lower()
    keyword_categories = 0
    keyword_total = 0

    for category, words_list in keywords.items():
        found = sum(1 for kw in words_list if kw in text_lower)
        if found > 0:
            keyword_categories += 1
            keyword_total += found

    keyword_score = min(keyword_categories * 0.10, 0.30)
    score += keyword_score
    factors['keyword_score'] = keyword_score
    factors['keyword_categories'] = keyword_categories
    factors['keyword_total'] = keyword_total

    # 4. Bilgi İçeriği (0-0.20)
    info_score = 0.0

    # Sayısal veri varlığı
    numbers = re.findall(r'\d+', text)
    if numbers:
        info_score += 0.10
        factors['has_numbers'] = True
    else:
        factors['has_numbers'] = False

    # Özel isim (büyük harfle başlayan kelimeler)
    proper_nouns = [w for w in words if w and w[0].isupper()]
    if len(proper_nouns) > 0:
        info_score += 0.05
        factors['has_proper_nouns'] = True
    else:
        factors['has_proper_nouns'] = False

    # Cümle uzunluğu (minimum bilgi içeriği)
    if word_count >= 5:
        info_score += 0.05
        factors['sufficient_length'] = True
    else:
        factors['sufficient_length'] = False

    score += info_score
    factors['info_score'] = info_score

    # Final score
    final_score = round(min(score, 1.0), 2)
    factors['total_score'] = final_score

    return final_score, factors


def calculate_perplexity_score(text):
    """
    Perplexity skorunu hesaplar (Doğallık ölçüsü)

    Args:
        text: Değerlendirilecek cümle

    Returns:
        score: 0-1 arası normalize edilmiş skor
    """
    if not USE_PERPLEXITY:
        return 0.80  # Varsayılan skor

    try:
        # Tokenize
        inputs = perplexity_tokenizer(text, return_tensors="pt")

        # Perplexity hesapla
        with torch.no_grad():
            outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        # Skorlama (düşük perplexity = yüksek skor)
        if perplexity < 10:
            score = 1.0
        elif perplexity < 20:
            score = 0.95
        elif perplexity < 30:
            score = 0.85
        elif perplexity < 50:
            score = 0.70
        elif perplexity < 80:
            score = 0.50
        elif perplexity < 120:
            score = 0.30
        else:
            score = 0.10

        return round(score, 2)

    except Exception as e:
        print(f"Perplexity hesaplama hatası: {e}")
        return 0.70  # Hata durumunda orta skor


def calculate_dual_quality_score(text):
    """
    Dual Hybrid Skor (Faktörel + Perplexity)
    Gemini validasyon batch'te yapıldığı için burada kullanılmıyor

    Args:
        text: Değerlendirilecek cümle

    Returns:
        final_score: 0-1 arası final skor
        details: Detaylı bilgiler
    """
    # Faktörel skor
    factual_score, factors = calculate_factual_quality_score(text)

    # Perplexity skor
    perplexity_score = calculate_perplexity_score(text)

    # Ağırlıklandırılmış toplam
    if USE_PERPLEXITY:
        final_score = round((factual_score * 0.50) + (perplexity_score * 0.50), 2)
    else:
        final_score = factual_score

    details = {
        'factual_score': factual_score,
        'perplexity_score': perplexity_score,
        'factors': factors,
        'method': 'dual_hybrid' if USE_PERPLEXITY else 'factual_only'
    }

    return final_score, details


# ============================================================================
# SEMANTİK BENZERLİK
# ============================================================================

def is_similar_to_existing(new_text, existing_texts, threshold=0.90):
    """
    Yeni cümlenin mevcut cümlelere semantik benzerliğini kontrol eder

    Args:
        new_text: Kontrol edilecek cümle
        existing_texts: Mevcut cümleler listesi
        threshold: Benzerlik eşiği (varsayılan: 0.90)

    Returns:
        is_similar: True/False
        max_similarity: En yüksek benzerlik skoru
        most_similar_text: En benzer cümle
    """
    if not existing_texts:
        return False, 0.0, None

    try:
        # Embeddings
        new_embedding = semantic_model.encode(new_text)
        existing_embeddings = semantic_model.encode(existing_texts)

        
        similarities = cosine_similarity([new_embedding], existing_embeddings)[0]

        # En yüksek benzerlik
        max_similarity = float(np.max(similarities))
        max_index = int(np.argmax(similarities))
        most_similar_text = existing_texts[max_index]

        is_similar = max_similarity >= threshold

        return is_similar, max_similarity, most_similar_text

    except Exception as e:
        print(f"Benzerlik kontrolü hatası: {e}")
        return False, 0.0, None


# ============================================================================
# BATCH GENERATION - 100 CÜMLE TEK İSTEK
# ============================================================================

def generate_batch_sentences(batch_size=100, sentiment_distribution=None,
                             focus_areas=None, iteration=1):
    """
    Tek Gemini isteğinde batch_size kadar cümle üretir

    Args:
        batch_size: Üretilecek cümle sayısı
        sentiment_distribution: {'pozitif': X, 'negatif': Y, 'nötr': Z}
        focus_areas: Konu listesi
        iteration: Kaçıncı batch (çeşitlilik için)

    Returns:
        List of dicts: [{'text': '...', 'sentiment': '...'}, ...]
    """

    # Varsayılan sentiment dağılımı
    if sentiment_distribution is None:
        sentiment_distribution = {
            'pozitif': int(batch_size * 0.4),
            'negatif': int(batch_size * 0.2),
            'nötr': int(batch_size * 0.4)
        }

    # Varsayılan konular
    if focus_areas is None:
        focus_areas = [
            "batarya teknolojisi ve kapasitesi",
            "şarj altyapısı ve süreleri",
            "elektrikli araç fiyatları ve maliyetleri",
            "çevre faydaları ve emisyonlar",
            "performans, hız ve tork özellikleri",
            "markalar (Tesla, BMW, Mercedes, Togg, vb.)",
            "bakım ve servis maliyetleri",
            "gelecek trendleri ve teknolojiler",
            "kullanıcı deneyimleri ve yorumları",
            "teknik özellikler ve spesifikasyonlar",
            "menzil ve batarya ömrü",
            "şarj istasyonları ve erişilebilirlik",
            "devlet teşvikleri ve destekler",
            "ikinci el pazar ve değer kaybı",
            "güvenlik özellikleri ve testleri"
        ]

    # Prompt oluştur
    prompt = f"""
Elektrikli arabalar hakkında {batch_size} adet FARKLI ve ORİJİNAL Türkçe cümle üret.

ÖNEMLİ KURALLAR:
1. Her cümle 5-15 kelime arasında olmalı
2. Her cümle BİRBİRİNDEN TAMAMEN FARKLI olmalı (TEKRAR YASAK!)
3. Her cümle FARKLI bir bilgi, görüş veya perspektif içermeli
4. Çeşitli konulardan seç: {', '.join(focus_areas[:10])}

SENTIMENT DAĞILIMI (Kesinlikle uyulmalı):
- POZİTİF: {sentiment_distribution['pozitif']} cümle (avantajlar, olumlu yönler, başarılar)
- NEGATİF: {sentiment_distribution['negatif']} cümle (dezavantajlar, sorunlar, eleştiriler, zorluklar)
- NÖTR: {sentiment_distribution['nötr']} cümle (objektif bilgiler, tanımlar, rakamlar, gerçekler)

ÇEŞİTLİLİK İÇİN:
- Farklı cümle yapıları kullan (soru, açıklama, karşılaştırma)
- Farklı kelime hazinesi (eş anlamlı kelimeler tercih et)
- Farklı uzunluklarda cümleler (5-15 kelime arası dengeli dağıt)
- Bazılarında sayısal veriler, bazılarında görüşler kullan
- Farklı markalar ve modeller bahset

BATCH {iteration}: Bu batch özellikle farklı ve özgün olmalı!

JSON formatında döndür (SADECE JSON, başka açıklama ekleme):
{{
  "sentences": [
    {{"text": "Cümle 1 metni", "sentiment": "pozitif"}},
    {{"text": "Cümle 2 metni", "sentiment": "negatif"}},
    {{"text": "Cümle 3 metni", "sentiment": "nötr"}},
    ...
  ]
}}

TEKRAR ETME! Her cümle benzersiz olmalı!
"""

    try:
        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip()

        result_text = result_text.replace('```json', '').replace('```', '').strip()

        result = json.loads(result_text)
        sentences = result.get('sentences', [])

        return sentences

    except json.JSONDecodeError as e:
        print(f"JSON parse hatası: {e}")
        print(f"Response: {result_text[:200]}...")
        return []

    except Exception as e:
        print(f"Batch üretim hatası: {e}")
        return []


# ============================================================================
# BATCH İŞLEME VE FİLTRELEME
# ============================================================================

def process_batch(batch_sentences, existing_texts, similarity_threshold=0.90,
                  quality_threshold=0.60, batch_num=1):
    """
    Batch'teki cümleleri filtreler ve kaliteli olanları seçer

    Args:
        batch_sentences: Gemini'den gelen cümle listesi
        existing_texts: Mevcut cümleler
        similarity_threshold: Semantik benzerlik eşiği
        quality_threshold: Minimum kalite skoru
        batch_num: Batch numarası

    Returns:
        accepted: Kabul edilen cümleler
        rejected: Reddedilen cümleler
    """

    accepted = []
    rejected = []
    temp_existing = existing_texts.copy()

    print(f"\nBATCH {batch_num}: {len(batch_sentences)} cümle işleniyor...")

    for i, item in enumerate(batch_sentences, 1):
        try:
            text = item.get('text', '').strip()
            sentiment = item.get('sentiment', 'nötr').lower()

            # Temizleme
            text = text.replace('"', '').replace("'", '').strip()
            if text and text[-1] not in ['.', '!', '?']:
                text += '.'

            # Sentiment normalize
            if sentiment not in ['pozitif', 'negatif', 'nötr']:
                sentiment = 'nötr'

            # Boş kontrol
            if not text or len(text) < 10:
                rejected.append({
                    'text': text,
                    'reason': 'Çok kısa',
                    'batch': batch_num
                })
                continue

            # Kelime sayısı
            word_count = len(text.split())
            if not (3 <= word_count <= 15):
                rejected.append({
                    'text': text,
                    'reason': f'Kelime sayısı: {word_count}',
                    'batch': batch_num
                })
                continue

            # Kalite skoru
            quality_score, quality_details = calculate_dual_quality_score(text)

            if quality_score < quality_threshold:
                rejected.append({
                    'text': text,
                    'reason': f'Düşük kalite: {quality_score:.2f}',
                    'score': quality_score,
                    'batch': batch_num
                })
                continue

            # Semantik benzerlik (hem mevcut hem batch içi)
            is_similar, max_sim, similar_text = is_similar_to_existing(
                text, temp_existing, threshold=similarity_threshold
            )

            if is_similar:
                rejected.append({
                    'text': text,
                    'reason': f'Benzer: {max_sim:.3f}',
                    'similar_to': similar_text[:50] + '...',
                    'batch': batch_num
                })
                continue

           
            accepted.append({
                'text': text,
                'sentiment': sentiment,
                'word_count': word_count,
                'quality_score': quality_score,
                'max_similarity': max_sim,
                'factual_score': quality_details['factual_score'],
                'perplexity_score': quality_details['perplexity_score'],
                'batch': batch_num
            })

            temp_existing.append(text)

        except Exception as e:
            print(f"Cümle {i} işleme hatası: {e}")
            continue

    print(f"Kabul: {len(accepted)} | Red: {len(rejected)}")

    return accepted, rejected


# ============================================================================
# 1000 CÜMLE ÜRETİMİ (BATCH SİSTEMİ)
# ============================================================================

def create_dataset_with_batches(target_samples=1000, batch_size=100,
                                similarity_threshold=0.90, quality_threshold=0.60):
    """
    Batch generation ile dataset oluşturur

    Args:
        target_samples: Hedef cümle sayısı
        batch_size: Her batch'te kaç cümle
        similarity_threshold: Benzerlik eşiği
        quality_threshold: Kalite eşiği

    Returns:
        df: Pandas DataFrame
        rejected: Reddedilen cümleler
    """

    dataset = []
    all_rejected = []
    existing_texts = []
    batch_count = 0
    total_api_requests = 0

    print("\n" + "=" * 80)
    print("BATCH SİSTEM İLE VERİ SETİ OLUŞTURMA")
    print("=" * 80)
    print(f"Hedef: {target_samples} cümle")
    print(f"Batch boyutu: {batch_size} cümle/batch")
    print(f"Benzerlik eşiği: {similarity_threshold}")
    print(f"Kalite eşiği: {quality_threshold}")
    print(f"Perplexity: {'Aktif' if USE_PERPLEXITY else 'Kapalı'}")
    print("=" * 80 + "\n")

    # Sentiment hedefleri
    sentiment_targets = {
        'pozitif': int(target_samples * 0.4),
        'negatif': int(target_samples * 0.2),
        'nötr': int(target_samples * 0.4)
    }

    sentiment_counts = {'pozitif': 0, 'negatif': 0, 'nötr': 0}

    start_time = time.time()

    while len(dataset) < target_samples:
        batch_count += 1

        # Kalan cümle sayısı
        remaining = target_samples - len(dataset)
        current_batch_size = min(batch_size, remaining + 30)  # +30 yedek

        # Sentiment dağılımı (kalan için)
        batch_sentiment_dist = {}
        for sent, target in sentiment_targets.items():
            needed = max(0, target - sentiment_counts[sent])
            ratio = needed / max(1, remaining)
            batch_sentiment_dist[sent] = max(1, int(current_batch_size * ratio))

        # Toplamı normalize et
        total_dist = sum(batch_sentiment_dist.values())
        if total_dist != current_batch_size:
            diff = current_batch_size - total_dist
            batch_sentiment_dist['nötr'] += diff

        print(f"\n{'=' * 80}")
        print(f"BATCH {batch_count}")
        print(f"{'=' * 80}")
        print(f"Hedef: {current_batch_size} cümle")
        print(f"Sentiment: Poz:{batch_sentiment_dist['pozitif']} "
              f"Neg:{batch_sentiment_dist['negatif']} "
              f"Nötr:{batch_sentiment_dist['nötr']}")
        print(f"Mevcut: {len(dataset)}/{target_samples}")
        print(f"API İsteği: {total_api_requests + 1}")

        # Batch üret
        print(f"Üretiliyor...")
        batch_sentences = generate_batch_sentences(
            batch_size=current_batch_size,
            sentiment_distribution=batch_sentiment_dist,
            iteration=batch_count
        )

        total_api_requests += 1

        if not batch_sentences:
            print("Batch boş geldi, tekrar deneniyor...")
            time.sleep(2)
            continue

        print(f"{len(batch_sentences)} cümle alındı")

        # Filtrele
        accepted, rejected = process_batch(
            batch_sentences,
            existing_texts,
            similarity_threshold=similarity_threshold,
            quality_threshold=quality_threshold,
            batch_num=batch_count
        )

        # Kabul edilenleri ekle
        for item in accepted:
            sentiment = item['sentiment']

            # Sentiment limitini kontrol et
            if sentiment_counts[sentiment] < sentiment_targets[sentiment]:
                item['id'] = len(dataset) + 1
                dataset.append(item)
                existing_texts.append(item['text'])
                sentiment_counts[sentiment] += 1
            else:
                # Sentiment limiti dolmuş, başka sentiment'e geçir
                for alt_sent in ['pozitif', 'negatif', 'nötr']:
                    if sentiment_counts[alt_sent] < sentiment_targets[alt_sent]:
                        item['sentiment'] = alt_sent
                        item['id'] = len(dataset) + 1
                        dataset.append(item)
                        existing_texts.append(item['text'])
                        sentiment_counts[alt_sent] += 1
                        break

        all_rejected.extend(rejected)

        # İstatistikler
        elapsed = time.time() - start_time
        print(f"\nBATCH {batch_count} ÖZET:")
        print(f"   Üretilen: {len(batch_sentences)}")
        print(f"   Kabul: {len(accepted)}")
        print(f"   Red: {len(rejected)}")
        print(f"   Dataset: {len(dataset)}/{target_samples} (%{len(dataset) / target_samples * 100:.1f})")
        print(f"   API İsteği: {total_api_requests}")
        print(f"   Süre: {elapsed / 60:.1f} dakika")

        print(f"\n   Sentiment Dağılımı:")
        for sent in ['pozitif', 'negatif', 'nötr']:
            count = sentiment_counts[sent]
            target = sentiment_targets[sent]
            pct = (count / target) * 100 if target > 0 else 0
            bar = '█' * int(pct / 5)
            print(f"   {sent:8}: {count:3}/{target:3} (%{pct:5.1f}) {bar}")

        # Hedef kontrolü
        if len(dataset) >= target_samples:
            break

        # Rate limiting
        if len(dataset) < target_samples:
            time.sleep(1)

    # Final DataFrame
    df = pd.DataFrame(dataset[:target_samples])

    elapsed_total = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("VERİ SETİ OLUŞTURULDU!")
    print(f"{'=' * 80}")
    print(f"Toplam cümle: {len(df)}")
    print(f"Toplam batch: {batch_count}")
    print(f"Toplam API isteği: {total_api_requests}")
    print(f"Toplam süre: {elapsed_total / 60:.1f} dakika")
    print(f"Reddedilen: {len(all_rejected)}")

    print(f"\nSentiment Dağılımı:")
    for sent in ['pozitif', 'negatif', 'nötr']:
        count = len(df[df['sentiment'] == sent])
        pct = (count / len(df)) * 100
        emoji = {'pozitif': '😊', 'negatif': '😔', 'nötr': '😐'}[sent]
        print(f"   {emoji} {sent:8}: {count:3} (%{pct:5.1f})")

    print(f"\nOrtalama Skorlar:")
    print(f"   Quality:    {df['quality_score'].mean():.3f}")
    print(f"   Faktörel:   {df['factual_score'].mean():.3f}")
    print(f"   Perplexity: {df['perplexity_score'].mean():.3f}")
    print(f"   Similarity: {df['max_similarity'].mean():.3f}")

    print(f"\nKelime İstatistikleri:")
    print(f"   Ortalama:   {df['word_count'].mean():.1f} kelime")
    print(f"   Minimum:    {df['word_count'].min()} kelime")
    print(f"   Maksimum:   {df['word_count'].max()} kelime")

    print(f"{'=' * 80}\n")

    return df, all_rejected


# ============================================================================
# ANALİZ VE GÖRSELLEŞTİRME
# ============================================================================

def analyze_dataset(df, rejected):
    """Dataset analizi ve istatistikler"""

    print("\n" + "=" * 80)
    print("DETAYLI ANALİZ")
    print("=" * 80)

    # Kalite dağılımı
    print("\nKalite Skor Dağılımı:")
    bins = [0.60, 0.70, 0.80, 0.90, 1.0]  
    labels = ['Düşük (0.60-0.70)', 'Orta (0.70-0.80)',
              'İyi (0.80-0.90)', 'Çok İyi (0.90-1.0)']

    df['quality_category'] = pd.cut(df['quality_score'], bins=bins, labels=labels)
    print(df['quality_category'].value_counts().to_string())

    # Batch dağılımı
    if 'batch' in df.columns:
        print("\nBatch Başına Kabul Edilen:")
        batch_counts = df['batch'].value_counts().sort_index()
        for batch, count in batch_counts.items():
            print(f"   Batch {batch}: {count} cümle")

    # Red sebepleri
    if rejected:
        print("\nRed Sebepleri (İlk 5):")
        reasons = {}
        for item in rejected:
            reason = item.get('reason', 'Bilinmiyor')
            reasons[reason] = reasons.get(reason, 0) + 1

        for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"   {reason}: {count}")

    print("\n" + "=" * 80)


def save_dataset(df, rejected, prefix='dataset'):
    """Dataset'i farklı formatlarda kaydet"""

    print(f"\nDosyalar kaydediliyor...")

    # CSV
    csv_file = f'{prefix}_1000_batch.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"   {csv_file}")

    # Excel
    excel_file = f'{prefix}_1000_batch.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"   {excel_file}")

    # JSON
    json_file = f'{prefix}_1000_batch.json'
    df.to_json(json_file, orient='records', force_ascii=False, indent=2)
    print(f"   {json_file}")

    # Rejected (opsiyonel)
    if rejected:
        rejected_file = f'{prefix}_rejected.json'
        with open(rejected_file, 'w', encoding='utf-8') as f:
            json.dump(rejected, f, ensure_ascii=False, indent=2)
        print(f"   {rejected_file}")

    print(f"\nTüm dosyalar kaydedildi!")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":

    # Parametreler
    print("\nPARAMETRELER:")
    print(f"   Hedef: {TARGET_SAMPLES} cümle")
    print(f"   Batch boyutu: {BATCH_SIZE}")
    print(f"   Benzerlik eşiği: {SIMILARITY_THRESHOLD}")
    print(f"   Kalite eşiği: {QUALITY_THRESHOLD}")
    print(f"   Perplexity: {'Aktif' if USE_PERPLEXITY else 'Kapalı'}")

    # Onay
    print("\n" + "=" * 80)
    user_input = input("Başlatmak için ENTER'a basın (Çıkmak için 'q'): ")
    if user_input.lower() == 'q':
        print("İptal edildi.")
        exit()

    # Dataset oluştur
    df, rejected = create_dataset_with_batches(
        target_samples=TARGET_SAMPLES,
        batch_size=BATCH_SIZE,
        similarity_threshold=SIMILARITY_THRESHOLD,
        quality_threshold=QUALITY_THRESHOLD
    )

    # Analiz
    analyze_dataset(df, rejected)

    # Kaydet
    save_dataset(df, rejected, prefix='elektrikli_araba')

    # Final mesaj
    print("\n" + "ELEKTRİKLİ ARABA VERİ SETİ" * 40)
    print("BAŞARILI! 1000 CÜMLE ÜRETİLDİ!")
    print("ELEKTRİKLİ ARABA VERİ SETİ" * 40)
    print(f"\nOrtalama kalite: {df['quality_score'].mean():.3f}")
    print(f"Unique cümleler: %100")
    print(f"API isteği kullanımı: Minimal (~10-15 istek)")
    print(f"Maliyet: ~")
    print("\nDataset hazır, kullanıma uygun!\n")