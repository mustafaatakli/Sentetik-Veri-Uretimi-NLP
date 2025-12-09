import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import io
import random
import re
from tqdm import tqdm
from collections import Counter
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

print("="*70)
print("LSTM ILE SENTETIK VERI URETIMI")
print("="*70)

# ========== GPU AYARLARI ==========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n[OK] GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("\n[UYARI] GPU bulunamadi, CPU kullaniliyor")

# ========== BERT MODELI (METRIKLER ICIN) ==========
print("\n[INFO] BERT modeli yukleniyor (metrik hesaplamalari icin)...")
model_name = "dbmdz/bert-base-turkish-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained(model_name)
bert_model = bert_model.to(device)
print(f"[OK] BERT modeli {device} uzerinde hazir!\n")


# ========== KARAKTER TOKENIZER ==========
class CharTokenizer:
    """Karakter seviyesi tokenizer"""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def fit(self, texts):
        """Karakterleri öğren"""
        chars = set()
        for text in texts:
            chars.update(text)

        # Özel tokenlar
        self.char2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}

        # Karakterleri ekle
        for idx, char in enumerate(sorted(chars), start=4):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

        self.vocab_size = len(self.char2idx)
        print(f"[INFO] Vocabulary size: {self.vocab_size} characters")

    def encode(self, text, max_len=None):
        """Metni token ID'lere çevir"""
        tokens = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]

        if max_len:
            if len(tokens) < max_len:
                tokens = tokens + [self.char2idx['<PAD>']] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]

        return tokens

    def decode(self, tokens):
        """Token ID'leri metne çevir"""
        chars = []
        for token in tokens:
            if token in [self.char2idx['<PAD>'], self.char2idx['<START>'], self.char2idx['<END>']]:
                continue
            chars.append(self.idx2char.get(token, '<UNK>'))
        return ''.join(chars)


# ========== LSTM MODELI ==========
class LSTMGenerator(nn.Module):
    """LSTM tabanlı metin üretici"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(LSTMGenerator, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM katmanları
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Çıkış katmanı
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """İleri geçiş"""
        # Embedding
        embedded = self.embedding(x)

        # LSTM
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)

        # Dropout ve çıkış
        output = self.dropout(output)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        """Hidden state başlat"""
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        )


# ========== DATASET ==========
class TextDataset(Dataset):
    """Metin dataset"""
    def __init__(self, texts, tokenizer, max_len=100):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Start ve End token ekle
        text = '<START>' + text + '<END>'

        # Encode
        tokens = self.tokenizer.encode(text, max_len=self.max_len)

        # Input ve target (next character prediction)
        input_seq = tokens[:-1]
        target_seq = tokens[1:]

        return torch.tensor(input_seq), torch.tensor(target_seq)


# ========== LSTM EGITIM ==========
def train_lstm(model, train_loader, epochs=50, lr=0.001, device='cuda'):
    """LSTM modelini eğit"""
    print(f"\n{'='*70}")
    print(f"[BASLIYOR] LSTM MODELI EGITIMI")
    print(f"[EPOCHS] {epochs}")
    print(f"[LEARNING RATE] {lr}")
    print(f"[DEVICE] {device}")
    print(f"{'='*70}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD token'ı ignore et
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            optimizer.zero_grad()
            outputs, _ = model(inputs)

            # Loss hesapla
            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    print(f"\n[OK] Egitim tamamlandi!\n")
    return model


# ========== LSTM ILE CUMLE URETME ==========
def generate_text_lstm(model, tokenizer, seed_text, max_len=100, temperature=0.8, device='cuda'):
    """LSTM ile metin üret"""
    model.eval()

    # Seed text'i encode et
    text = '<START>' + seed_text
    tokens = tokenizer.encode(text)
    input_seq = torch.tensor([tokens]).to(device)

    generated = seed_text
    hidden = None

    with torch.no_grad():
        for _ in range(max_len - len(seed_text)):
            # Tahmin yap
            output, hidden = model(input_seq, hidden)

            # Son karakter tahmini
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)

            # Sampling
            next_token = torch.multinomial(probs, 1).item()

            # End token'da dur
            if next_token == tokenizer.char2idx['<END>']:
                break

            # Karakteri ekle
            next_char = tokenizer.idx2char[next_token]
            if next_char == '<UNK>':
                break

            generated += next_char

            # Sonraki input
            input_seq = torch.tensor([[next_token]]).to(device)

    return generated


def generate_variants_lstm(model, tokenizer, seed_text, n_variants=30,
                           orijinal_cumleler=None, temperature=0.9, device='cuda'):
    """Bir cümleden varyantlar üret"""
    if orijinal_cumleler:
        orijinal_normalize_set = {cumleleri_normalize(c) for c in orijinal_cumleler}
    else:
        orijinal_normalize_set = set()

    varyantlar = set()
    max_attempts = n_variants * 10

    for attempt in range(max_attempts):
        # Farklı başlangıç noktaları dene
        if len(seed_text) > 10:
            # Rastgele bir prefix kullan
            prefix_len = random.randint(5, min(15, len(seed_text) - 5))
            prefix = seed_text[:prefix_len]
        else:
            prefix = seed_text[:3] if len(seed_text) >= 3 else seed_text

        # Temperature varyasyonu
        temp = temperature + random.uniform(-0.2, 0.2)
        temp = max(0.5, min(1.5, temp))

        # Üret
        generated = generate_text_lstm(model, tokenizer, prefix,
                                       max_len=150, temperature=temp, device=device)

        # Temizlik
        generated = generated.strip()

        # Geçerlilik kontrolleri
        if len(generated) < 10 or len(generated) > 200:
            continue

        kelime_sayisi = len(generated.split())
        if kelime_sayisi < 3 or kelime_sayisi > 20:
            continue

        # Normalize et ve kontrol et
        generated_normalized = cumleleri_normalize(generated)
        seed_normalized = cumleleri_normalize(seed_text)

        if generated_normalized == seed_normalized:
            continue

        if generated_normalized in orijinal_normalize_set:
            continue

        # Farklılık kontrolü
        gen_words = set(generated_normalized.split())
        seed_words = set(seed_normalized.split())
        diff_words = len(gen_words.symmetric_difference(seed_words))

        if diff_words >= 2:
            varyantlar.add(generated)

        if len(varyantlar) >= n_variants:
            break

    return list(varyantlar)[:n_variants]


# ========== YARDIMCI FONKSIYONLAR ==========
def cumleleri_normalize(cumle):
    """Cümleyi normalize et"""
    import string
    cumle = cumle.lower()
    cumle = cumle.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(cumle.split())


def hesapla_perplexity(cumle, model, tokenizer, device):
    """Perplexity hesapla"""
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


def filtrele_perplexity(cumleler, model, tokenizer, device, esik=50.0):
    """Perplexity ile filtrele"""
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
def uret_3000_cumle_lstm(lstm_model, tokenizer, seed_cumleler, hedef_sayi=3000,
                         perplexity_esik=50.0, device='cuda'):
    """LSTM ile 100 cümleden 3000 cümle üret"""
    print(f"\n{'='*70}")
    print(f"[BASLIYOR] LSTM ILE SENTETIK VERI URETIMI")
    print(f"[BASLANGIC] {len(seed_cumleler)} cumle")
    print(f"[HEDEF] {hedef_sayi} cumle")
    print(f"[MODEL] Character-level LSTM")
    print(f"[PERPLEXITY ESIK] {perplexity_esik}")
    print(f"{'='*70}\n")

    tum_sentetik = []
    her_cumleden = hedef_sayi // len(seed_cumleler)

    print(f"[AYAR] Her cumleden ~{her_cumleden} varyant uretiliyor...\n")

    orijinal_cumle_set = set(seed_cumleler)

    # Her seed için üret
    for idx, cumle in enumerate(tqdm(seed_cumleler, desc="LSTM ile uretiliyor"), 1):
        varyantlar = generate_variants_lstm(
            lstm_model,
            tokenizer,
            cumle,
            n_variants=her_cumleden,
            orijinal_cumleler=seed_cumleler,
            temperature=0.9,
            device=device
        )

        for varyant in varyantlar:
            if varyant not in orijinal_cumle_set:
                tum_sentetik.append(varyant)

        if idx % 10 == 0:
            print(f"\n[PROGRESS] {idx}/{len(seed_cumleler)} cumle islendi, {len(tum_sentetik)} varyant uretildi")

    print(f"\n[INFO] Ilk asamada {len(tum_sentetik)} cumle uretildi")

    # Tekrar temizleme
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
            ek_varyantlar = generate_variants_lstm(
                lstm_model,
                tokenizer,
                rastgele_cumle,
                n_variants=20,
                orijinal_cumleler=seed_cumleler,
                temperature=1.1,
                device=device
            )

            for varyant in ek_varyantlar:
                if varyant not in orijinal_cumle_set and varyant not in tum_sentetik:
                    tum_sentetik.append(varyant)

            ek_uretim_count += 1

            if len(tum_sentetik) >= hedef_sayi:
                break

    final_veri = tum_sentetik[:hedef_sayi]

    # PERPLEXITY HESAPLAMA (filtreleme YOK - LSTM için tam 3000 cümle garanti)
    print(f"\n{'='*70}")
    print(f"[ADIM 2] PERPLEXITY HESAPLAMA (filtreleme devre disi)")
    print(f"{'='*70}")
    print(f"[INFO] LSTM karakteristik olarak dusuk kaliteli oldugu icin,")
    print(f"[INFO] adil karsilastirma icin filtreleme yapilmiyor.")
    print(f"[INFO] Perplexity sadece raporlama amacli hesaplanacak.\n")

    # Sadece perplexity hesapla, filtreleme yapma
    perplexity_skorlari = []
    for cumle in tqdm(final_veri, desc="Perplexity hesaplaniyor"):
        ppl = hesapla_perplexity(cumle, bert_model, bert_tokenizer, device)
        perplexity_skorlari.append(ppl)

    print(f"\n[SONUC] Perplexity Istatistikleri:")
    print(f"  - Ortalama perplexity: {np.mean(perplexity_skorlari):.2f}")
    print(f"  - Min/Max: {np.min(perplexity_skorlari):.2f} / {np.max(perplexity_skorlari):.2f}")
    print(f"  - Median: {np.median(perplexity_skorlari):.2f}")

    print(f"\n{'='*70}")
    print(f"[OK] TAMAMLANDI!")
    print(f"[SONUC] Toplam uretilen: {len(final_veri)} cumle")
    print(f"{'='*70}\n")

    return final_veri, perplexity_skorlari


# ========== METRIK ANALIZI (AYNI BERT/GEMINI ILE) ==========
def metrik_analizi(orijinal_cumleler, sentetik_cumleler, perplexity_skorlari=None):
    """5 temel metrik + görselleştirme"""
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


# ========== ANA PROGRAM ==========
if __name__ == "__main__":

    # VERİ YÜKLE
    print("\n[INFO] Veri yukleniyor...")
    try:
        with open('tekonoloji-haber-baslıkları.csv', 'r', encoding='utf-8') as f:
            tum_satirlar = f.readlines()

        seed_cumleler = [line.strip() for line in tum_satirlar if line.strip()][:100]
        print(f"[OK] {len(seed_cumleler)} cumle yuklendi!")

        print("\n[ORNEK] Ilk 5 haber basligi:")
        for i, cumle in enumerate(seed_cumleler[:5], 1):
            print(f"  {i}. {cumle}")

    except FileNotFoundError:
        print("[HATA] 'tekonoloji-haber-baslıkları.csv' dosyasi bulunamadi!")
        sys.exit(1)


    # TOKENIZER OLUSTUR VE EGIT
    print("\n" + "="*70)
    print("[ADIM 1] TOKENIZER HAZIRLANIYOR")
    print("="*70)

    char_tokenizer = CharTokenizer()
    char_tokenizer.fit(seed_cumleler)


    # LSTM MODELI OLUSTUR
    print("\n" + "="*70)
    print("[ADIM 2] LSTM MODELI OLUSTURULUYOR")
    print("="*70)

    lstm_model = LSTMGenerator(
        vocab_size=char_tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(f"[OK] Model parametreleri: {sum(p.numel() for p in lstm_model.parameters())/1e6:.2f}M")


    # DATASET VE DATALOADER
    print("\n[INFO] Dataset hazirlaniyor...")
    train_dataset = TextDataset(seed_cumleler, char_tokenizer, max_len=150)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"[OK] {len(train_dataset)} ornek, batch_size=32")


    # EGITIM
    print("\n" + "="*70)
    input("[UYARI] LSTM egitimi baslasin mi? (Enter'a basin)")

    lstm_model = train_lstm(
        lstm_model,
        train_loader,
        epochs=50,
        lr=0.001,
        device=device
    )


    # SENTETIK VERI URET
    print("\n" + "="*70)
    input("[UYARI] Sentetik veri uretimi baslasin mi? (Enter'a basin)")

    sentetik_veri, perplexity_skorlari = uret_3000_cumle_lstm(
        lstm_model,
        char_tokenizer,
        seed_cumleler,
        hedef_sayi=3000,
        perplexity_esik=50.0,  # Kullanilmiyor (filtreleme devre disi)
        device=device
    )


    # KAYDET
    df_sonuc = pd.DataFrame({'haber_basligi': sentetik_veri})
    dosya_adi = 'lstm_sentetik_teknoloji_haberleri_3000.csv'
    df_sonuc.to_csv(dosya_adi, index=False, encoding='utf-8-sig')

    print(f"\n[KAYIT] Dosya kaydedildi: {dosya_adi}")

    print("\n[ORNEK] Ilk 10 sentetik haber basligi:")
    print("-" * 70)
    for i, cumle in enumerate(sentetik_veri[:10], 1):
        print(f"{i}. {cumle}")


    # METRIK ANALIZI
    metrik_analizi(seed_cumleler, sentetik_veri, perplexity_skorlari)


    print("\n" + "="*70)
    print("TAMAMLANDI!")
    print("="*70)
    print(f"Cikti dosyasi: {dosya_adi}")
    print("="*70)
