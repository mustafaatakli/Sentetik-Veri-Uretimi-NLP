DATASET_PATH = '/kaggle/input/sentences/sentences.txt'  
# Eğitim
NUM_EPOCHS = 200  
BATCH_SIZE = 32
HIDDEN_DIM = 512

# Üretim
TOTAL_TO_GENERATE = 10000  
GENERATION_BATCH_SIZE = 500
TOP_N = 50

# Filtreleme 
MIN_WORDS = 3
MAX_WORDS = 20  
MAX_SIMILARITY = 0.92  

# =====================================================
# KÜTÜPHANELER
# =====================================================

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import re
from typing import List, Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Kütüphaneler yüklendi!\n")

# =====================================================
# DATASET SINIFI
# =====================================================

class TurkishTextDataset(Dataset):
    def __init__(self, sentences: List[str], max_len: int = 20):
        self.sentences = sentences
        self.max_len = max_len
        self.word2idx, self.idx2word = self._build_vocab()
        self.vocab_size = len(self.word2idx)
        self.encoded_sentences = self._encode_sentences()
        
    def _clean_sentence(self, sentence: str) -> str:
        sentence = sentence.lower()
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence.strip()
    
    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        idx = 4
        for sentence in self.sentences:
            clean_sent = self._clean_sentence(sentence)
            for word in clean_sent.split():
                if word not in word2idx:
                    word2idx[word] = idx
                    idx += 1
        idx2word = {v: k for k, v in word2idx.items()}
        return word2idx, idx2word
    
    def _encode_sentences(self) -> List[torch.Tensor]:
        encoded = []
        for sentence in self.sentences:
            clean_sent = self._clean_sentence(sentence)
            words = clean_sent.split()
            if len(words) < 3 or len(words) > 15:
                continue
            indices = [self.word2idx['<SOS>']]
            for word in words[:self.max_len]:
                indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
            indices.append(self.word2idx['<EOS>'])
            while len(indices) < self.max_len + 2:
                indices.append(self.word2idx['<PAD>'])
            encoded.append(torch.tensor(indices[:self.max_len + 2], dtype=torch.long))
        return encoded
    
    def decode_sentence(self, indices: torch.Tensor) -> str:
        words = []
        for idx in indices:
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            word = self.idx2word.get(idx, '<UNK>')
            if word == '<EOS>':
                break
            if word not in ['<PAD>', '<SOS>', '<UNK>']:
                words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.encoded_sentences)
    
    def __getitem__(self, idx):
        return self.encoded_sentences[idx]

print("Dataset sınıfı hazır!")

# =====================================================
# GAN MODELLERİ
# =====================================================

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, latent_dim=100, max_seq_len=15, num_layers=3, dropout=0.3):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len  
        self.num_layers = num_layers
        
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * num_layers),
            nn.Tanh()
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, noise, seq_len=None):
        batch_size = noise.size(0)
        seq_len = seq_len or self.max_seq_len
        
        hidden = self.latent_to_hidden(noise)
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_dim)
        cell = torch.zeros_like(hidden)
        
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=noise.device)
        outputs = []
        samples = []
        
        for _ in range(seq_len):
            embedded = self.embedding(input_token)
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            output = self.fc(self.dropout(lstm_out.squeeze(1)))
            outputs.append(output)
            
            probs = F.gumbel_softmax(output, tau=1.0, hard=True)
            next_token = probs.argmax(dim=-1, keepdim=True)
            samples.append(next_token)
            input_token = next_token
        
        logits = torch.stack(outputs, dim=1)
        samples = torch.cat(samples, dim=1)
        return logits, samples
    
    def generate(self, noise, temperature=1.0):
        self.eval()
        with torch.no_grad():
            batch_size = noise.size(0)
            hidden = self.latent_to_hidden(noise)
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_dim)
            cell = torch.zeros_like(hidden)
            input_token = torch.ones(batch_size, 1, dtype=torch.long, device=noise.device)
            generated = []
            
            for _ in range(self.max_seq_len):
                embedded = self.embedding(input_token)
                lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
                output = self.fc(lstm_out.squeeze(1))
                logits = output / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token)
                input_token = next_token
            
            return torch.cat(generated, dim=1)


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output

print("GAN modelleri hazır!")

# =====================================================
# DATASET YÜKLEME
# =====================================================

print("\n" + "="*70)
print("DATASET YÜKLEME")
print("="*70)

def load_dataset(file_path):
    sentences = []
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        sentences = df.iloc[:, 0].astype(str).tolist()
    sentences = [s for s in sentences if s and len(s.strip()) > 0]
    return sentences

sentences = load_dataset(DATASET_PATH)
print(f"{len(sentences)} cümle yüklendi")
print(f"\nİlk 3 örnek:")
for i, s in enumerate(sentences[:3], 1):
    print(f"   {i}. {s}")

dataset = TurkishTextDataset(sentences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"\nDataLoader hazır!")
print(f"   Vocabulary: {dataset.vocab_size}")
print(f"   Batch sayısı: {len(dataloader)}")

# =====================================================
# MODEL OLUŞTURMA
# =====================================================

print("\n" + "="*70)
print("MODEL OLUŞTURMA")
print("="*70)

generator = Generator(
    vocab_size=dataset.vocab_size,
    embedding_dim=256,
    hidden_dim=512,
    latent_dim=100,
    max_seq_len=15  
).to(device)

discriminator = Discriminator(
    vocab_size=dataset.vocab_size,
    embedding_dim=256,
    hidden_dim=512
).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=0.00015, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00010, betas=(0.5, 0.999))
criterion = nn.BCELoss()

total_params = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters())
print(f"Model oluşturuldu! Toplam parametre: {total_params:,}")

# =====================================================
# EĞİTİM
# =====================================================

print("\n" + "="*70)
print(f"EĞİTİM ({NUM_EPOCHS} EPOCH)")
print("="*70)

history = {'g_loss': [], 'd_loss': [], 'd_real_acc': [], 'd_fake_acc': []}

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_g_loss = []
    epoch_d_loss = []
    epoch_d_real_acc = []
    epoch_d_fake_acc = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
    
    for batch in pbar:
        batch = batch.to(device)
        batch_size = batch.size(0)
        
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=device) * 0.9
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        real_output = discriminator(batch)
        d_real_loss = criterion(real_output, real_labels)
        
        noise = torch.randn(batch_size, 100, device=device)
        _, fake_data = generator(noise)
        fake_data = fake_data.detach()
        
        fake_output = discriminator(fake_data)
        d_fake_loss = criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        d_real_acc = (real_output > 0.5).float().mean().item()
        d_fake_acc = (fake_output < 0.5).float().mean().item()
        
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, 100, device=device)
        _, fake_data = generator(noise)
        fake_output = discriminator(fake_data)
        real_labels = torch.ones(batch_size, 1, device=device)
        
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_g.step()
        
        epoch_g_loss.append(g_loss.item())
        epoch_d_loss.append(d_loss.item())
        epoch_d_real_acc.append(d_real_acc)
        epoch_d_fake_acc.append(d_fake_acc)
        
        pbar.set_postfix({
            'G': f'{g_loss.item():.3f}',
            'D': f'{d_loss.item():.3f}',
            'D_real': f'{d_real_acc:.2f}',
            'D_fake': f'{d_fake_acc:.2f}'
        })
    
    history['g_loss'].append(np.mean(epoch_g_loss))
    history['d_loss'].append(np.mean(epoch_d_loss))
    history['d_real_acc'].append(np.mean(epoch_d_real_acc))
    history['d_fake_acc'].append(np.mean(epoch_d_fake_acc))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: G_loss={history['g_loss'][-1]:.4f}, D_loss={history['d_loss'][-1]:.4f}")

print("\nEğitim tamamlandı!")

# Grafik
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history['g_loss'], label='Generator', color='blue')
axes[0].plot(history['d_loss'], label='Discriminator', color='red')
axes[0].set_title('Loss Değerleri')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['d_real_acc'], label='Real Acc', color='green')
axes[1].plot(history['d_fake_acc'], label='Fake Acc', color='orange')
axes[1].set_title('Discriminator Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()

# =====================================================
# CÜMLE ÜRETİMİ
# =====================================================

print("\n" + "="*70)
print("CÜMLE ÜRETİMİ")
print("="*70)

generator.eval()
all_generated = []

temperatures = [0.6, 0.7, 0.8, 0.9, 1.0] 
sentences_per_temp = TOTAL_TO_GENERATE // len(temperatures)

for temp in temperatures:
    print(f"\nTemperature: {temp}")
    num_batches = (sentences_per_temp + GENERATION_BATCH_SIZE - 1) // GENERATION_BATCH_SIZE
    
    for _ in tqdm(range(num_batches), desc=f"Üretim (T={temp})"):
        current_batch = min(GENERATION_BATCH_SIZE, TOTAL_TO_GENERATE - len(all_generated))
        if current_batch <= 0:
            break
            
        noise = torch.randn(current_batch, 100, device=device)
        generated_indices = generator.generate(noise, temperature=temp)
        
        for indices in generated_indices:
            sentence = dataset.decode_sentence(indices)
            all_generated.append(sentence)
    
    if len(all_generated) >= TOTAL_TO_GENERATE:
        break

print(f"\n{len(all_generated)} cümle üretildi!")
print(f"\nİlk 5 örnek:")
for i, s in enumerate(all_generated[:5], 1):
    word_count = len(s.split())
    print(f"   {i}. [{word_count} kelime] {s}")

# =====================================================
# FİLTRELEME 
# =====================================================

print("\n" + "="*70)
print("FİLTRELEME")
print("="*70)

# 1. Kelime sayısı filtresi
filtered = [s for s in all_generated if MIN_WORDS <= len(s.split()) <= MAX_WORDS and len(s.strip()) > 0]
print(f"   Kelime sayısı ({MIN_WORDS}-{MAX_WORDS}): {len(filtered)} kaldı")

if len(filtered) == 0:
    print(f"\nUYARI: Tüm cümleler filtrelendi!")
    print(f"   Çözüm: Filtreleme olmadan en iyi 100 cümleyi alıyorum...")
    filtered = [s for s in all_generated if len(s.strip()) > 0][:100]

# 2. Tekrarları çıkarma
seen = set()
unique = []
for s in filtered:
    normalized = s.lower().strip()
    if normalized not in seen and normalized:
        seen.add(normalized)
        unique.append(s)
filtered = unique
print(f"   Tekrar temizleme: {len(filtered)} kaldı")

# 3. Orijinal ile aynı olanları çıkar
original_set = set(s.lower().strip() for s in sentences)
filtered = [s for s in filtered if s.lower().strip() not in original_set]
print(f"   Orijinal filtresi: {len(filtered)} kaldı")

# 4. Cosine similarity
if len(filtered) > 0:
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        original_vectors = vectorizer.fit_transform(sentences)
        
        final_filtered = []
        for i in range(0, len(filtered), 100):
            batch = filtered[i:i+100]
            batch_vectors = vectorizer.transform(batch)
            similarities = cosine_similarity(batch_vectors, original_vectors)
            max_sims = similarities.max(axis=1)
            
            for s, sim in zip(batch, max_sims):
                if sim < MAX_SIMILARITY:
                    final_filtered.append(s)
        
        if len(final_filtered) > 0:
            filtered = final_filtered
            print(f"   Similarity < {MAX_SIMILARITY}: {len(filtered)} kaldı")
        else:
            print(f"   Similarity kontrolü atlandı (çok katı, hepsi elendi)")
    except Exception as e:
        print(f"   Similarity atlandı (hata: {str(e)[:50]})")

print(f"\nFiltreleme tamamlandı: {len(filtered)} cümle")

# =====================================================
# KALİTE SKORLAMA 
# =====================================================

print("\n" + "="*70)
print("KALİTE SKORLAMA")
print("="*70)

def quality_score(sentence):
    if not sentence:
        return 0.0
    words = sentence.split()
    num_words = len(words)
    
    if num_words < 3 or num_words > 20:
        word_score = 0.0
    elif 5 <= num_words <= 12:
        word_score = 1.0
    else:
        word_score = 0.7
    
    avg_len = np.mean([len(w) for w in words]) if words else 0
    len_score = 1.0 if 3 <= avg_len <= 10 else 0.7
    
    unique_ratio = len(set(words)) / len(words) if words else 0
    
    short_ratio = sum(1 for w in words if len(w) <= 2) / len(words) if words else 0
    short_penalty = max(0, 1 - short_ratio * 2)
    
    score = word_score * 0.35 + len_score * 0.25 + unique_ratio * 0.25 + short_penalty * 0.15
    return round(score, 3)

if len(filtered) > 0:
    scores = [quality_score(s) for s in filtered]
    
    print(f"\nKalite İstatistikleri:")
    print(f"   Ortalama: {np.mean(scores):.3f}")
    print(f"   Medyan: {np.median(scores):.3f}")
    print(f"   Min/Max: {np.min(scores):.3f} / {np.max(scores):.3f}")
    
    # En iyileri seç
    top_n = min(TOP_N, len(filtered))
    sorted_indices = np.argsort(scores)[::-1]
    top_sentences = [filtered[i] for i in sorted_indices[:top_n]]
    top_scores = [scores[i] for i in sorted_indices[:top_n]]
    
    print(f"\nEn iyi 10 cümle:")
    for i, (s, score) in enumerate(zip(top_sentences[:10], top_scores[:10]), 1):
        print(f"   {i}. [{score:.3f}] {s}")
else:
    print("\nHATA: Hiç cümle kalmadı! Parametreleri değiştir.")
    print("\nÖneriler:")
    print("  - NUM_EPOCHS'u 80-100'e çıkarın")
    print("  - TOTAL_TO_GENERATE'i 10000'e çıkarın")
    print("  - MAX_WORDS'u 25'e çıkarın")
    top_sentences = []
    top_scores = []

# =====================================================
# CSV EXPORT
# =====================================================

if len(top_sentences) > 0:
    print("\n" + "="*70)
    print("CSV EXPORT")
    print("="*70)
    
    results_df = pd.DataFrame({
        'Cümle': top_sentences,
        'Kalite Skoru': top_scores,
        'Kelime Sayısı': [len(s.split()) for s in top_sentences],
        'Karakter Sayısı': [len(s) for s in top_sentences]
    })
    
    results_df.to_csv('uretilen_cumleler.csv', index=False, encoding='utf-8-sig')
    print(f"\n{len(results_df)} cümle kaydedildi: uretilen_cumleler.csv")
    print(f"\nİlk 5 satır:")
    print(results_df.head().to_string())
    
    # ÖZET RAPOR
    word_counts = [len(s.split()) for s in top_sentences]
    
    print("\n" + "="*70)
    print("ÖZET RAPOR")
    print("="*70)
    print(f"\nDataset: {len(sentences)} cümle")
    print(f"Vocabulary: {dataset.vocab_size} kelime")
    print(f"\nEğitim: {NUM_EPOCHS} epoch")
    print(f"   Son G Loss: {history['g_loss'][-1]:.4f}")
    print(f"   Son D Loss: {history['d_loss'][-1]:.4f}")
    print(f"\nÜretim:")
    print(f"   Ham üretilen: {len(all_generated)}")
    print(f"   Filtreleme sonrası: {len(filtered)}")
    print(f"   Seçilen en iyi: {len(top_sentences)}")
    print(f"\nKalite:")
    print(f"   Ortalama skor: {np.mean(top_scores):.3f}")
    print(f"   Ortalama kelime: {np.mean(word_counts):.1f}")
    print(f"\nÇıktılar:")
    print(f"   ✓ uretilen_cumleler.csv")
    print(f"   ✓ training_history.png")
    print("\n" + "="*70)
    print("TÜM İŞLEMLER TAMAMLANDI!")
    print("="*70)
else:
    print("\nCSV oluşturulamadı - hiç cümle yok!")
    print("Lütfen parametreleri değiştirip tekrar deneyin.")