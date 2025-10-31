# ===============================
# KÜTÜPHANE KURULUMU
# ===============================

print("Gerekli kütüphaneler kontrol ediliyor...\n")

# Transformers ve openpyxl'i yükle (yoksa)
!pip install transformers --quiet
!pip install openpyxl --quiet

print("✓ Tüm kütüphaneler hazır!\n")
print("="*70)

# ===============================
# KÜTÜPHANE İMPORT
# ===============================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # AdamW artık torch.optim'de
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ===============================
# AYARLAR - BURADAN DEĞİŞTİRİN
# ===============================

# Dosya yolları
EGITIM_DOSYASI = '/kaggle/input/muh-proje/egitim-veriseti.xlsx'           # Eğitim verisi
ETIKETSIZ_DOSYA = '/kaggle/input/muh-proje/etiketsiz-test-gemini-etiketlenmis.xlsx'   # Gemini etiketli dosyanız
CIKTI_DOSYASI = '/kaggle/working/bert_vs_gemini_sonuc.xlsx'     # Sonuç dosyası

# Model ayarları
MODEL_ADI = 'dbmdz/bert-base-turkish-cased'     
EPOCHS = 4                                      
BATCH_SIZE = 16                                  
LEARNING_RATE = 2e-5                            
MAX_LENGTH = 128                               

# ===============================
# GPU KONTROLÜ
# ===============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
else:
    print("GPU bulunamadı, CPU kullanılıyor (yavaş olacak!)\n")

# ===============================
# DATASET SINIFI
# ===============================

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ===============================
# EĞİTİM FONKSİYONLARI
# ===============================

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(data_loader), accuracy_score(actual_labels, predictions)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average='weighted')
    
    return total_loss / len(data_loader), accuracy, precision, recall, f1, predictions, actual_labels

# ===============================
# ANA PROGRAM
# ===============================

print("="*70)
print("TÜRKÇE BERT DUYGU ANALİZİ SİSTEMİ")
print("="*70)

# ===============================
# 1. EĞİTİM VERİSİNİ YÜKLE
# ===============================

print("\nADIM 1: Eğitim verisi yükleniyor...")
df = pd.read_excel(EGITIM_DOSYASI)
print(f"✓ {len(df)} örnek yüklendi")

# Label mapping
label_map = {'pozitif': 2, 'negatif': 0, 'nötr': 1}
df['label'] = df['sentiment'].map(label_map)

print("\nSınıf Dağılımı:")
for sentiment, count in df['sentiment'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count} (%{percentage:.1f})")

# ===============================
# 2. VERİYİ BÖL (Stratified)
# ===============================

print("\nADIM 2: Veri bölünüyor (stratified split)...")

train_df, temp_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"✓ Eğitim seti: {len(train_df)} örnek")
print(f"✓ Doğrulama seti: {len(val_df)} örnek")
print(f"✓ Test seti: {len(test_df)} örnek")

# ===============================
# 3. MODEL VE TOKENİZER YÜKLE
# ===============================

print(f"\nADIM 3: Model yükleniyor ({MODEL_ADI})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ADI, num_labels=3).to(device)
print("✓ Model yüklendi")

# ===============================
# 4. DATASET VE DATALOADER OLUŞTUR
# ===============================

print("\nADIM 4: Dataset'ler oluşturuluyor...")

train_dataset = SentimentDataset(train_df['text'].values, train_df['label'].values, tokenizer, MAX_LENGTH)
val_dataset = SentimentDataset(val_df['text'].values, val_df['label'].values, tokenizer, MAX_LENGTH)
test_dataset = SentimentDataset(test_df['text'].values, test_df['label'].values, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("✓ Dataset'ler hazır")

# ===============================
# 5. OPTİMİZER VE SCHEDULER
# ===============================

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ===============================
# 6. MODELİ EĞİT
# ===============================

print("\n" + "="*70)
print("ADIM 5: MODEL EĞİTİMİ BAŞLIYOR")
print("="*70)
print(f"Epoch sayısı: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}\n")

best_val_accuracy = 0

for epoch in range(EPOCHS):
    print(f"\n{'─'*70}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'─'*70}")
    
    # Eğitim
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Eğitim    → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    
    # Doğrulama
    val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device)
    print(f"Doğrulama → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    # En iyi modeli kaydet
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"En iyi model kaydedildi! (Accuracy: {val_acc:.4f})")

# En iyi modeli yükle
print("\nEn iyi model yükleniyor...")
model.load_state_dict(torch.load('best_model.pt'))

# ===============================
# 7. TEST SETİ DEĞERLENDİRMESİ
# ===============================

print("\n" + "="*70)
print("ADIM 6: TEST SETİ DEĞERLENDİRMESİ")
print("="*70)

test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)

print(f"\nTest Sonuçları:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

print("\nDetaylı Sınıflandırma Raporu:")
label_names = ['negatif', 'nötr', 'pozitif']
print(classification_report(test_labels, test_preds, target_names=label_names))

# ===============================
# 8. YENİ VERİ SETİNİ ETİKETLE
# ===============================

print("\n" + "="*70)
print("ADIM 7: YENİ VERİ SETİ ETİKETLENİYOR")
print("="*70)

try:
    # Etiketsiz veriyi yükle
    print(f"\nDosya okunuyor: {ETIKETSIZ_DOSYA}")
    new_df = pd.read_excel(ETIKETSIZ_DOSYA)
    print(f"✓ {len(new_df)} örnek yüklendi")
    
    if 'text' not in new_df.columns:
        print("HATA: Excel dosyasında 'text' sütunu bulunamadı!")
        print("Lütfen Excel dosyanızda 'text' adında bir sütun olduğundan emin olun.")
    else:
        # Gemini etiketlerini yedekle
        if 'sentiment' in new_df.columns:
            new_df['gemini_sentiment'] = new_df['sentiment']
            print("✓ Gemini etiketleri 'gemini_sentiment' sütununa yedeklendi")
        
        model.eval()
        predictions = []
        probabilities = []
        
        print("\nBERT tahminleri yapılıyor...")
        batch_size = 32
        
        for i in range(0, len(new_df), batch_size):
            batch_texts = new_df['text'].iloc[i:i+batch_size].tolist()
            
            encodings = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  İşlenen: {min(i+batch_size, len(new_df))}/{len(new_df)}")
        
        # BERT sonuçlarını ekle
        label_map_reverse = {0: 'negatif', 1: 'nötr', 2: 'pozitif'}
        new_df['bert_sentiment'] = [label_map_reverse[p] for p in predictions]
        new_df['bert_confidence_negatif'] = [p[0] for p in probabilities]
        new_df['bert_confidence_notr'] = [p[1] for p in probabilities]
        new_df['bert_confidence_pozitif'] = [p[2] for p in probabilities]
        new_df['bert_confidence_score'] = [max(p) for p in probabilities]
        
        # Kaydet
        new_df.to_excel(CIKTI_DOSYASI, index=False)
        print(f"\nSonuçlar '{CIKTI_DOSYASI}' dosyasına kaydedildi!")
        
        print("\nBERT Tahmin Dağılımı:")
        for sentiment, count in new_df['bert_sentiment'].value_counts().items():
            percentage = (count / len(new_df)) * 100
            print(f"  {sentiment}: {count} (%{percentage:.1f})")
        
        print(f"\nBERT ortalama güven skoru: {new_df['bert_confidence_score'].mean():.4f}")

        # ===============================
        # 9. GEMİNİ VS BERT KARŞILAŞTIRMA
        # ===============================
        
        if 'gemini_sentiment' in new_df.columns:
            print("\n" + "="*70)
            print("GEMİNİ VS BERT KARŞILAŞTIRMASI")
            print("="*70)
            
            # Gerçek etiketleri sayısal formata çevir
            true_labels = new_df['gemini_sentiment'].map(label_map).values
            predicted_labels = [label_map[pred] for pred in new_df['bert_sentiment']]
            
            # Metrikler
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
            
            print("\nBERT'ün GEMİNİ ETİKETLERİNE GÖRE BAŞARI ORANI:")
            print(f"  ✓ Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
            print(f"  ✓ Precision: {precision:.4f}")
            print(f"  ✓ Recall:    {recall:.4f}")
            print(f"  ✓ F1 Score:  {f1:.4f}")
            
            # Uyuşma analizi
            agreement = (new_df['gemini_sentiment'] == new_df['bert_sentiment']).sum()
            disagreement = len(new_df) - agreement
            
            print(f"\nUYUŞMA ANALİZİ:")
            print(f"  Uyuşan tahminler: {agreement} (%{(agreement/len(new_df))*100:.2f})")
            print(f"  Farklı tahminler: {disagreement} (%{(disagreement/len(new_df))*100:.2f})")
            
            # Sınıf bazında detaylı rapor
            print("\nSINIF BAZINDA KARŞILAŞTIRMA:")
            print(classification_report(true_labels, predicted_labels, target_names=label_names))
            
            # Confusion Matrix
            print("\nKARMAŞIKLIK MATRİSİ (Gemini vs BERT):")
            cm = confusion_matrix(true_labels, predicted_labels)
            print("\n                  BERT Tahmini")
            print("              negatif  nötr  pozitif")
            for i, label in enumerate(label_names):
                print(f"Gemini {label:8s}  {cm[i][0]:4d}   {cm[i][1]:4d}   {cm[i][2]:4d}")
            
            # Farklı tahmin örnekleri
            print("\nFARKLI TAHMİN ÖRNEKLERİ (İlk 10):")
            different_predictions = new_df[new_df['gemini_sentiment'] != new_df['bert_sentiment']].head(10)
            
            if len(different_predictions) > 0:
                for idx, row in different_predictions.iterrows():
                    print(f"\n  Cümle: {row['text'][:80]}...")
                    print(f"     Gemini: {row['gemini_sentiment']:8s} | BERT: {row['bert_sentiment']:8s} | Güven: {row['bert_confidence_score']:.3f}")
            else:
                print("Tüm tahminler uyuşuyor!")
            
            # Güven skoruna göre analiz
            print("\nGÜVEN SKORUNA GÖRE ANALİZ:")
            
            # Uyuşan ve farklı tahminlerin güven skorları
            agreement_mask = new_df['gemini_sentiment'] == new_df['bert_sentiment']
            
            avg_confidence_agree = new_df[agreement_mask]['bert_confidence_score'].mean()
            avg_confidence_disagree = new_df[~agreement_mask]['bert_confidence_score'].mean()
            
            print(f"  Uyuşan tahminlerde BERT güveni: {avg_confidence_agree:.4f}")
            print(f"  Farklı tahminlerde BERT güveni: {avg_confidence_disagree:.4f}")
            
            # Düşük güvenli farklı tahminler
            low_conf_different = new_df[(~agreement_mask) & (new_df['bert_confidence_score'] < 0.7)]
            print(f"\n  Düşük güvenle farklı tahmin edilen: {len(low_conf_different)} adet")
            
            if len(low_conf_different) > 0:
                print(f"  (Bu tahminler belirsiz olabilir, manuel kontrol önerilir)")
            
            # Sınıf bazında uyuşma
            print("\nSINIF BAZINDA UYUŞMA ORANLARI:")
            for sentiment in ['pozitif', 'negatif', 'nötr']:
                gemini_subset = new_df[new_df['gemini_sentiment'] == sentiment]
                if len(gemini_subset) > 0:
                    agree_count = (gemini_subset['gemini_sentiment'] == gemini_subset['bert_sentiment']).sum()
                    agree_pct = (agree_count / len(gemini_subset)) * 100
                    print(f"  {sentiment:8s}: {agree_count}/{len(gemini_subset)} (%{agree_pct:.1f})")

except FileNotFoundError:
    print(f"\n'{ETIKETSIZ_DOSYA}' dosyası bulunamadı!")
    print("   Dosyayı yükleyin ve ETIKETSIZ_DOSYA değişkenini güncelleyin.")

# ===============================
# 10. MODEL KAYDET
# ===============================

print("\n" + "="*70)
print("ADIM 8: Model kaydediliyor...")
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
print("✓ Model ve tokenizer './sentiment_model' klasörüne kaydedildi!")

print("\n" + "="*70)
print("TAMAMLANDI!")
print("="*70)
print("\nSonuçlar:")
print(f"1. ✓ Karşılaştırma dosyası: {CIKTI_DOSYASI}")
print("2. ✓ Model dosyaları: ./sentiment_model klasöründe")
print("3. ✓ Gemini vs BERT karşılaştırması tamamlandı")
print("\n  İpucu: Excel dosyasında şu sütunlar var:")
print("   - gemini_sentiment: Gemini'nin etiketleri")
print("   - bert_sentiment: BERT'ün etiketleri")
print("   - bert_confidence_score: BERT'ün güven skoru")