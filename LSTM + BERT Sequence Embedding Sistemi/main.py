"""
LSTM + BERT Embedding Hibrit Duygu Analizi Sistemi
- BERT'ten HER KELİME için embedding alır (sequence)
- LSTM ile sequence'i işler
- Gemini ile karşılaştırır

KULLANIM:
1. Kaggle'da GPU T4 aktif edin
2. 'egitimveriseti.xlsx' ve Gemini etiketli dosyayı yükleyin
3. Bu kodu çalıştırın
"""

# ===============================
# KÜTÜPHANE KURULUMU
# ===============================

print("Gerekli kütüphaneler kontrol ediliyor...\n")

!pip
install
transformers - -quiet
!pip
install
openpyxl - -quiet
!pip
install
tensorflow - -quiet

print("Tüm kütüphaneler hazır!\n")
print("=" * 70)

# ===============================
# KÜTÜPHANE İMPORT
# ===============================

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Transformers (BERT için)
from transformers import AutoTokenizer, AutoModel
import torch

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# ===============================
# AYARLAR
# ===============================

# Dosya yolları
EGITIM_DOSYASI = '/kaggle/input/muh-proje3/egitim-veriseti.xlsx'
ETIKETSIZ_DOSYA = '/kaggle/input/muh-proje3/etiketsiz-test-gemini-etiketlenmis.xlsx'
CIKTI_DOSYASI = '/kaggle/working/bert_vs_gemini_sonuc.xlsx'

# Model ayarları
BERT_MODEL = 'dbmdz/bert-base-turkish-cased'
MAX_LENGTH = 64  
LSTM_UNITS = 128 
DROPOUT = 0.3  
BATCH_SIZE = 16  
EPOCHS = 20  
LEARNING_RATE = 0.001  

# ===============================
# GPU KONTROLÜ
# ===============================

print(f"TensorFlow versiyonu: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU tespit edildi: {len(gpus)} adet")
        for gpu in gpus:
            print(f"  - {gpu}")
        print("✓ GPU memory growth ayarlandı")
    except RuntimeError as e:
        print(f"GPU ayarı hatası: {e}")
else:
    print("GPU bulunamadı, CPU kullanılıyor")

# PyTorch GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}\n")


# ===============================
# BERT SEQUENCE EMBEDDING EXTRACTION
# ===============================

def extract_bert_sequence_embeddings(texts, tokenizer, model, max_length, batch_size=8):
    """
    BERT'ten cümleler için SEQUENCE embedding'lerini çıkar
    Her kelime için ayrı vektör (LSTM için gerekli)

    Args:
        texts: Metin listesi
        tokenizer: BERT tokenizer
        model: BERT modeli
        max_length: Maksimum sequence uzunluğu
        batch_size: Batch boyutu

    Returns:
        embeddings: (n_samples, max_length, 768) shape'de numpy array
        attention_masks: (n_samples, max_length) - padding için
    """
    model.eval()
    all_embeddings = []
    all_masks = []

    print(f"BERT sequence embedding'leri çıkarılıyor... ({len(texts)} örnek)")
    print(f"   Her cümle için {max_length} token × 768 boyut = sequence")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize (padding ve truncation)
        encoded = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # BERT'ten TÜM token'ların embedding'lerini al
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: (batch_size, sequence_length, 768)
            sequence_embeddings = outputs.last_hidden_state.cpu().numpy()
            attention_masks = attention_mask.cpu().numpy()

            all_embeddings.append(sequence_embeddings)
            all_masks.append(attention_masks)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  İşlenen: {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    masks = np.vstack(all_masks)

    print(f"✓ Sequence Embedding shape: {embeddings.shape}")
    print(f"  → {embeddings.shape[0]} cümle")
    print(f"  → {embeddings.shape[1]} token/cümle (sequence length)")
    print(f"  → {embeddings.shape[2]} boyutlu vektör/token")

    return embeddings, masks


# ===============================
# ANA PROGRAM
# ===============================

print("=" * 70)
print("LSTM + BERT SEQUENCE EMBEDDING SİSTEMİ")
print("=" * 70)

# ===============================
# 1. BERT MODELİNİ YÜKLE
# ===============================

print("\nADIM 1: BERT modeli yükleniyor (sequence embedding için)...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)
bert_model.eval()
print("✓ BERT hazır (her kelime için embedding çıkaracak)")

# ===============================
# 2. EĞİTİM VERİSİNİ YÜKLE
# ===============================

print("\nADIM 2: Eğitim verisi yükleniyor...")
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
# 3. VERİYİ BÖL
# ===============================

print("\nADIM 3: Veri bölünüyor (stratified split)...")

train_df, temp_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"✓ Eğitim seti: {len(train_df)} örnek")
print(f"✓ Doğrulama seti: {len(val_df)} örnek")
print(f"✓ Test seti: {len(test_df)} örnek")

# ===============================
# 4. BERT SEQUENCE EMBEDDING'LERİNİ ÇIKAR
# ===============================

print("\nADIM 4: BERT sequence embedding'leri çıkarılıyor...")
print("(Her kelime için ayrı vektör - LSTM için gerekli!)")
print("(Bu işlem biraz zaman alabilir...)\n")

# Eğitim seti
X_train_seq, X_train_masks = extract_bert_sequence_embeddings(
    train_df['text'].tolist(),
    tokenizer,
    bert_model,
    MAX_LENGTH
)

# Doğrulama seti
X_val_seq, X_val_masks = extract_bert_sequence_embeddings(
    val_df['text'].tolist(),
    tokenizer,
    bert_model,
    MAX_LENGTH
)

# Test seti
X_test_seq, X_test_masks = extract_bert_sequence_embeddings(
    test_df['text'].tolist(),
    tokenizer,
    bert_model,
    MAX_LENGTH
)

# Label'lar
y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

print(f"\n✓ Sequence Embedding'ler hazır!")
print(f"  Train shape: {X_train_seq.shape} (samples × sequence × embedding_dim)")
print(f"  Val shape: {X_val_seq.shape}")
print(f"  Test shape: {X_test_seq.shape}")

# BERT modelini bellekten temizleme
del bert_model
torch.cuda.empty_cache()
print("\n✓ BERT modeli bellekten temizlendi (artık sadece LSTM kullanılacak)")

# ===============================
# 5. GERÇEK LSTM MODELİNİ OLUŞTUR
# ===============================

print("\nADIM 5: GERÇEK LSTM modeli oluşturuluyor...")
print("(Input: BERT sequence embeddings - her kelime için 768-dim vektör)")


model = Sequential([
    # Input: Sequence of embeddings (max_length, 768)
    Input(shape=(MAX_LENGTH, 768)),

    # Masking layer (padding için)
    Masking(mask_value=0.0),

    # Bidirectional LSTM (hem ileri hem geri oku)
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),

    # İkinci LSTM katmanı
    Bidirectional(LSTM(LSTM_UNITS // 2, dropout=0.2, recurrent_dropout=0.2)),

    # Dense katmanlar
    Dense(64, activation='relu'),
    Dropout(DROPOUT),

    Dense(32, activation='relu'),
    Dropout(DROPOUT),

    # Çıkış katmanı
    Dense(3, activation='softmax')
])

# Model derleme
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Mimarisi:")
model.summary()

print("\nBu model:")
print("  1. BERT'ten gelen sequence'i alıyor (her kelime ayrı vektör)")
print("  2. Bidirectional LSTM ile hem ileri hem geri okuyor")
print("  3. İkinci LSTM katmanı ile özetliyor")
print("  4. Dense katmanlarla sınıflandırıyor")

# ===============================
# 6. CALLBACK'LER
# ===============================

callbacks = [
    ModelCheckpoint(
        'best_real_lstm_bert_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# ===============================
# 7. MODEL EĞİTİMİ
# ===============================

print("\n" + "=" * 70)
print("🎓 ADIM 6: MODEL EĞİTİMİ BAŞLIYOR")
print("=" * 70)
print(f"Maksimum epoch: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Input: BERT sequence embeddings ({MAX_LENGTH} × 768)")
print(f"LSTM: Bidirectional (2 katman)")
print(f"Early stopping: 5 epoch patience\n")

history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Eğitim tamamlandı!")

# ===============================
# 8. TEST SETİ DEĞERLENDİRMESİ
# ===============================

print("\n" + "=" * 70)
print("ADIM 7: TEST SETİ DEĞERLENDİRMESİ")
print("=" * 70)

# Test seti tahminleri
test_loss, test_accuracy = model.evaluate(X_test_seq, y_test, verbose=0)
test_predictions = model.predict(X_test_seq, verbose=0)
test_pred_classes = np.argmax(test_predictions, axis=1)

# Metrikler
precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred_classes, average='weighted')

print(f"\nTest Sonuçları:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

print("\nDetaylı Sınıflandırma Raporu:")
label_names = ['negatif', 'nötr', 'pozitif']
print(classification_report(y_test, test_pred_classes, target_names=label_names))

print("\nKarmaşıklık Matrisi:")
cm = confusion_matrix(y_test, test_pred_classes)
print("\n              Tahmin Edilen")
print("          negatif  nötr  pozitif")
for i, label in enumerate(label_names):
    print(f"Gerçek {label:8s}  {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}")

# ===============================
# 9. YENİ VERİ SETİNİ ETİKETLE
# ===============================

print("\n" + "=" * 70)
print("ADIM 8: YENİ VERİ SETİ ETİKETLENİYOR")
print("=" * 70)

try:
    
    print(f"\nDosya okunuyor: {ETIKETSIZ_DOSYA}")
    new_df = pd.read_excel(ETIKETSIZ_DOSYA)
    print(f"✓ {len(new_df)} örnek yüklendi")

    if 'text' not in new_df.columns:
        print("HATA: Excel dosyasında 'text' sütunu bulunamadı!")
    else:
        # Gemini etiketlerini yedekle
        if 'sentiment' in new_df.columns:
            new_df['gemini_sentiment'] = new_df['sentiment']
            print("✓ Gemini etiketleri 'gemini_sentiment' sütununa yedeklendi")

        # BERT'i tekrar yükle (tahmin için)
        print("\nBERT modeli yeniden yükleniyor (tahmin için)...")
        bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)
        bert_model.eval()

        # Yeni veriler için BERT sequence embedding'leri çıkar
        X_new_seq, X_new_masks = extract_bert_sequence_embeddings(
            new_df['text'].tolist(),
            tokenizer,
            bert_model,
            MAX_LENGTH
        )

        # LSTM ile tahmin yap
        print("\nLSTM tahminleri yapılıyor...")
        predictions = model.predict(X_new_seq, batch_size=32, verbose=1)
        pred_classes = np.argmax(predictions, axis=1)

        label_map_reverse = {0: 'negatif', 1: 'nötr', 2: 'pozitif'}
        new_df['real_lstm_bert_sentiment'] = [label_map_reverse[p] for p in pred_classes]
        new_df['real_lstm_bert_conf_negatif'] = predictions[:, 0]
        new_df['real_lstm_bert_conf_notr'] = predictions[:, 1]
        new_df['real_lstm_bert_conf_pozitif'] = predictions[:, 2]
        new_df['real_lstm_bert_conf_score'] = np.max(predictions, axis=1)

        # Kaydet
        new_df.to_excel(CIKTI_DOSYASI, index=False)
        print(f"\nSonuçlar '{CIKTI_DOSYASI}' dosyasına kaydedildi!")

        print("\nGERÇEK LSTM+BERT Tahmin Dağılımı:")
        for sentiment, count in new_df['real_lstm_bert_sentiment'].value_counts().items():
            percentage = (count / len(new_df)) * 100
            print(f"  {sentiment}: {count} (%{percentage:.1f})")

        print(f"\nGERÇEK LSTM+BERT ortalama güven skoru: {new_df['real_lstm_bert_conf_score'].mean():.4f}")

        # ===============================
        # 10. GEMİNİ VS LSTM+BERT KARŞILAŞTIRMA
        # ===============================

        if 'gemini_sentiment' in new_df.columns:
            print("\n" + "=" * 70)
            print("GEMİNİ VS GERÇEK LSTM+BERT KARŞILAŞTIRMASI")
            print("=" * 70)

            # Gerçek etiketleri sayısal formata çevir
            true_labels = new_df['gemini_sentiment'].map(label_map).values
            predicted_labels = [label_map[pred] for pred in new_df['real_lstm_bert_sentiment']]

            # Metrikler
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                       average='weighted')

            print("\nGERÇEK LSTM+BERT'ün GEMİNİ ETİKETLERİNE GÖRE BAŞARI ORANI:")
            print(f"  ✓ Accuracy:  {accuracy:.4f}  ({accuracy * 100:.2f}%)")
            print(f"  ✓ Precision: {precision:.4f}")
            print(f"  ✓ Recall:    {recall:.4f}")
            print(f"  ✓ F1 Score:  {f1:.4f}")

            # Uyuşma analizi
            agreement = (new_df['gemini_sentiment'] == new_df['real_lstm_bert_sentiment']).sum()
            disagreement = len(new_df) - agreement

            print(f"\nUYUŞMA ANALİZİ:")
            print(f"  Uyuşan tahminler: {agreement} (%{(agreement / len(new_df)) * 100:.2f})")
            print(f"  Farklı tahminler: {disagreement} (%{(disagreement / len(new_df)) * 100:.2f})")

            # Sınıf bazında detaylı rapor
            print("\nSINIF BAZINDA KARŞILAŞTIRMA:")
            print(classification_report(true_labels, predicted_labels, target_names=label_names))

            # Confusion Matrix
            print("\nKARMAŞIKLIK MATRİSİ (Gemini vs GERÇEK LSTM+BERT):")
            cm = confusion_matrix(true_labels, predicted_labels)
            print("\n                  LSTM+BERT Tahmini")
            print("              negatif  nötr  pozitif")
            for i, label in enumerate(label_names):
                print(f"Gemini {label:8s}  {cm[i][0]:4d}   {cm[i][1]:4d}   {cm[i][2]:4d}")

            # Farklı tahmin örnekleri
            print("\nFARKLI TAHMİN ÖRNEKLERİ (İlk 10):")
            different_predictions = new_df[new_df['gemini_sentiment'] != new_df['real_lstm_bert_sentiment']].head(10)

            if len(different_predictions) > 0:
                for idx, row in different_predictions.iterrows():
                    print(f"\n  Cümle: {row['text'][:80]}...")
                    print(
                        f"     Gemini: {row['gemini_sentiment']:8s} | LSTM+BERT: {row['real_lstm_bert_sentiment']:8s} | Güven: {row['real_lstm_bert_conf_score']:.3f}")
            else:
                print("Tüm tahminler uyuşuyor!")

            # Güven skoruna göre analiz
            print("\nGÜVEN SKORUNA GÖRE ANALİZ:")

            agreement_mask = new_df['gemini_sentiment'] == new_df['real_lstm_bert_sentiment']

            avg_confidence_agree = new_df[agreement_mask]['real_lstm_bert_conf_score'].mean()
            avg_confidence_disagree = new_df[~agreement_mask]['real_lstm_bert_conf_score'].mean()

            print(f"  Uyuşan tahminlerde LSTM+BERT güveni: {avg_confidence_agree:.4f}")
            print(f"  Farklı tahminlerde LSTM+BERT güveni: {avg_confidence_disagree:.4f}")

            # Düşük güvenli farklı tahminler
            low_conf_different = new_df[(~agreement_mask) & (new_df['real_lstm_bert_conf_score'] < 0.7)]
            print(f"\n  Düşük güvenle farklı tahmin edilen: {len(low_conf_different)} adet")

            if len(low_conf_different) > 0:
                print(f"  (Bu tahminler belirsiz olabilir, manuel kontrol önerilir)")

            # Sınıf bazında uyuşma
            print("\nSINIF BAZINDA UYUŞMA ORANLARI:")
            for sentiment in ['pozitif', 'negatif', 'nötr']:
                gemini_subset = new_df[new_df['gemini_sentiment'] == sentiment]
                if len(gemini_subset) > 0:
                    agree_count = (gemini_subset['gemini_sentiment'] == gemini_subset['real_lstm_bert_sentiment']).sum()
                    agree_pct = (agree_count / len(gemini_subset)) * 100
                    print(f"  {sentiment:8s}: {agree_count}/{len(gemini_subset)} (%{agree_pct:.1f})")

except FileNotFoundError:
    print(f"\n'{ETIKETSIZ_DOSYA}' dosyası bulunamadı!")
    print("   Dosyayı yükleyin ve ETIKETSIZ_DOSYA değişkenini güncelleyin.")

# ===============================
# 11. MODEL KAYDET
# ===============================

print("\n" + "=" * 70)
print("ADIM 9: Model kaydediliyor...")

model.save('real_lstm_bert_model.h5')

import pickle

config = {
    'bert_model': BERT_MODEL,
    'max_length': MAX_LENGTH,
    'lstm_units': LSTM_UNITS
}
with open('real_lstm_bert_config.pickle', 'wb') as f:
    pickle.dump(config, f)

print("✓ Model 'real_lstm_bert_model.h5' olarak kaydedildi!")
print("✓ Config 'real_lstm_bert_config.pickle' olarak kaydedildi!")

print("\n" + "=" * 70)
print("TAMAMLANDI!")
print("=" * 70)
print("\nSonuçlar:")
print(f"1. ✓ Karşılaştırma dosyası: {CIKTI_DOSYASI}")
print("2. ✓ Model dosyası: real_lstm_bert_model.h5")
print("3. ✓ Config: real_lstm_bert_config.pickle")
print("4. ✓ Gemini vs GERÇEK LSTM+BERT karşılaştırması tamamlandı")
print("\nBu model:")
print("BERT'ten HER KELİME için embedding alıyor (sequence)")
print("Bidirectional LSTM ile sequence'i işliyor")
print("Gerçek anlamda 'LSTM + BERT Embedding' yapıyor")
print("\nBeklenen performans: %89-92 accuracy")
print("   (Önceki Dense versiyondan %2-4 daha iyi olmalı)")