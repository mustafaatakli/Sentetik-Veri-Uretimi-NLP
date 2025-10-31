# ===============================
# KÜTÜPHANE KURULUMU
# ===============================

!pip install openpyxl --quiet
!pip install tensorflow --quiet

print("✓ Tüm kütüphaneler hazır!\n")
print("="*70)

# ===============================
# KÜTÜPHANE İMPORT
# ===============================

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ===============================
# AYARLAR 
# ===============================

# Dosya yolları
EGITIM_DOSYASI = 'egitimveriseti.xlsx'
ETIKETSIZ_DOSYA = 'gemini_etiketli_veri.xlsx'
CIKTI_DOSYASI = 'lstm_vs_gemini_sonuc.xlsx'

# Model hiperparametreleri
MAX_WORDS = 10000          # Sözlük boyutu
MAX_LENGTH = 100           # Maksimum cümle uzunluğu
EMBEDDING_DIM = 128        # Embedding boyutu
LSTM_UNITS = 128           # LSTM unit sayısı
DROPOUT = 0.5              # Dropout oranı
BATCH_SIZE = 32            # Batch boyutu
EPOCHS = 20                # Maksimum epoch (early stopping ile duracak)
LEARNING_RATE = 0.001      # Öğrenme oranı

# ===============================
# GPU KONTROLÜ
# ===============================

print(f"TensorFlow versiyonu: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU tespit edildi: {len(gpus)} adet")
    for gpu in gpus:
        print(f"  - {gpu}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("GPU bulunamadı, CPU kullanılıyor")
print()

# ===============================
# METİN TEMİZLEME FONKSİYONU
# ===============================

def clean_text(text):
    """Türkçe metni temizle"""
    text = str(text).lower()
    text = re.sub(r'[^\wğüşöçıİĞÜŞÖÇ\s]', ' ', text)  # Türkçe karakterleri koru
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===============================
# ANA PROGRAM
# ===============================

print("="*70)
print("TÜRKÇE LSTM DUYGU ANALİZİ SİSTEMİ")
print("="*70)

# ===============================
# 1. EĞİTİM VERİSİNİ YÜKLE
# ===============================

print("\nADIM 1: Eğitim verisi yükleniyor...")
df = pd.read_excel(EGITIM_DOSYASI)
print(f"✓ {len(df)} örnek yüklendi")

# Metinleri temizle
print("\nMetinler temizleniyor...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Label mapping
label_map = {'pozitif': 2, 'negatif': 0, 'nötr': 1}
df['label'] = df['sentiment'].map(label_map)

print("\nSınıf Dağılımı:")
for sentiment, count in df['sentiment'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count} (%{percentage:.1f})")

# ===============================
# 2. (Stratified)
# ===============================

print("\nADIM 2: Veri bölünüyor (stratified split)...")

train_df, temp_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"✓ Eğitim seti: {len(train_df)} örnek")
print(f"✓ Doğrulama seti: {len(val_df)} örnek")
print(f"✓ Test seti: {len(test_df)} örnek")

# ===============================
# 3. TOKENİZASYON VE PADDING
# ===============================

print("\nADIM 3: Tokenizasyon yapılıyor...")

# Tokenizer oluştur
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['cleaned_text'])

# Metinleri sayı dizilerine çevir
X_train = tokenizer.texts_to_sequences(train_df['cleaned_text'])
X_val = tokenizer.texts_to_sequences(val_df['cleaned_text'])
X_test = tokenizer.texts_to_sequences(test_df['cleaned_text'])

# Padding
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Label'ları numpy array'e çevir
y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
print(f"✓ Sözlük boyutu: {vocab_size}")
print(f"✓ Diziler padding ile {MAX_LENGTH} uzunluğa ayarlandı")

# ===============================
# 4. LSTM MODELİNİ OLUŞTUR
# ===============================

print("\nADIM 4: LSTM modeli oluşturuluyor...")

model = Sequential([
    # Embedding katmanı
    Embedding(input_dim=vocab_size, 
              output_dim=EMBEDDING_DIM, 
              input_length=MAX_LENGTH),
    
    SpatialDropout1D(0.2),
    
    # Bidirectional LSTM katmanları
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    Bidirectional(LSTM(LSTM_UNITS // 2, dropout=0.2, recurrent_dropout=0.2)),
    
    # Dense katmanlar
    Dense(64, activation='relu'),
    Dropout(DROPOUT),
    Dense(32, activation='relu'),
    Dropout(DROPOUT),
    
    # Çıkış katmanı (3 sınıf)
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

# ===============================
# 5. CALLBACK'LER
# ===============================

callbacks = [
    ModelCheckpoint(
        'best_lstm_model.h5',
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
# 6. MODEL EĞİTİMİ
# ===============================

print("\n" + "="*70)
print("ADIM 5: MODEL EĞİTİMİ BAŞLIYOR")
print("="*70)
print(f"Maksimum epoch: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Early stopping: 5 epoch patience\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Eğitim tamamlandı!")

# ===============================
# 7. TEST SETİ DEĞERLENDİRMESİ
# ===============================

print("\n" + "="*70)
print("ADIM 6: TEST SETİ DEĞERLENDİRMESİ")
print("="*70)

# Test seti tahminleri
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
test_predictions = model.predict(X_test, verbose=0)
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
    else:
        # Gemini etiketlerini yedekle
        if 'sentiment' in new_df.columns:
            new_df['gemini_sentiment'] = new_df['sentiment']
            print("✓ Gemini etiketleri 'gemini_sentiment' sütununa yedeklendi")
        
        # Metinleri temizle ve tokenize et
        print("\nLSTM tahminleri yapılıyor...")
        new_df['cleaned_text'] = new_df['text'].apply(clean_text)
        X_new = tokenizer.texts_to_sequences(new_df['cleaned_text'])
        X_new = pad_sequences(X_new, maxlen=MAX_LENGTH, padding='post', truncating='post')
        
        # Tahmin yap
        predictions = model.predict(X_new, batch_size=64, verbose=1)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Sonuçları ekle
        label_map_reverse = {0: 'negatif', 1: 'nötr', 2: 'pozitif'}
        new_df['lstm_sentiment'] = [label_map_reverse[p] for p in pred_classes]
        new_df['lstm_confidence_negatif'] = predictions[:, 0]
        new_df['lstm_confidence_notr'] = predictions[:, 1]
        new_df['lstm_confidence_pozitif'] = predictions[:, 2]
        new_df['lstm_confidence_score'] = np.max(predictions, axis=1)
        
        # Kaydet
        new_df.to_excel(CIKTI_DOSYASI, index=False)
        print(f"\nSonuçlar '{CIKTI_DOSYASI}' dosyasına kaydedildi!")
        
        print("\nLSTM Tahmin Dağılımı:")
        for sentiment, count in new_df['lstm_sentiment'].value_counts().items():
            percentage = (count / len(new_df)) * 100
            print(f"  {sentiment}: {count} (%{percentage:.1f})")
        
        print(f"\nLSTM ortalama güven skoru: {new_df['lstm_confidence_score'].mean():.4f}")

        # ===============================
        # 9. GEMİNİ VS LSTM KARŞILAŞTIRMA
        # ===============================
        
        if 'gemini_sentiment' in new_df.columns:
            print("\n" + "="*70)
            print("GEMİNİ VS LSTM KARŞILAŞTIRMASI")
            print("="*70)
            
            # Gerçek etiketleri sayısal formata çevir
            true_labels = new_df['gemini_sentiment'].map(label_map).values
            predicted_labels = [label_map[pred] for pred in new_df['lstm_sentiment']]
            
            # Metrikler
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
            
            print("\nLSTM'in GEMİNİ ETİKETLERİNE GÖRE BAŞARI ORANI:")
            print(f"  ✓ Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
            print(f"  ✓ Precision: {precision:.4f}")
            print(f"  ✓ Recall:    {recall:.4f}")
            print(f"  ✓ F1 Score:  {f1:.4f}")
            
            # Uyuşma analizi
            agreement = (new_df['gemini_sentiment'] == new_df['lstm_sentiment']).sum()
            disagreement = len(new_df) - agreement
            
            print(f"\nUYUŞMA ANALİZİ:")
            print(f"  Uyuşan tahminler: {agreement} (%{(agreement/len(new_df))*100:.2f})")
            print(f"  Farklı tahminler: {disagreement} (%{(disagreement/len(new_df))*100:.2f})")
            
            # Sınıf bazında detaylı rapor
            print("\nSINIF BAZINDA KARŞILAŞTIRMA:")
            print(classification_report(true_labels, predicted_labels, target_names=label_names))
            
            # Confusion Matrix
            print("\nKARMAŞIKLIK MATRİSİ (Gemini vs LSTM):")
            cm = confusion_matrix(true_labels, predicted_labels)
            print("\n                  LSTM Tahmini")
            print("              negatif  nötr  pozitif")
            for i, label in enumerate(label_names):
                print(f"Gemini {label:8s}  {cm[i][0]:4d}   {cm[i][1]:4d}   {cm[i][2]:4d}")
            
            # Farklı tahmin örnekleri
            print("\nFARKLI TAHMİN ÖRNEKLERİ (İlk 10):")
            different_predictions = new_df[new_df['gemini_sentiment'] != new_df['lstm_sentiment']].head(10)
            
            if len(different_predictions) > 0:
                for idx, row in different_predictions.iterrows():
                    print(f"\nCümle: {row['text'][:80]}...")
                    print(f"     Gemini: {row['gemini_sentiment']:8s} | LSTM: {row['lstm_sentiment']:8s} | Güven: {row['lstm_confidence_score']:.3f}")
            else:
                print("Tüm tahminler uyuşuyor!")
            
            # Güven skoruna göre analiz
            print("\nGÜVEN SKORUNA GÖRE ANALİZ:")
            
            agreement_mask = new_df['gemini_sentiment'] == new_df['lstm_sentiment']
            
            avg_confidence_agree = new_df[agreement_mask]['lstm_confidence_score'].mean()
            avg_confidence_disagree = new_df[~agreement_mask]['lstm_confidence_score'].mean()
            
            print(f"  Uyuşan tahminlerde LSTM güveni: {avg_confidence_agree:.4f}")
            print(f"  Farklı tahminlerde LSTM güveni: {avg_confidence_disagree:.4f}")
            
            # Düşük güvenli farklı tahminler
            low_conf_different = new_df[(~agreement_mask) & (new_df['lstm_confidence_score'] < 0.7)]
            print(f"\n  Düşük güvenle farklı tahmin edilen: {len(low_conf_different)} adet")
            
            if len(low_conf_different) > 0:
                print(f"  (Bu tahminler belirsiz olabilir, manuel kontrol önerilir)")
            
            # Sınıf bazında uyuşma
            print("\nSINIF BAZINDA UYUŞMA ORANLARI:")
            for sentiment in ['pozitif', 'negatif', 'nötr']:
                gemini_subset = new_df[new_df['gemini_sentiment'] == sentiment]
                if len(gemini_subset) > 0:
                    agree_count = (gemini_subset['gemini_sentiment'] == gemini_subset['lstm_sentiment']).sum()
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

# Keras modelini kaydet
model.save('lstm_sentiment_model.h5')

# Tokenizer'ı kaydet
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✓ Model 'lstm_sentiment_model.h5' olarak kaydedildi!")
print("✓ Tokenizer 'tokenizer.pickle' olarak kaydedildi!")

# ===============================
# 11. EĞİTİM GRAFİKLERİ
# ===============================

print("\nEğitim grafiklerini çizmek için:")
print("""
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
""")

print("\n" + "="*70)
print("TAMAMLANDI!")
print("="*70)
print("\nSonuçlar:")
print(f"1. ✓ Karşılaştırma dosyası: {CIKTI_DOSYASI}")
print("2. ✓ Model dosyası: lstm_sentiment_model.h5")
print("3. ✓ Tokenizer: tokenizer.pickle")
print("4. ✓ Gemini vs LSTM karşılaştırması tamamlandı")
print("\n  İpucu: Excel dosyasında şu sütunlar var:")
print("   - gemini_sentiment: Gemini'nin etiketleri")
print("   - lstm_sentiment: LSTM'in etiketleri")
print("   - lstm_confidence_score: LSTM'in güven skoru")