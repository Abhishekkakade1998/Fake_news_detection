# ==========================================
# 1. Install & Import Libraries 
# ==========================================

# !pip install -q tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

print("TensorFlow Version:", tf.__version__)


# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv("fake_news_dataset_4000_rows.csv")

print(df.head())
print("\nLabel Distribution:\n", df['label'].value_counts())
print("\nLabel Balance (%):\n", df['label'].value_counts(normalize=True) * 100)


# ==========================================
# 3. Preprocessing
# ==========================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


X = df['text'].astype(str).apply(preprocess)
y = df['label']

# Encode labels if not already 0/1
if y.dtype == object:
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================================
# 4. Tokenization & Padding
# ==========================================

vocab_size = 10000
max_length = 150
embedding_dim = 64

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')


# ==========================================
# 5. Handle Class Imbalance
# ==========================================

unique, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
class_weight = {int(cls): total / count for cls, count in zip(unique, counts)}
print("\nClass Weights:", class_weight)


# ==========================================
# 6. Build BiLSTM Model
# ==========================================


model = Sequential()

model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_length
))

model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()


# ==========================================
# 7. Callbacks
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "final_fake_news_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)


# ==========================================
# 8. Train Model
# ==========================================

history = model.fit(
    X_train_pad,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)


# ==========================================
# 9. Evaluate Model
# ==========================================

loss, accuracy, precision, recall, auc = model.evaluate(X_test_pad, y_test)
print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test AUC      : {auc:.4f}")


# ==========================================
# 10. Predictions & Classification Report
# ==========================================

y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================
# 11. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ==========================================
# 12. Plot Accuracy & Loss
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()


# ==========================================
# 13. Save Best Model & Tokenizer
# ==========================================

# Load best checkpoint weights before saving final model
model.save("final_fake_news_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
