import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "RAVDESS"
MODEL_PATH = "emotion_model.h5"
LABEL_ENCODER_PATH = "label_encoder.npy"
MAX_LEN = 174  # sequence length (can be adjusted based on dataset)

# -----------------------------
# LOAD AUDIO FILES AND EXTRACT MFCC
# -----------------------------
emotions = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

X = []
y = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            label = emotions[file.split("-")[2]]  # emotion code from filename
            audio, sr = librosa.load(path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc = mfcc.T  # shape (time_steps, n_mfcc)
            X.append(mfcc)
            y.append(label)

# -----------------------------
# PAD SEQUENCES
# -----------------------------
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_padded = pad_sequences(X, maxlen=MAX_LEN, dtype="float32", padding="post", truncating="post")

# -----------------------------
# ENCODE LABELS
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
np.save(LABEL_ENCODER_PATH, le.classes_)

# -----------------------------
# BUILD BiLSTM MODEL
# -----------------------------
model = Sequential([
    Masking(mask_value=0., input_shape=(MAX_LEN, 40)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='accuracy', save_best_only=True, verbose=1)
model.fit(X_padded, y_categorical, epochs=60, batch_size=32, validation_split=0.1, callbacks=[checkpoint])
