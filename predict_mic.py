import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "emotion_model.h5"
LABEL_ENCODER_PATH = "label_encoder.npy"
MAX_LEN = 174
CHUNK = 22050  # 1 second chunks
SR = 22050

# -----------------------------
# LOAD MODEL AND LABELS
# -----------------------------
model = load_model(MODEL_PATH)
labels = np.load(LABEL_ENCODER_PATH)

# -----------------------------
# REAL-TIME AUDIO STREAM
# -----------------------------
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SR,
                input=True,
                frames_per_buffer=CHUNK)

plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(labels, np.zeros(len(labels)))
ax.set_ylim(0,1)
ax.set_ylabel("Probability")
title = ax.set_title("Listening...")

print("Listening... Press Ctrl+C to stop")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)

        # Extract MFCC and pad
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40).T
        mfcc_padded = pad_sequences([mfcc], maxlen=MAX_LEN, dtype="float32", padding="post", truncating="post")

        # Predict probabilities
        probs = model.predict(mfcc_padded, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_emotion = labels[pred_idx]

        # Update bar graph
        for bar, prob in zip(bars, probs):
            bar.set_height(prob)
        title.set_text(f"Predicted Emotion: {pred_emotion}")
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
