**Speech Emotion Recognition (SER) with BiLSTM**

This project uses BiLSTM networks to classify emotions from speech using the RAVDESS dataset. It supports real-time emotion detection from the microphone and displays emotion probabilities dynamically as a bar graph.

Features

Trains a BiLSTM model on MFCC sequences from RAVDESS dataset.

Handles 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Real-time microphone emotion detection.

Live bar graph showing probabilities of all emotions.

Displays predicted emotion label in real-time.

Robust sequence padding and normalization to prevent mispredictions.

**Project Structure:**

DEEPELEARNING/

│

├── .venv                 ← virtual environment

├── RAVDESS/              ← dataset folder

├── emotion_model.h5      ← trained BiLSTM model

├── label_encoder.npy     ← label classes

├── main.py               ← training script

├── predict_mic.py        ← real-time mic prediction script

├── requirements.txt

└── README.md

Setup Instructions

1️⃣ Clone the repository

git clone <https://github.com/lillysanjana/speech_emotion.git>

cd DEEPELEARNING

2️⃣ Install dependencies

pip install -r requirements.txt


If pyaudio fails to install on Windows:

pip install pipwin

pipwin install pyaudio

3️⃣ Download RAVDESS dataset

Kaggle link: RAVDESS Emotional Speech Audio

Unzip inside the project folder as RAVDESS/

Training the Model

python main.py


Extracts MFCC features from all audio files.

Pads sequences to fixed length for LSTM input.

Trains a BiLSTM network.

Saves:

emotion_model.h5 → trained model

label_encoder.npy → label encoder for mapping classes

Real-Time Emotion Prediction

python predict_mic.py


Opens microphone stream.

Predicts emotion probabilities in real-time.

Displays bar graph and predicted emotion label.

Stop the script: Press Ctrl+C in terminal.

**Emotions Recognized:**

Code	Emotion

01	neutral

02	calm

03	happy

04	sad

05	angry

06	fearful

07	disgust

08	surprised

**Requirements:**

Python 3.8+

TensorFlow

librosa

numpy

matplotlib

pyaudio

scikit-learn
