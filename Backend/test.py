from pydub import AudioSegment
import speech_recognition as sr
# EDA
import pandas as pd
import numpy as np

# Load Data Viz Pkgs
import seaborn as sns

# Load Text Cleaning Pkgs
import neattext.functions as nfx

from io import BytesIO
import matplotlib.pyplot as plt

# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline

import seaborn as sns

from joblib import load, dump


def generate_plot(audio_file):
    # Replace 'your_file.mp3' with the path to your mp3 file
    audio = AudioSegment.from_file(audio_file, "m4a")
    audio.export("converted.wav", format="wav")

    # Initialize the recognizer
    r = sr.Recognizer()

    with sr.AudioFile("converted.wav") as source:
        # Listen for the data (load audio to memory)
        audio_data = r.record(source)
        # Recognize (convert from speech to text)
        try:
            text = r.recognize_google(audio_data)
            print(text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    
    pipe_lr = load('./emotion_classifier.joblib')
    pipe_lr.predict([text])
    # Prediction Prob
    probabilities = pipe_lr.predict_proba([text])[0]

    emotion_classes = pipe_lr.classes_
    # Normalize the probabilities to sum to 1
    normalized_probabilities = probabilities / np.sum(probabilities)

    # Convert to percentages
    percentages = normalized_probabilities * 100

    # Convert to a Series for easy plotting
    emotion_series = pd.Series(percentages, index=emotion_classes)

    # Plot
    plt.figure(figsize=(14, 6))
    sns.barplot(x=emotion_series.index, y=emotion_series.values, palette='viridis')
    plt.title('Probabilities of Emotions for the Sentence')
    plt.ylabel('Percentage')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Go to the beginning of the BytesIO buffer
    plt.close()
    return buf, text
