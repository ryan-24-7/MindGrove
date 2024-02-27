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

from joblib import dump, load

import seaborn as sns

def train_model():
    df = pd.read_csv("./Backend/emotion_dataset_raw.csv")
    dir(nfx)
    df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
    # Stopwords
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

    Xfeatures = df['Clean_Text']
    ylabels = df['Emotion']

    #  Split Data
    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)
    # Build Pipeline
    
    # LogisticRegression Pipeline
    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
    # Train and Fit Data
    pipe_lr.fit(x_train,y_train)
    pipe_lr.score(x_test,y_test)
    dump(pipe_lr, 'emotion_classifier.joblib')

if __name__ == '__main__':
    train_model()