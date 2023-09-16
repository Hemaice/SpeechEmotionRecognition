import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
from keras.models import load_model

# Load the trained model
model = load_model('audio.h5')  # Replace 'audio.h5' with the path to your model file

# Function to extract MFCC features from audio data
def extract_mfcc(audio_data):
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40).T, axis=0)
    return mfcc

# Streamlit UI
st.title("Speech Emotion Recognition")

# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file (in WAV format)", type=["wav"])

if uploaded_file is not None:
    # Load and preprocess the uploaded audio file
    audio_data, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
    mfcc_features = extract_mfcc(audio_data)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Reshape for model input
    
    # Make a prediction
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasantSurprise', 'sad']
    prediction = model.predict(mfcc_features)
    predicted_emotion = emotions[np.argmax(prediction)]

    # Display the audio and prediction
    st.audio(uploaded_file, format='audio/wav')
    st.write(f"Predicted Emotion: {predicted_emotion}")
