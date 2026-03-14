import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# -----------------------------
# Load IMDB word index
# -----------------------------
word_index = imdb.get_word_index()

reverse_word_index = {value: key for key, value in word_index.items()}


# -----------------------------
# Build model architecture
# -----------------------------
def build_model():
    model = Sequential()
    # Corrected output_dim to 128 (matches your training notebook)
    # Added input_length=500 to initialize the weights immediately
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))
    model.add(SimpleRNN(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


# -----------------------------
# Load weights
# -----------------------------
@st.cache_resource
def load_my_model():

    model = build_model()

    model.load_weights("simple_rnn_model.h5")

    return model


model = load_my_model()


# -----------------------------
# Preprocess text
# -----------------------------
def preprocess_text(text):

    words = text.lower().split()

    encoded_review = [word_index.get(word, 2) + 3 for word in words]

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

    return padded_review


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):

    if user_input.strip() == "":
        st.warning("Please enter a review")

    else:

        processed = preprocess_text(user_input)

        prediction = model.predict(processed)

        score = float(prediction[0][0])

        sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        st.subheader(f"Sentiment: {sentiment}")

        st.write(f"Prediction Score: {score:.4f}")

        st.progress(score)