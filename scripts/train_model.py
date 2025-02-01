import numpy as np
import tensorflow as tf
from keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Embedding
import os

# Function to train LSTM model
def train_model():
    # Check if TF-IDF matrix exists
    if not os.path.exists("data/tfidf_matrix.npy"):
        raise FileNotFoundError(" TF-IDF data missing! Run preprocessing first.")

    data = np.load("data/tfidf_matrix.npy", allow_pickle=True)
    labels = np.random.randint(0, 2, size=(data.shape[0],))  # Fake labels for testing

    # Build LSTM model
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64),
        LSTM(100, return_sequences=True),
        LSTM(50),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(data, labels, epochs=5, batch_size=32)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")
    print("Model training complete. Model saved!")