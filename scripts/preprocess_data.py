import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    text = text.lower()
    text = word_tokenize(text)  # Tokenize words
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return " ".join(text)

# Function to process Reddit posts
def preprocess_data():
    input_path = "data/technology_posts.csv"
    output_path = "data/cleaned_posts.csv"
    tfidf_output_path = "data/tfidf_matrix.npy"

    # Ensure raw data exists
    if not os.path.exists(input_path):
        raise FileNotFoundError("❌ Raw Reddit data not found! Run data collection first.")

    # Load the data
    df = pd.read_csv(input_path)

    # Ensure there's valid text data
    if "title" not in df.columns or "selftext" not in df.columns:
        raise ValueError("🚨 Missing required text columns in dataset.")

    # Apply cleaning function to title and selftext
    df['cleaned_text'] = df['title'].apply(clean_text) + " " + df['selftext'].apply(clean_text)

    # Ensure there is data after cleaning
    if df['cleaned_text'].isnull().all():
        raise ValueError("🚨 No valid text found after preprocessing!")

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

    # Save the TF-IDF matrix
    os.makedirs("data", exist_ok=True)
    np.save(tfidf_output_path, tfidf_matrix.toarray())  # Save matrix as .npy file
    df.to_csv(output_path, index=False)  # Save cleaned data

    print("✅ Preprocessing complete! Cleaned text and TF-IDF matrix saved.")

# Run the script
if __name__ == "__main__":
    preprocess_data()
