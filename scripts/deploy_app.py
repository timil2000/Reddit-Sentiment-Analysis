from flask import Flask, request, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Set the static folder inside 'scripts/'
app = Flask(__name__, static_folder="C:\Intellij Projects\SentimentAnalysis\scripts\static")

# Load the trained model
model_path = "models/lstm_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None


# Load processed data
def load_sentiment_data():
    file_path = "data/cleaned_posts.csv"
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

# Store previous results
history_path = "data/sentiment_history.csv"
if not os.path.exists(history_path):
    pd.DataFrame(columns=["topic", "sentiment"]).to_csv(history_path, index=False)


# Preprocess user input
def preprocess_text(user_text, vectorizer):
    user_text_cleaned = " ".join(user_text.lower().split())  # Simple cleanup
    user_text_vectorized = vectorizer.transform([user_text_cleaned])
    return user_text_vectorized.toarray()

# Predict sentiment using the trained model
def classify_sentiment(text, vectorizer):
    if model is None:
        return "unknown"

    text_vectorized = preprocess_text(text, vectorizer)
    text_vectorized = pad_sequences(text_vectorized, maxlen=5000)  # Match LSTM input size
    prediction = model.predict(text_vectorized)[0][0]

    if prediction > 0.6:
        return "Positive"
    elif prediction < 0.4:
        return "Negative"
    else:
        return "Neutral"


def generate_sentiment_chart(df, topic, vectorizer):
    # Convert user input to lowercase & remove non-alphabetic characters for better matching
    topic_cleaned = re.sub(r"[^a-zA-Z\s]", "", topic.lower())

    # Ensure cleaned text is also processed similarly
    df["cleaned_text"] = df["cleaned_text"].astype(str).apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))

    # Filter posts containing the topic
    filtered_df = df[df["cleaned_text"].str.contains(topic_cleaned, case=False, na=False)]

    if filtered_df.empty:
        print(f" No matching posts found for '{topic_cleaned}'")
        return None

    # Predict sentiment for filtered text
    filtered_df["sentiment"] = filtered_df["cleaned_text"].apply(lambda x: classify_sentiment(x, vectorizer))

    # Generate sentiment distribution
    sentiment_counts = filtered_df["sentiment"].value_counts()

    # Ensure 'static/' folder exists inside 'scripts/'
    static_path = os.path.join("scripts", "static")
    os.makedirs(static_path, exist_ok=True)

    # Create bar chart
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Posts")
    plt.title(f"Sentiment Analysis for '{topic_cleaned}'")
    plt.xticks(rotation=0)

    # Save chart
    chart_path = os.path.join(static_path, "sentiment_chart.png")
    plt.savefig(chart_path)
    plt.close()

    return "static/sentiment_chart.png"  # Relative path for Flask


@app.route("/", methods=["GET", "POST"])
def home():
    df = load_sentiment_data()

    if df is None or df.empty:
        return "🚨 No data available. Run preprocessing first."

    image_path = None
    topic = ""

    if request.method == "POST":
        topic = request.form["topic"]

        # Ensure vectorizer is trained on entire dataset first
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(df["cleaned_text"].astype(str))

        image_path = generate_sentiment_chart(df, topic, vectorizer)

        # Store result in history
        new_entry = pd.DataFrame({"topic": [topic], "sentiment": ["Processing"]})
        new_entry.to_csv(history_path, mode='a', header=False, index=False)

    return render_template("index.html", image_path=image_path, topic=topic)


@app.route("/history", methods=["GET"])
def history():
    history_df = pd.read_csv(history_path)
    return jsonify(history_df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)