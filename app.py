from scripts.collect_reddit_data import fetch_reddit_posts
from scripts.preprocess_data import preprocess_data
from scripts.train_model import train_model
from scripts.deploy_app import app
import nltk


if __name__ == "__main__":
    print("Starting Sentiment Analysis Pipeline...")

    # Download NLTK resources
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    # Phase 1: Collect Data
    fetch_reddit_posts("technology", limit=50)

    # Phase 2: Preprocess Data
    preprocess_data()

    # Phase 3: Train Model
    train_model()  # Now correctly calls the function

    # Phase 4: Run Web App
    app.run(debug=True)
