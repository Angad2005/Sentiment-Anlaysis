import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def run_vader_analysis():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '../Tweets.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure download_data.py has run.")
        return

    print("Data loaded. Shape:", df.shape)

    # Initialize VADER
    sia = SentimentIntensityAnalyzer()

    # Function to get sentiment score
    def get_vader_score(text):
        return sia.polarity_scores(text)['compound']

    print("Running VADER analysis...")
    df['vader_score'] = df['text'].apply(get_vader_score)

    # Categorize
    def categorize(score):
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['vader_sentiment'] = df['vader_score'].apply(categorize)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'vader_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(x='vader_sentiment', data=df, order=['negative', 'neutral', 'positive'])
    plt.title('VADER Sentiment Distribution')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'vader_distribution.png'))
    print("Graph saved to vader_distribution.png")

    # Accuracy check (quick printing)
    if 'airline_sentiment' in df.columns:
        accuracy = (df['vader_sentiment'] == df['airline_sentiment']).mean()
        print(f"VADER Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_vader_analysis()
