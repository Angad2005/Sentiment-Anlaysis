import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def run_roberta_analysis():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '../Tweets.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    # Use a subset for testing due to CPU constraints
    print("Sampling 100 rows for RoBERTa analysis...")
    df = df.sample(n=100, random_state=42)
    print(f"Data loaded. Shape: {df.shape}")

    print("Loading RoBERTa model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    labels_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def get_roberta_sentiment(text):
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        top_label_id = ranking[0]
        return labels_map[top_label_id]

    print("Running RoBERTa analysis (this may take a while)...")
    # Batch processing is better, but let's stick to apply with progress bar for simplicity in this environment
    tqdm.pandas()
    df['roberta_sentiment'] = df['text'].progress_apply(get_roberta_sentiment)

    output_path = os.path.join(os.path.dirname(__file__), 'roberta_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(x='roberta_sentiment', data=df, order=['negative', 'neutral', 'positive'])
    plt.title('RoBERTa Sentiment Distribution')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'roberta_distribution.png'))
    print("Graph saved to roberta_distribution.png")

    if 'airline_sentiment' in df.columns:
        accuracy = (df['roberta_sentiment'] == df['airline_sentiment']).mean()
        print(f"RoBERTa Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_roberta_analysis()
