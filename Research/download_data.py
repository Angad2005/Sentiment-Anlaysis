import os
import requests
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/master/Tweets.csv"
OUTPUT_FILE = "Research/Tweets.csv"

def download_data():
    if os.path.exists(OUTPUT_FILE):
        print(f"File {OUTPUT_FILE} already exists.")
        return

    print(f"Downloading from {DATA_URL}...")
    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved to {OUTPUT_FILE}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
