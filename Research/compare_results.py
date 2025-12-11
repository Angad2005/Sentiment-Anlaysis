import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import os

def run_comparison():
    vader_path = 'Research/vader_project/vader_results.csv'
    roberta_path = 'Research/roberta_project/roberta_results.csv'

    if not os.path.exists(vader_path) or not os.path.exists(roberta_path):
        print("Results files not found. Please run individual project scripts first.")
        return

    print("Loading results...")
    df_vader = pd.read_csv(vader_path)
    df_roberta = pd.read_csv(roberta_path)

    # Merge on tweet_id to compare the same tweets
    # VADER has all, RoBERTa has subset
    df_merged = pd.merge(df_roberta, df_vader[['tweet_id', 'vader_sentiment']], on='tweet_id', how='inner')
    
    print(f"Compared {len(df_merged)} items.")

    y_true = df_merged['airline_sentiment']
    y_vader = df_merged['vader_sentiment']
    y_roberta = df_merged['roberta_sentiment']

    # Metrics
    metrics = {
        'Model': ['VADER', 'RoBERTa'],
        'Accuracy': [accuracy_score(y_true, y_vader), accuracy_score(y_true, y_roberta)],
        'F1 Weighted': [f1_score(y_true, y_vader, average='weighted'), f1_score(y_true, y_roberta, average='weighted')]
    }
    df_metrics = pd.DataFrame(metrics)
    print("\nPerformance Comparison:")
    print(df_metrics)

    # Detailed Reports
    print("\n--- VADER Classification Report ---")
    print(classification_report(y_true, y_vader))
    print("\n--- RoBERTa Classification Report ---")
    print(classification_report(y_true, y_roberta))

    # Visualization: Accuracy Comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y='Accuracy', data=df_metrics)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.savefig('Research/model_accuracy_comparison.png')
    print("Saved Research/model_accuracy_comparison.png")

    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_vader = confusion_matrix(y_true, y_vader, labels=['negative', 'neutral', 'positive'])
    sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    axes[0].set_title('VADER Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_roberta = confusion_matrix(y_true, y_roberta, labels=['negative', 'neutral', 'positive'])
    sns.heatmap(cm_roberta, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    axes[1].set_title('RoBERTa Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('Research/confusion_matrices.png')
    print("Saved Research/confusion_matrices.png")

    # Example Disagreements
    # Merge for easy comparison
    df_comp = pd.DataFrame({
        'Text': df_merged['text'],
        'Ground_Truth': y_true,
        'VADER': y_vader,
        'RoBERTa': y_roberta
    })

    print("\n--- Comparative Examples (RoBERTa Correct vs VADER Wrong) ---")
    examples = []
    
    for sentiment in ['negative', 'positive', 'neutral']:
        # Filter: Ground Truth is X, RoBERTa is X, VADER is NOT X
        subset = df_comp[
            (df_comp['Ground_Truth'] == sentiment) & 
            (df_comp['RoBERTa'] == sentiment) & 
            (df_comp['VADER'] != sentiment)
        ]
        
        if not subset.empty:
            # Take the first one
            row = subset.iloc[0]
            examples.append(row)
            print(f"\nCategory: {sentiment.upper()}")
            print(f"Text: {row['Text']}")
            print(f"VADER: {row['VADER']}")
        else:
            print(f"\nCategory: {sentiment.upper()} - No examples found where RoBERTa > VADER in this sample.")

    if examples:
        df_examples = pd.DataFrame(examples)
        print(df_examples[['Text', 'Ground_Truth', 'VADER', 'RoBERTa']].to_markdown(index=False))
        df_examples.to_csv('Research/comparison_examples_diverse.csv', index=False)
        print("Saved Research/comparison_examples_diverse.csv")

if __name__ == "__main__":
    run_comparison()
