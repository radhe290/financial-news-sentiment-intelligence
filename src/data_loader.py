"""
Step 1: Data Loader
Loads and explores the Financial PhraseBank dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "financial_phrasebank.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Strip whitespace from both columns
    df["Sentence"]  = df["Sentence"].str.strip()
    df["Sentiment"] = df["Sentiment"].str.strip().str.lower()

    # Add a helper column for text length
    df["text_length"] = df["Sentence"].str.split().str.len()

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    print("DATASET OVERVIEW")
    print(f"Total samples     : {len(df):,}")
    print(f"Columns           : {df.columns.tolist()}")
    print()

    #  Class distribution
    # WHY THIS MATTERS: imbalanced classes hurt model performance.
    # If 90% of data is 'neutral', a model can cheat by always predicting neutral.
    print("CLASS DISTRIBUTION")
    counts = df["Sentiment"].value_counts()
    for label, count in counts.items():
        bar = "█" * (count // 100)
        pct = count / len(df) * 100
        print(f"  {label:10s}: {count:5,} ({pct:5.1f}%)  {bar}")
    print()

    #  Text length stats
    print("TEXT LENGTH (words)")
    stats = df.groupby("Sentiment")["text_length"].describe()[["mean", "min", "max"]]
    print(stats.round(1).to_string())
    print()

    #  Sample sentences from each class
    print("SAMPLE SENTENCES (1 per class)")

    for label in ["positive", "negative", "neutral"]:
        sample = df[df["Sentiment"] == label]["Sentence"].sample(1, random_state=42).values[0]
        print(f"\n  [{label.upper()}]")
        print(f"  {sample[:100]}{'...' if len(sample) > 100 else ''}")
    print()

    #  Class imbalance warning 
    majority_pct = counts.max() / len(df) * 100
    if majority_pct > 60:
        print(f"⚠  CLASS IMBALANCE: '{counts.idxmax()}' = {majority_pct:.0f}% of data.")
        print("   Consider class-weighted training or oversampling (SMOTE).")



def get_features_and_labels(df: pd.DataFrame):
    X = df["Sentence"].values   # array of strings
    y = df["Sentiment"].values  # array of label strings
    return X, y


if __name__ == "__main__":
    df = load_dataset()
    describe_dataset(df)
    X, y = get_features_and_labels(df)
    print(f"\nX shape: {X.shape}  (one sentence per row)")
    print(f"y shape: {y.shape}  (one label per row)")
    print(f"Unique labels: {np.unique(y).tolist()}")
