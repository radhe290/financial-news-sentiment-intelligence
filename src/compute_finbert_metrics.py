"""
Compute REAL FinBERT metrics from outputs/scored_headlines.csv

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent

scored = pd.read_csv(BASE_DIR / "outputs" / "scored_headlines.csv")

# Recreate the SAME split used in classical_models.py and finbert_model.py
y = scored["Sentiment"].values
_, test_idx = train_test_split(
    range(len(scored)), test_size=0.2, random_state=42, stratify=y
)

test = scored.iloc[test_idx]
y_true = test["Sentiment"]
y_pred = test["pred_label"]

acc = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true, y_pred, target_names=["negative", "neutral", "positive"], output_dict=True
)

print(f"FinBERT test-set accuracy: {acc*100:.2f}%")
print()
for label in ["negative", "neutral", "positive"]:
    print(f"  {label:10s} F1: {report[label]['f1-score']:.2f}")