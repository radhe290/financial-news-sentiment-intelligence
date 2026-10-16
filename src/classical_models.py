"""
Step 3: Classical ML Baseline Models

Three models to establish the performance baseline before FinBERT.

"""

import numpy as np
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_dataset, get_features_and_labels
from preprocessor import FinancialTextPreprocessor

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# TF-IDF configuration
# These hyperparameters control the vocabulary the model learns.
#   - max_features: only keep the N most informative words
#   - ngram_range=(1,2): include both single words AND word pairs
#       e.g. "profit growth" together is more signal than "profit" alone
#   - min_df=2: ignore words that appear fewer than 2 times (likely typos)
#   - sublinear_tf: dampens high term counts (log scaling - common in NLP)

TFIDF_CONFIG = dict(
    max_features  = 10_000,
    ngram_range   = (1, 2),
    min_df        = 2,
    sublinear_tf  = True,
)


def build_pipelines() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("clf",   LogisticRegression(
                C            = 1.0,          # regularization strength (higher = less regularized)
                max_iter     = 1000,
                class_weight = "balanced",   # compensate for class imbalance
                solver       = "lbfgs",
            )),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            # Wrap SVC in CalibratedClassifierCV to get probability estimates
            # (LinearSVC doesn't output probabilities by default)
            ("clf",   CalibratedClassifierCV(
                LinearSVC(
                    C            = 0.5,
                    max_iter     = 2000,
                    class_weight = "balanced",
                )
            )),
        ]),
    }


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"  MODEL: {name}")
    print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print()

    print(classification_report(y_test, y_pred,
                                target_names=["negative", "neutral", "positive"]))

    # Confusion matrix
    print("  Confusion matrix (rows=actual, cols=predicted):")
    cm_labels = ["neg", "neu", "pos"]
    cm = confusion_matrix(y_test, y_pred,
                          labels=["negative", "neutral", "positive"])
    header = "        " + "  ".join(f"{l:>5}" for l in cm_labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {cm_labels[i]:>5}  " + "  ".join(f"{v:5}" for v in row))

    return {"name": name, "accuracy": acc, "predictions": y_pred}


def run_vader_baseline(X_test, y_test) -> dict:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
    sentiment tool that requires NO training. It uses a hand-crafted
    lexicon of ~7500 words with sentiment scores.
    """
    analyzer = SentimentIntensityAnalyzer()

    # VADER returns compound score: -1 (most negative) to +1 (most positive)
    def vader_label(text: str) -> str:
        score = analyzer.polarity_scores(str(text))["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    y_pred = np.array([vader_label(t) for t in X_test])
    acc    = accuracy_score(y_test, y_pred)

    print(f"  MODEL: VADER (rule-based, no training)")
    print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["negative", "neutral", "positive"]))

    return {"name": "VADER", "accuracy": acc, "predictions": y_pred}


def train_and_evaluate():
    """
    Full training and evaluation pipeline.
    """
    print("CLASSICAL ML BASELINE")

    # Load and preprocess data
    print("\n[1/4] Loading data ")
    df = load_dataset()
    X_raw, y = get_features_and_labels(df)
    print(f"      {len(X_raw):,} samples loaded")

    print("\n[2/4] Preprocessing text ")
    preprocessor = FinancialTextPreprocessor()
    X_raw = np.array(X_raw)          # ensure plain numpy array (not pandas Series)
    y     = np.array(y)
    X     = np.array(preprocessor.clean_batch(X_raw))
    print(f"      Done. Sample: '{X[0][:60]}'")

    
    print("\n[3/4] Splitting data (80% train / 20% test, stratified) ")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train and evaluate each model
    print("\n[4/4] Training and evaluating models \n")
    results = []

    # Classical models
    pipelines = build_pipelines()
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)

        # Cross-validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                    scoring="accuracy", n_jobs=-1)
        print(f"  {name}: CV accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        result = evaluate_model(name, pipeline, X_test, y_test)
        results.append(result)

        # Save the trained pipeline
        model_path = MODEL_DIR / f"{name.replace(' ', '_').lower()}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"   Model saved to {model_path.name}")

    # VADER (no training) 
    vader_result = run_vader_baseline(X_test_raw, y_test)
    results.append(vader_result)

    # Summary comparison
    print("RESULTS SUMMARY")
    print(f"{'Model':<30} {'Accuracy':>10}")
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        bar = "█" * int(r["accuracy"] * 30)
        print(f"  {r['name']:<28} {r['accuracy']*100:8.2f}%  {bar}")

    print("\n  NEXT STEP: FinBERT should achieve ~90%+ by understanding")
    print("  financial language context, not just individual word counts.")

    return results


if __name__ == "__main__":
    train_and_evaluate()
