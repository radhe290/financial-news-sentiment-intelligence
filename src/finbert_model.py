"""
Step 4: FinBERT Sentiment Model

"""

import time
import numpy as np
from pathlib import Path
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from sklearn.metrics import classification_report, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_dataset, get_features_and_labels
from sklearn.model_selection import train_test_split

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# HuggingFace model id
FINBERT_MODEL_ID = "ProsusAI/finbert"

# FinBERT label mapping
FINBERT_LABELS = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
}


class FinBERTSentiment:
    """
    Wrapper around HuggingFace ProsusAI/finbert for easy inference.

    Usage:
        model = FinBERTSentiment()
        result = model.predict("Operating profit rose 14% on strong demand")
        # -> {'label': 'positive', 'score': 0.9823, 'all_scores': {...}}
    """

    def __init__(self, model_id: str = FINBERT_MODEL_ID, device: str = None):
        """
        Load the FinBERT model and tokenizer.

        Args:
            model_id: HuggingFace model identifier
            device:   'cuda' for GPU, 'cpu' for CPU, None = auto-detect
        """
        # Auto-detect GPU availability
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device   = device
        self.model_id = model_id

        print(f"  Loading {model_id} on {device}...")
        t0 = time.time()

        # Load tokenizer: converts text -> token IDs that BERT understands
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the classification model (BERT + linear head for 3 classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()   # disable dropout for inference

        # HuggingFace pipeline wraps tokenize + forward pass + decode into one call
        self.pipe = pipeline(
            "text-classification",
            model     = self.model,
            tokenizer = self.tokenizer,
            device    = 0 if device == "cuda" else -1,
            top_k     = None,  # return scores for ALL 3 classes
        )

        elapsed = time.time() - t0
        print(f"  ✓ Model loaded in {elapsed:.1f}s")
        print(f"  Parameters: ~{sum(p.numel() for p in self.model.parameters())/1e6:.0f}M")

    def predict(self, text: str) -> dict:
        """
        Classify a single sentence.

        Returns:
            {
              'label': 'positive',
              'score': 0.9823,         # confidence for predicted label
              'all_scores': {           # softmax probabilities for all 3
                'positive': 0.9823,
                'negative': 0.0102,
                'neutral':  0.0075,
              }
            }
        """
        # Truncate long texts to BERT's max sequence length (512 tokens)
        text = str(text)[:512]
        results = self.pipe(text)[0]      # list of {label, score} dicts

        # Build a friendly all_scores dict
        all_scores = {r["label"]: r["score"] for r in results}

        # Winning label is the one with highest score
        best = max(results, key=lambda r: r["score"])

        return {
            "label":      best["label"],
            "score":      best["score"],
            "all_scores": all_scores,
            "score_type": "softmax probability",
        }

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> List[dict]:
        """
        Classify a list of sentences efficiently using batching.

        """
        results = []
        n = len(texts)
        texts = [str(t)[:512] for t in texts]

        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self.pipe(batch)    # list of lists

            for sentence_results in batch_results:
                all_scores = {r["label"]: r["score"] for r in sentence_results}
                best = max(sentence_results, key=lambda r: r["score"])
                results.append({
                    "label":      best["label"],
                    "score":      best["score"],
                    "all_scores": all_scores,
                    "score_type": "softmax probability",
                })

            if show_progress:
                done = min(i + batch_size, n)
                pct  = done / n * 100
                bar  = "█" * (done * 30 // n)
                print(f"\r  Progress: [{bar:<30}] {done:4d}/{n} ({pct:.0f}%)", end="")

        if show_progress:
            print()

        return results


def evaluate_finbert(model: FinBERTSentiment, X_test, y_test) -> dict:
    """
    Run FinBERT on the test set and print evaluation metrics.
    """
    print("\n  Running FinBERT inference on test set...")
    t0 = time.time()

    batch_results = model.predict_batch(X_test.tolist(), batch_size=16)
    elapsed = time.time() - t0

    y_pred = np.array([r["label"] for r in batch_results])
    acc    = accuracy_score(y_test, y_pred)

    print(f"  MODEL: FinBERT (ProsusAI/finbert)")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Inference  : {elapsed:.1f}s for {len(X_test)} samples "
          f"({len(X_test)/elapsed:.1f} samples/sec)")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["negative", "neutral", "positive"],
    ))

    return {"name": "FinBERT", "accuracy": acc, "predictions": y_pred}


def show_example_predictions(model: FinBERTSentiment) -> None:
    """
    Show FinBERT's reasoning on sample sentences.

    """
    examples = [
        ("Operating profit rose 14% on strong demand for products",   "positive"),
        ("$ESI on lows, down $1.50 BK a real possibility",            "negative"),
        ("The company operates stores across 12 European countries",   "neutral"),
        ("Analysts upgraded the stock citing bullish revenue growth",  "positive"),
        ("The firm announced restructuring, cutting 2,000 jobs",      "negative"),
        ("Net sales increased despite challenging market conditions",  "positive"),
    ]

    print("\nFINBERT EXAMPLE PREDICTIONS")
    for text, true_label in examples:
        result = model.predict(text)
        pred   = result["label"]
        conf   = result["score"]
        scores = result["all_scores"]
        match  = "✓" if pred == true_label else "✗"
        print(f"\n{match} Text   : {text}")
        print(f"  True   : {true_label}")
        print(f"  Pred   : {pred}  ({conf*100:.1f}% confident)")
        # Show the full probability distribution
        score_bar = "  Scores: " + " | ".join(
            f"{k}: {v*100:4.1f}%" for k, v in sorted(scores.items())
        )
        print(score_bar)


class FinBERTFallback:
    """
    Offline fallback used when ProsusAI/finbert cannot be downloaded.

    """
    import pickle as _pickle

    def __init__(self):
        import pickle
        from pathlib import Path
        from preprocessor import FinancialTextPreprocessor

        model_path = Path(__file__).resolve().parent.parent / "models" / "logisticregression.pkl"
        with open(model_path, "rb") as f:
            self._pipeline = pickle.load(f)

        self._preprocessor = FinancialTextPreprocessor()
        print("    Using classical LogReg as offline stand-in for FinBERT.")

    def predict(self, text: str) -> dict:
        cleaned = self._preprocessor.clean(str(text))
        proba   = self._pipeline.predict_proba([cleaned])[0]
        classes = self._pipeline.classes_          # ['negative','neutral','positive']
        all_scores = dict(zip(classes, proba))
        best = max(all_scores, key=all_scores.get)
        return {
            "label": best,
            "score": all_scores[best],
            "all_scores": all_scores,
            "score_type": "model probability estimate (not calibrated)",
        }

    def predict_batch(self, texts, batch_size=16, show_progress=True):
        results = []
        for i, text in enumerate(texts):
            results.append(self.predict(text))
            if show_progress and (i + 1) % 100 == 0:
                pct = (i + 1) / len(texts) * 100
                bar = "█" * ((i + 1) * 30 // len(texts))
                print(f"\r  Progress: [{bar:<30}] {i+1:4d}/{len(texts)} ({pct:.0f}%)", end="")
        if show_progress:
            print()
        return results


def get_sentiment_model():
    """
    function: returns real FinBERT if available, fallback otherwise.

    """
    try:
        model = FinBERTSentiment()
        return model
    except Exception as e:
        print(f"    Could not load ProsusAI/finbert: {type(e).__name__}: {e}")
        print(f"    Falling back to classical LogReg - predictions will NOT be FinBERT.")
        return FinBERTFallback()


def run_finbert_evaluation():
    """Full FinBERT (or fallback) evaluation pipeline."""
    print("FINBERT EVALUATION")

    df = load_dataset()
    X_raw, y = get_features_and_labels(df)
    X_raw = np.array(X_raw)
    y     = np.array(y)

    _, X_test, _, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_sentiment_model()           # real FinBERT or fallback
    show_example_predictions(model)
    result = evaluate_finbert(model, X_test, y_test)
    return model, result


if __name__ == "__main__":
    model, result = run_finbert_evaluation()
    print(f"\n  Model accuracy: {result['accuracy']*100:.2f}%")