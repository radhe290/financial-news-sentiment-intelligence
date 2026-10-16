"""
Step 6: Sentiment-Price Correlation Analysis

The core research question: does news sentiment today predict
stock direction tomorrow?

"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_dataset, get_features_and_labels
from finbert_model import get_sentiment_model
from stock_data import get_stock_data,DEFAULT_TICKERS

BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def score_all_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run sentiment model on every headline in the dataset.
    Adds 'pred_label' and 'confidence' columns.
    """
    print("Scoring all headlines")
    model = get_sentiment_model()

    sentences = df["Sentence"].tolist()
    results   = model.predict_batch(sentences, batch_size=32, show_progress=True)

    df = df.copy()
    df["pred_label"]  = [r["label"]      for r in results]
    df["confidence"]  = [r["score"]      for r in results]
    df["pos_score"]   = [r["all_scores"].get("positive", 0) for r in results]
    df["neg_score"]   = [r["all_scores"].get("negative", 0) for r in results]
    df["neu_score"]   = [r["all_scores"].get("neutral",  0) for r in results]
    # Composite score: +1 = fully positive, -1 = fully negative
    df["sentiment_score"] = df["pos_score"] - df["neg_score"]

    return df


def aggregate_daily_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-headline sentiment into a single daily signal.
    """
    n = len(scored_df)

    # Spread headlines across ~503 trading days (2 years)
    n_days  = 503
    indices = np.linspace(0, n - 1, n_days, dtype=int)

    daily_rows = []
    chunk_size = n // n_days

    for i in range(n_days):
        start = i * chunk_size
        end   = start + chunk_size if i < n_days - 1 else n
        chunk = scored_df.iloc[start:end]

        pos_pct = (chunk["pred_label"] == "positive").mean() * 100
        neg_pct = (chunk["pred_label"] == "negative").mean() * 100
        neu_pct = (chunk["pred_label"] == "neutral").mean()  * 100

        daily_rows.append({
            "day_idx":        i,
            "headline_count": len(chunk),
            "mean_score":     chunk["sentiment_score"].mean(),
            "positive_pct":   pos_pct,
            "negative_pct":   neg_pct,
            "neutral_pct":    neu_pct,
        })

    return pd.DataFrame(daily_rows)


def compute_correlation(daily_sentiment: pd.DataFrame, stock_df: pd.DataFrame, ticker: str) -> dict:
    """
    Correlate daily sentiment score with next-day stock return.

    Returns a dict of correlation statistics.
    """
    price_df = stock_df[stock_df["ticker"] == ticker].reset_index(drop=True)

    # Align by index (both should have ~503 rows for 2 trading years)
    n = min(len(daily_sentiment), len(price_df))
    sentiment_scores = daily_sentiment["mean_score"].values[:n]
    next_day_returns = price_df["next_day_return"].values[:n]

    # Remove any NaN pairs
    mask   = ~(np.isnan(sentiment_scores) | np.isnan(next_day_returns))
    scores = sentiment_scores[mask]
    returns = next_day_returns[mask]

    # Pearson: assumes linear relationship and it is sensitive to outliers
    pearson_r, pearson_p  = stats.pearsonr(scores, returns)

    # Spearman: rank-based and robust to outliers - preferred for finance
    spearman_r, spearman_p = stats.spearmanr(scores, returns)

    # Direction accuracy: does positive sentiment -> positive return next day?
    pred_direction = np.sign(scores)
    true_direction = np.sign(returns)
    direction_acc  = (pred_direction == true_direction).mean()

    results = {
        "ticker":      ticker,
        "n_days":      int(mask.sum()),
        "pearson_r":   float(pearson_r),
        "pearson_p":   float(pearson_p),
        "spearman_r":  float(spearman_r),
        "spearman_p":  float(spearman_p),
        "direction_acc": float(direction_acc),
        "scores":      scores,
        "returns":     returns,
    }

    return results


def print_correlation_report(results: dict) -> None:
    """Print a readable correlation report."""
    print(f"  CORRELATION REPORT: {results['ticker']}")
    print(f"  Days analyzed     : {results['n_days']}")
    print()
    print(f"  Pearson  r = {results['pearson_r']:+.4f}  (p = {results['pearson_p']:.4f})")
    print(f"  Spearman ρ = {results['spearman_r']:+.4f}  (p = {results['spearman_p']:.4f})")
    print()
    print(f"  Direction accuracy: {results['direction_acc']*100:.1f}%")
    print(f"  (Random baseline = 50.0%)")
    print()

    # Interpret
    r = abs(results['spearman_r'])
    strength = ("negligible" if r < 0.1 else
                "weak"       if r < 0.3 else
                "moderate"   if r < 0.5 else
                "strong")

    sig = results['spearman_p'] < 0.05
    print(f"  Interpretation: {strength} {'significant' if sig else 'non-significant'} correlation")
    if not sig:
        print(f"  Note: p > 0.05 means this could be random chance.")
        print(f"  With real financial news data, correlations are typically")
        print(f"  in the 0.1–0.25 range - small but profitable at scale.")


def run_full_correlation_analysis():
    """
    Full pipeline: score all headlines, build daily aggregates,
    compute correlations per ticker, save results.
    """
    print("SENTIMENT-PRICE CORRELATION ANALYSIS")

    # Score all headlines
    df = load_dataset()
    scored_df = score_all_headlines(df)

    # Aggregate to daily level
    print("\nAggregating to daily sentiment signals...")
    daily_sentiment = aggregate_daily_sentiment(scored_df)

    # Fetch stock data
    print("\nLoading stock price data...")
    stock_df = get_stock_data(tickers=DEFAULT_TICKERS)

    # Compute correlation per ticker
    all_results = {}
    for ticker in stock_df["ticker"].unique():
        result = compute_correlation(daily_sentiment, stock_df, ticker)
        all_results[ticker] = result
        print_correlation_report(result)

    # Save scored data for Streamlit
    scored_df.to_csv(OUTPUTS_DIR / "scored_headlines.csv", index=False)
    daily_sentiment.to_csv(OUTPUTS_DIR / "daily_sentiment.csv", index=False)
    stock_df.to_csv(OUTPUTS_DIR / "stock_data.csv", index=False)

    print(f"\n  Results saved to {OUTPUTS_DIR}")
    return all_results, daily_sentiment, stock_df


if __name__ == "__main__":
    results, daily_sentiment, stock_df = run_full_correlation_analysis()
