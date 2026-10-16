"""
Step 5: Stock Price Data
Fetches historical OHLCV (Open/High/Low/Close/Volume) data using yfinance
and computes daily returns for correlation with sentiment scores.

"""

import time
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# The 5 major tickers it track
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def fetch_stock_data(
    tickers: list  = DEFAULT_TICKERS,
    period: str    = "2y",          # 2 years of daily data
    interval: str  = "1d",          # daily bars
) -> pd.DataFrame:
    """
    Download OHLCV data for multiple tickers.

    Args:
        tickers:  list of ticker symbols
        period:   yfinance period string ('1y', '2y', '5y', etc.)
        interval: bar size ('1d' = daily, '1wk' = weekly)

    Returns:
        DataFrame with columns: [ticker, date, open, high, low, close,
                                  volume, daily_return, next_day_return]
    """
    print(f"Fetching stock data for: {', '.join(tickers)} ({period}, {interval} bars)")

    all_frames = []

    for ticker in tickers:
        print(f"  Downloading {ticker}...", end=" ")
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)

            if df.empty:
                print("✗ No data returned")
                continue

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            df.index.name = "date"
            df             = df.reset_index()
            df["ticker"]   = ticker

            # Daily return: % change from previous close
            # pct_change() computes (current - previous) / previous
            df["daily_return"] = df["close"].pct_change() * 100   # in percent

            # Next-day return: this is what I try to PREDICT from today's news.
            # shift(-1) moves values one row UP - so row[t] gets row[t+1]'s return.
            df["next_day_return"] = df["daily_return"].shift(-1)

            # Log return (alternative metric, useful for statistical tests)
            df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100

            # Direction labels (up / down / flat)
            df["direction"] = df["next_day_return"].apply(
                lambda r: "up"   if r > 0.5
                     else "down" if r < -0.5
                     else "flat"
            )

            all_frames.append(df)
            print(f"✓ {len(df)} rows")

        except Exception as e:
            print(f"✗ Error: {e}")

        time.sleep(0.3)   # polite rate limiting

    if not all_frames:
        raise RuntimeError("No stock data was downloaded. Check your internet connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # Drop last row per ticker
    combined = combined.dropna(subset=["next_day_return"])

    print(f"\n  Total rows: {len(combined):,} across {combined['ticker'].nunique()} tickers")
    return combined


def get_ticker_summary(df: pd.DataFrame) -> None:
    """Print summary stats per ticker"""
    print("\nSTOCK DATA SUMMARY")
    print(f"{'Ticker':<8} {'Rows':>6} {'Start':>12} {'End':>12} {'Avg Return':>12} {'Volatility':>12}")
    for ticker, grp in df.groupby("ticker"):
        start    = grp["date"].min().date()
        end      = grp["date"].max().date()
        avg_ret  = grp["daily_return"].mean()
        vol      = grp["daily_return"].std()   # std dev of daily returns = volatility
        print(f"{ticker:<8} {len(grp):>6} {str(start):>12} {str(end):>12} {avg_ret:>11.3f}% {vol:>11.3f}%")


def create_sentiment_price_dataset(
    sentiment_scores: pd.DataFrame,
    stock_df:         pd.DataFrame,
    ticker:           str,
) -> pd.DataFrame:
    """
    Merge daily aggregate sentiment scores with next-day stock returns.


    Args:
        sentiment_scores: DataFrame with columns [date, mean_score, positive_pct, ...]
                          (aggregated from multiple headlines per day)
        stock_df:         full stock price DataFrame from fetch_stock_data()
        ticker:           which ticker to merge with

    Returns:
        Merged DataFrame ready for correlation analysis.
    """
    stock = stock_df[stock_df["ticker"] == ticker][
        ["date", "close", "daily_return", "next_day_return", "direction"]
    ].copy()

    stock["date"] = pd.to_datetime(stock["date"]).dt.date

    sentiment_scores = sentiment_scores.copy()
    sentiment_scores["date"] = pd.to_datetime(sentiment_scores["date"]).dt.date

    # Inner join - only days where I have both sentiment and price data
    merged = pd.merge(sentiment_scores, stock, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    print(f"  Merged {len(merged)} trading days with sentiment + price data")
    return merged



def get_stock_data(tickers=DEFAULT_TICKERS, period="2y") -> pd.DataFrame:
    """
    loader: tries live yfinance.
    """
    return fetch_stock_data(tickers=tickers, period=period)


if __name__ == "__main__":
    stock_df = get_stock_data(tickers=["AAPL", "MSFT", "TSLA"])
    get_ticker_summary(stock_df)

    out_path = OUTPUTS_DIR / "stock_data.csv"
    stock_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    print("\nSample rows (AAPL, last 5):")
    aapl = stock_df[stock_df["ticker"] == "AAPL"][
        ["date", "close", "daily_return", "next_day_return", "direction"]
    ].tail(5)
    print(aapl.to_string(index=False))
