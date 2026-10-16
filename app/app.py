"""
Financial News Sentiment Dashboard

"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
sys.path.insert(0, str(BASE_DIR / "src"))

from data_loader import load_dataset
from preprocessor import FinancialTextPreprocessor
from stock_data import get_stock_data, DEFAULT_TICKERS
from correlation import aggregate_daily_sentiment, score_all_headlines

# Page config
st.set_page_config(
    page_title="Financial Sentiment Analyser",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    border-left: 4px solid;
    margin-bottom: 1rem;
  }
  .positive { border-color: #2ecc71; }
  .negative { border-color: #e74c3c; }
  .neutral  { border-color: #95a5a6; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("📈 Financial News Sentiment Analyser")
st.markdown(
    "Classify financial headlines with FinBERT · Correlate sentiment with stock returns"
)
st.divider()


# Caching helpers
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model():
    """Load once and cache for the session."""
    from finbert_model import get_sentiment_model

    return get_sentiment_model()


@st.cache_resource(show_spinner="Loading classifier...")
def load_classical_model():
    model_path = BASE_DIR / "models" / "logisticregression.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return load_dataset()


@st.cache_data(show_spinner="Loading scored headlines...")
def get_scored_headlines():
    """
    Load precomputed FinBERT scores from outputs/scored_headlines.csv.

    """
    precomputed = OUTPUTS_DIR / "scored_headlines.csv"
    if precomputed.exists():
        return pd.read_csv(precomputed)

    # Fallback: compute live
    st.warning(" outputs/scored_headlines.csv not found — scoring all headlines ")
    df = load_data()
    return score_all_headlines(df)


@st.cache_data(show_spinner="Loading stock data...")
def get_stocks(tickers):
    """
    Load precomputed stock data from outputs/stock_data.csv.

    """
    precomputed = OUTPUTS_DIR / "stock_data.csv"
    if precomputed.exists():
        df = pd.read_csv(precomputed, parse_dates=["date"])
        if set(tickers).issubset(set(df["ticker"].unique())):
            return df

    st.warning(
        " outputs/stock_data.csv missing or incomplete - fetching live "
        "from yfinance."
    )
    return get_stock_data(tickers=tickers)


@st.cache_data(show_spinner="Loading daily sentiment...")
def get_daily_sentiment():
    """
    Load precomputed daily sentiment aggregates.

    """
    precomputed = OUTPUTS_DIR / "daily_sentiment.csv"
    if precomputed.exists():
        return pd.read_csv(precomputed)

    scored = get_scored_headlines()
    return aggregate_daily_sentiment(scored)


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_ticker = st.selectbox("Stock ticker", DEFAULT_TICKERS, index=0)
    show_raw = st.checkbox("Show raw data tables", value=False)
    theme_dark = st.checkbox("Dark chart theme", value=False)
    chart_theme = "plotly_dark" if theme_dark else "plotly_white"

    st.divider()
    st.markdown("**About this project**")
    st.caption(
        "Uses ProsusAI/FinBERT fine-tuned on Financial PhraseBank "
        "(5,842 labelled sentences). Stock prices via yfinance."
    )
    st.caption("Built with: Python · Transformers · scikit-learn · Streamlit · Plotly")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔍 Live Classifier",
        "📊 Model Comparison",
        "📉 Sentiment vs. Price",
        "🗃️ Dataset Explorer",
        "📥 Batch Analyzer",
    ]
)


# TAB 1 - Live Classifier
with tab1:
    st.subheader("Classify a Financial Headline")
    st.markdown(
        "Enter any financial news headline and see how the model scores it. "
        "Try examples like: *'Operating profit rose 14% on strong demand'*"
    )

    # Example buttons
    examples = [
        "Apple reports record quarterly profit, beating analyst estimates",
        "Company files for bankruptcy amid mounting debt concerns",
        "The board approved a $2B share buyback programme",
        "Revenue declined 8% year-over-year due to weak consumer spending",
        "Merger talks between the two firms collapsed over valuation disputes",
    ]
    st.write("**Try an example:**")
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(f"#{i+1}", help=ex, use_container_width=True):
            st.session_state["user_text"] = ex

    # Text input
    user_text = st.text_area(
        "Your headline",
        value=st.session_state.get("user_text", ""),
        height=100,
        placeholder="Paste or type a financial news headline here...",
    )

    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            model = load_model()
            preproc = FinancialTextPreprocessor()

            with st.spinner("Running inference..."):
                t0 = time.time()
                result = model.predict(user_text)
                elapsed = time.time() - t0

            label = result["label"]
            conf = result["score"]
            scores = result["all_scores"]

            # Result card
            colour_map = {
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
            }
            emoji_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
            colour = colour_map[label]

            st.markdown(
                f"""
            <div style='background:{colour}22; border-left:4px solid {colour};
                        border-radius:8px; padding:1rem 1.5rem; margin:1rem 0'>
              <h2 style='margin:0; color:{colour}'>{emoji_map[label]} {label.upper()}</h2>
              <p style='margin:0; color:#666'>Confidence: {conf*100:.1f}% &nbsp;·&nbsp; Latency: {elapsed*1000:.0f}ms</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Probability bars
            st.markdown("**Probability distribution:**")
            fig_bar = go.Figure()
            for sent, score in sorted(scores.items()):
                fig_bar.add_trace(
                    go.Bar(
                        name=sent,
                        x=[score * 100],
                        y=[sent],
                        orientation="h",
                        marker_color=colour_map[sent],
                        text=f"{score*100:.1f}%",
                        textposition="outside",
                    )
                )
            fig_bar.update_layout(
                height=200,
                margin=dict(l=0, r=60, t=10, b=10),
                showlegend=False,
                template=chart_theme,
                xaxis=dict(range=[0, 105], title="Probability (%)"),
                barmode="overlay",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Show preprocessing
            with st.expander("🔬 See preprocessing details"):
                cleaned = preproc.clean(user_text)
                col_a, col_b = st.columns(2)
                col_a.markdown("**Original text:**")
                col_a.code(user_text)
                col_b.markdown("**After preprocessing:**")
                col_b.code(cleaned)
                st.caption(
                    "FinBERT uses the original text (BERT's WordPiece tokenizer "
                    "handles subwords). Preprocessing is used for classical models only."
                )


# TAB 2 — Model Comparison
with tab2:
    st.subheader("Model Performance Comparison")
    st.markdown(
        "Benchmark results on the Financial PhraseBank test set "
        "(20% holdout, stratified split, same random seed)."
    )
    st.caption(
        "ℹ️ FinBERT here is used **zero-shot** (no fine-tuning on this dataset). "
        "Commonly-cited ~91-97% FinBERT figures are usually reported on the "
        "*AllAgree* subset of Financial PhraseBank (sentences where all 16 "
        "annotators agreed). This dataset includes the full agreement spectrum, "
        "so 75.1% reflects performance on harder, more ambiguous sentences too."
    )

    # Real measured results - outputs/scored_headlines.csv for FinBERT,
    # src/classical_models.py for the others (all on the same 20% test split)
    results_df = pd.DataFrame(
        [
            {
                "Model": "VADER (rule-based)",
                "Accuracy": 49.87,
                "Neg F1": 0.29,
                "Neu F1": 0.56,
                "Pos F1": 0.50,
                "Training": "None",
            },
            {
                "Model": "TF-IDF + LogReg",
                "Accuracy": 66.38,
                "Neg F1": 0.40,
                "Neu F1": 0.74,
                "Pos F1": 0.69,
                "Training": "< 1 min",
            },
            {
                "Model": "TF-IDF + SVM",
                "Accuracy": 67.24,
                "Neg F1": 0.14,
                "Neu F1": 0.75,
                "Pos F1": 0.71,
                "Training": "< 1 min",
            },
            {
                "Model": "FinBERT (ProsusAI)",
                "Accuracy": 75.11,
                "Neg F1": 0.60,
                "Neu F1": 0.78,
                "Pos F1": 0.79,
                "Training": "Zero-shot",
            },
        ]
    )

    # Accuracy bar chart
    colours = [
        "#e74c3c" if a < 50 else "#f39c12" if a < 75 else "#2ecc71"
        for a in results_df["Accuracy"]
    ]

    fig_acc = go.Figure(
        go.Bar(
            x=results_df["Model"],
            y=results_df["Accuracy"],
            marker_color=colours,
            text=[f"{a:.1f}%" for a in results_df["Accuracy"]],
            textposition="outside",
        )
    )
    fig_acc.add_hline(
        y=53.6,
        line_dash="dot",
        line_color="gray",
        annotation_text="Majority class baseline (53.6%)",
    )
    fig_acc.update_layout(
        title="Test Set Accuracy by Model",
        yaxis=dict(range=[0, 100], title="Accuracy (%)"),
        template=chart_theme,
        height=350,
        margin=dict(t=50),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # F1 per class
    st.markdown(
        "**Per-class F1 scores** — the real measure of quality on imbalanced data:"
    )
    fig_f1 = go.Figure()
    for cls, colour in [
        ("Neg F1", "#e74c3c"),
        ("Neu F1", "#95a5a6"),
        ("Pos F1", "#2ecc71"),
    ]:
        fig_f1.add_trace(
            go.Bar(
                name=cls.replace(" F1", "").capitalize(),
                x=results_df["Model"],
                y=results_df[cls],
                marker_color=colour,
            )
        )
    fig_f1.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="F1 Score"),
        template=chart_theme,
        height=320,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.caption(
        "Classical models struggle with the negative class (only 14.7% of data) — "
        "SVM's negative F1 of 0.14 means it almost never catches bearish headlines. "
        "FinBERT's contextual attention more than quadruples this to 0.60, even "
        "zero-shot."
    )

    if show_raw:
        st.dataframe(results_df, use_container_width=True)


# TAB 3 — Sentiment vs. Price
with tab3:
    st.subheader(f"Sentiment vs. Next-Day Price Movement — {selected_ticker}")
    st.markdown(
        "Each point is a trading day. X-axis = aggregate news sentiment score. "
        "Y-axis = next-day stock return. A positive correlation means bullish news "
        "predicts upward price movement the following day."
    )
    st.caption(
        " **Methodology note:** Financial PhraseBank sentences have no real "
        "publication dates, so the 5,842 headlines are spread evenly across "
        "the trading calendar as a placeholder alignment. This lets us "
        "demonstrate the analysis pipeline end-to-end, but any correlation "
        "(or lack of one) below isn't a claim about real markets — see the "
        "interpretation box at the bottom for what a real study would need."
    )

    with st.spinner("Preparing correlation data..."):
        daily_sentiment = get_daily_sentiment()
        stock_df = get_stocks(DEFAULT_TICKERS)

    from scipy import stats

    price_df = stock_df[stock_df["ticker"] == selected_ticker].reset_index(drop=True)
    n = min(len(daily_sentiment), len(price_df))
    x = daily_sentiment["mean_score"].values[:n]
    y = price_df["next_day_return"].values[:n]

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    # SAFETY GUARD: pearsonr/spearmanr require at least 2 data points.
    if len(x) < 2:
        st.error(
            f"Not enough overlapping data for {selected_ticker} "
            f"({len(x)} day(s) found, need at least 2). "
            "This usually means stock_data.csv doesn't contain this "
            "ticker — check outputs/stock_data.csv covers all 5 tickers."
        )
        st.stop()

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    dir_acc = (np.sign(x) == np.sign(y)).mean()

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pearson r", f"{pearson_r:+.4f}", delta=f"p={pearson_p:.3f}")
    c2.metric("Spearman ρ", f"{spearman_r:+.4f}", delta=f"p={spearman_p:.3f}")
    c3.metric(
        "Direction acc",
        f"{dir_acc*100:.1f}%",
        delta=f"{(dir_acc-0.5)*100:+.1f}% vs random",
    )
    c4.metric("Days analysed", f"{mask.sum()}")

    # Scatter plot with regression line
    colours_scatter = ["#2ecc71" if yi > 0 else "#e74c3c" for yi in y]
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color=colours_scatter, size=6, opacity=0.6),
            name="Trading days",
            hovertemplate="Sentiment: %{x:.3f}<br>Next-day return: %{y:.2f}%<extra></extra>",
        )
    )

    # OLS regression line
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    fig_scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            line=dict(color="#3498db", width=2, dash="dash"),
            name=f"Regression (r={pearson_r:+.3f})",
        )
    )
    fig_scatter.update_layout(
        title=f"{selected_ticker}: Daily Sentiment Score vs. Next-Day Return",
        xaxis=dict(title="Aggregate Sentiment Score (pos - neg)"),
        yaxis=dict(title="Next-Day Return (%)"),
        template=chart_theme,
        height=450,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Time series overlay
    st.subheader("Sentiment and Price Over Time")
    fig_ts = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Daily Sentiment Score", f"{selected_ticker} Close Price"),
        vertical_spacing=0.08,
    )

    days = list(range(n))
    fig_ts.add_trace(
        go.Scatter(
            x=days, y=x, fill="tozeroy", line=dict(color="#3498db"), name="Sentiment"
        ),
        row=1,
        col=1,
    )
    closes = price_df["close"].values[:n]
    fig_ts.add_trace(
        go.Scatter(x=days, y=closes, line=dict(color="#2ecc71"), name="Close"),
        row=2,
        col=1,
    )
    fig_ts.update_layout(
        template=chart_theme,
        height=500,
        showlegend=True,
        margin=dict(t=40),
    )
    fig_ts.update_yaxes(title_text="Score", row=1)
    fig_ts.update_yaxes(title_text="Price (USD)", row=2)
    st.plotly_chart(fig_ts, use_container_width=True)

    if pearson_p < 0.05 or spearman_p < 0.05:
        st.success(
            f"📌 **Finding:** {selected_ticker} shows a statistically significant "
            f"relationship (p < 0.05) between daily sentiment and next-day return "
            f"under this alignment. Given the methodology note above (synthetic "
            f"date alignment), treat this as a successful pipeline test rather "
            f"than a validated trading signal — re-run with real dated headlines "
            f"before drawing market conclusions."
        )
    else:
        st.info(
            f"📌 **Finding:** No statistically significant relationship for "
            f"{selected_ticker} (Pearson p = {pearson_p:.2f}, Spearman p = "
            f"{spearman_p:.2f}), and direction accuracy "
            f"({dir_acc*100:.1f}%) is statistically indistinguishable from a "
            f"coin flip.\n\n"
            f"**This is the expected result, not a bug.** With headlines spread "
            f"evenly across {n} trading days (no real timestamps available in "
            f"Financial PhraseBank), there's no genuine temporal signal for the "
            f"correlation to detect. The pipeline itself — daily aggregation, "
            f"Pearson/Spearman tests, direction accuracy — is correct and ready "
            f"to use. The missing piece for a real signal is a news source with "
            f"actual publication timestamps (e.g. a financial news API), so each "
            f"headline maps to the trading day it was actually published."
        )


# TAB 4 — Dataset Explorer
with tab4:
    st.subheader("Financial PhraseBank Dataset Explorer")

    df = load_data()

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sentences", f"{len(df):,}")
    col2.metric("Positive", f"{(df['Sentiment']=='positive').sum():,}")
    col3.metric("Negative", f"{(df['Sentiment']=='negative').sum():,}")
    col4.metric("Neutral", f"{(df['Sentiment']=='neutral').sum():,}")

    # Distribution pie
    counts = df["Sentiment"].value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
        title="Label Distribution",
        hole=0.4,
        template=chart_theme,
    )
    fig_pie.update_layout(height=320)

    # Text length distribution
    df["word_count"] = df["Sentence"].str.split().str.len()
    fig_len = px.histogram(
        df,
        x="word_count",
        color="Sentiment",
        nbins=40,
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
        title="Headline Length Distribution (words)",
        template=chart_theme,
        barmode="overlay",
        opacity=0.7,
    )
    fig_len.update_layout(height=320)

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_pie, use_container_width=True)
    c2.plotly_chart(fig_len, use_container_width=True)

    # Filter and search
    st.subheader("Browse Headlines")
    filter_label = st.multiselect(
        "Filter by sentiment",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )
    search_term = st.text_input(
        "Search for a keyword", placeholder="e.g. profit, acquisition"
    )

    filtered = df[df["Sentiment"].isin(filter_label)]
    if search_term:
        filtered = filtered[
            filtered["Sentence"].str.lower().str.contains(search_term.lower(), na=False)
        ]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} sentences")
    st.dataframe(
        filtered[["Sentence", "Sentiment"]]
        .rename(columns={"Sentence": "Headline", "Sentiment": "Label"})
        .head(200),
        use_container_width=True,
        height=400,
    )


# TAB 5 - Batch Financial News Analyzer
with tab5:
    st.subheader("Batch Financial News Analyzer")
    st.markdown(
        "Upload a CSV containing financial headlines to classify them in one run. "
        "Only non-empty values in the selected column are sent to the model."
    )

    uploaded_file = st.file_uploader(
        "Upload headlines CSV",
        type=["csv"],
        help="The file must contain a header row and one column with headlines.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(
                "Could not read this CSV. Check that it is a non-empty UTF-8 CSV "
                f"with a valid header and rows. Details: {exc}"
            )
        except Exception as exc:
            st.error(f"Could not read this upload: {exc}")
        else:
            if uploaded_df.empty or len(uploaded_df.columns) == 0:
                st.error("The uploaded CSV contains no data columns or rows.")
            else:
                headline_column = st.selectbox(
                    "Headline column",
                    options=list(uploaded_df.columns),
                    help="Choose the column containing the financial news text.",
                )
                batch_size = st.number_input(
                    "Inference batch size",
                    min_value=1,
                    max_value=128,
                    value=16,
                    step=1,
                    help="Lower this value if the model runs out of memory.",
                )

                headline_values = uploaded_df[headline_column].fillna("").astype(str).str.strip()
                valid_mask = headline_values.ne("")
                valid_headlines = headline_values[valid_mask].tolist()
                original_headlines = uploaded_df.loc[valid_mask, headline_column].tolist()
                skipped_count = int((~valid_mask).sum())

                st.caption(
                    f"{len(uploaded_df):,} uploaded rows · "
                    f"{len(valid_headlines):,} valid headlines · "
                    f"{skipped_count:,} skipped as missing or empty"
                )

                if not valid_headlines:
                    st.warning(
                        "No valid headlines were found in the selected column. "
                        "Choose another column or upload a file with headline text."
                    )
                elif st.button("Analyze batch", type="primary", use_container_width=True):
                    try:
                        model = load_model()
                        with st.spinner(
                            f"Classifying {len(valid_headlines):,} headlines..."
                        ):
                            predictions = model.predict_batch(
                                valid_headlines,
                                batch_size=int(batch_size),
                                show_progress=False,
                            )
                    except Exception as exc:
                        st.error(f"Batch analysis failed: {exc}")
                    else:
                        result_df = pd.DataFrame(
                            {
                                "Headline": original_headlines,
                                "Predicted sentiment": [
                                    result["label"] for result in predictions
                                ],
                                "Confidence / score": [
                                    result["score"] for result in predictions
                                ],
                            }
                        )

                        score_keys = ["positive", "negative", "neutral"]
                        score_labels = {
                            "positive": "Positive score",
                            "negative": "Negative score",
                            "neutral": "Neutral score",
                        }
                        for score_key in score_keys:
                            result_df[score_labels[score_key]] = [
                                result.get("all_scores", {}).get(score_key, np.nan)
                                for result in predictions
                            ]

                        score_type = predictions[0].get(
                            "score_type", "model score"
                        )
                        st.caption(
                            "Score columns show the active model's "
                            f"{score_type}. They are not a guarantee of prediction correctness."
                        )

                        counts = result_df["Predicted sentiment"].value_counts()
                        summary_cols = st.columns(4)
                        summary_cols[0].metric("Total valid headlines", f"{len(result_df):,}")
                        summary_cols[1].metric(
                            "Positive", f"{int(counts.get('positive', 0)):,}"
                        )
                        summary_cols[2].metric(
                            "Negative", f"{int(counts.get('negative', 0)):,}"
                        )
                        summary_cols[3].metric(
                            "Neutral", f"{int(counts.get('neutral', 0)):,}"
                        )

                        distribution = (
                            result_df["Predicted sentiment"]
                            .value_counts()
                            .rename_axis("Sentiment")
                            .reset_index(name="Count")
                        )
                        distribution["Percentage"] = (
                            distribution["Count"] / len(result_df) * 100
                        )
                        fig_batch = px.bar(
                            distribution,
                            x="Sentiment",
                            y="Count",
                            color="Sentiment",
                            text=distribution["Percentage"].map(lambda value: f"{value:.1f}%"),
                            color_discrete_map={
                                "positive": "#2ecc71",
                                "negative": "#e74c3c",
                                "neutral": "#95a5a6",
                            },
                            template=chart_theme,
                            title="Batch Sentiment Distribution",
                        )
                        fig_batch.update_traces(textposition="outside")
                        fig_batch.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_batch, use_container_width=True)

                        st.dataframe(result_df, use_container_width=True, height=450)
                        st.download_button(
                            "Download predictions as CSV",
                            data=result_df.to_csv(index=False).encode("utf-8"),
                            file_name="batch_sentiment_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )


# Footer
st.divider()
st.caption(
    "Financial Sentiment Analyser · "
    "Built with ProsusAI/FinBERT · "
    "Data: Financial PhraseBank (Malo et al., 2014) · "
    "Stock prices via yfinance"
)
