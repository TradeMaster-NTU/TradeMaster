from __future__ import annotations

import pandas as pd


ADANOS_SENTIMENT_COLUMNS = [
    "adanos_source",
    "adanos_buzz_score",
    "adanos_mentions",
    "adanos_sentiment_score",
    "adanos_bullish_pct",
    "adanos_bearish_pct",
    "adanos_trend",
    "adanos_latest_buzz_score",
    "adanos_latest_sentiment_score",
]


def format_social_sentiment_text(
    sentiment: pd.DataFrame | None,
    date: str,
) -> str:
    """Render the latest available social sentiment slice for prompt templates."""
    if sentiment is None or sentiment.empty:
        return "There is no social sentiment data today.\n"

    frame = sentiment.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    else:
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        frame["timestamp"] = frame.index

    current_date = pd.to_datetime(date)
    frame = frame[frame["timestamp"] <= current_date]

    if frame.empty:
        return "There is no social sentiment data today.\n"

    latest_timestamp = frame["timestamp"].max()
    latest_rows = frame[frame["timestamp"] == latest_timestamp].copy()

    if "adanos_source" in latest_rows.columns:
        summaries = []
        for _, row in latest_rows.iterrows():
            summaries.append(
                "Adanos Social Sentiment\n"
                f"Source: {row.get('adanos_source', 'unknown')}\n"
                f"Date: {latest_timestamp.strftime('%Y-%m-%d')}\n"
                f"Buzz Score: {row.get('adanos_latest_buzz_score', row.get('adanos_buzz_score', 'n/a'))}\n"
                f"Sentiment Score: {row.get('adanos_latest_sentiment_score', row.get('adanos_sentiment_score', 'n/a'))}\n"
                f"Bullish: {row.get('adanos_bullish_pct', 'n/a')}%\n"
                f"Bearish: {row.get('adanos_bearish_pct', 'n/a')}%\n"
                f"Mentions: {row.get('adanos_mentions', 'n/a')}\n"
                f"Trend: {row.get('adanos_trend', 'n/a')}\n"
            )
        return "\n".join(summaries)

    row = latest_rows.iloc[-1]
    return (
        "Social Sentiment\n"
        f"Date: {latest_timestamp.strftime('%Y-%m-%d')}\n"
        f"Stocktwits Posts: {row.get('stocktwits_posts', 'n/a')}\n"
        f"Stocktwits Comments: {row.get('stocktwits_comments', 'n/a')}\n"
        f"Stocktwits Likes: {row.get('stocktwits_likes', 'n/a')}\n"
        f"Stocktwits Impressions: {row.get('stocktwits_impressions', 'n/a')}\n"
        f"Stocktwits Sentiment: {row.get('stocktwits_sentiment', 'n/a')}\n"
    )
