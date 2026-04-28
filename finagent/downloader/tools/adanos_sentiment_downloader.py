import os
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from finagent.downloader.custom import Downloader
from finagent.registry import DOWNLOADER

load_dotenv(verbose=True)

ADANOS_SOURCES = ("reddit", "x", "news", "polymarket")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_adanos_sentiment_frame(
    stock_payload: dict[str, Any] | None,
    compare_entry: dict[str, Any] | None,
    source: str,
) -> pd.DataFrame:
    trend_history = (compare_entry or {}).get("trend_history") or []
    daily_trend = (stock_payload or {}).get("daily_trend") or []

    rows: dict[str, dict[str, Any]] = {}

    for point in trend_history:
        date = point.get("date")
        if not date:
            continue
        rows[date] = {
            "timestamp": date,
            "adanos_source": source,
            "adanos_buzz_score": _safe_float(point.get("buzz_score")),
            "adanos_mentions": 0,
            "adanos_sentiment_score": 0.0,
            "adanos_bullish_pct": _safe_float((compare_entry or {}).get("bullish_pct")),
            "adanos_bearish_pct": _safe_float((compare_entry or {}).get("bearish_pct")),
            "adanos_trend": (compare_entry or {}).get("trend", "stable"),
            "adanos_latest_buzz_score": _safe_float((compare_entry or {}).get("buzz_score")),
            "adanos_latest_sentiment_score": _safe_float((compare_entry or {}).get("sentiment_score")),
        }

    for point in daily_trend:
        date = point.get("date")
        if not date:
            continue
        row = rows.setdefault(
            date,
            {
                "timestamp": date,
                "adanos_source": source,
                "adanos_buzz_score": _safe_float((compare_entry or {}).get("buzz_score")),
                "adanos_mentions": 0,
                "adanos_sentiment_score": 0.0,
                "adanos_bullish_pct": _safe_float((compare_entry or {}).get("bullish_pct")),
                "adanos_bearish_pct": _safe_float((compare_entry or {}).get("bearish_pct")),
                "adanos_trend": (compare_entry or {}).get("trend", "stable"),
                "adanos_latest_buzz_score": _safe_float((compare_entry or {}).get("buzz_score")),
                "adanos_latest_sentiment_score": _safe_float((compare_entry or {}).get("sentiment_score")),
            },
        )
        row["adanos_mentions"] = _safe_int(point.get("mentions"))
        row["adanos_sentiment_score"] = _safe_float(point.get("sentiment_score"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
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
        )

    frame = pd.DataFrame(rows.values())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["timestamp", "adanos_source"]).reset_index(drop=True)
    return frame


@DOWNLOADER.register_module(force=True)
class AdanosSentimentDownloader(Downloader):
    def __init__(
        self,
        root: str = "",
        token: str = None,
        base_url: str = "https://api.adanos.org",
        delay: float = 0.5,
        start_date: str = "2023-04-01",
        end_date: str = None,
        stocks_path: str = None,
        workdir: str = "",
        tag: str = "",
        days: int = 30,
        sources: list[str] | tuple[str, ...] | None = None,
        timeout: int = 30,
        **kwargs,
    ):
        self.root = root
        self.token = token if token is not None else os.environ.get("ADANOS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)
        self.days = days
        self.sources = tuple(sources or ADANOS_SOURCES)
        self.timeout = timeout
        self.log_path = os.path.join(self.workdir, f"{tag}.txt")

        os.makedirs(self.workdir, exist_ok=True)
        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()
        super().__init__(**kwargs)

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-API-Key"] = self.token
        return headers

    def _fetch_json(self, path: str, params: dict[str, Any]) -> Any:
        response = requests.get(
            url=f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _download_source_frame(self, stock: str, source: str) -> pd.DataFrame:
        compare_payload = self._fetch_json(
            path=f"/{source}/stocks/v1/compare",
            params={"tickers": stock, "days": self.days},
        )
        stock_payload = self._fetch_json(
            path=f"/{source}/stocks/v1/stock/{stock}",
            params={"days": self.days},
        )

        compare_entry = None
        if isinstance(compare_payload, list) and compare_payload:
            compare_entry = compare_payload[0]

        return build_adanos_sentiment_frame(
            stock_payload=stock_payload if isinstance(stock_payload, dict) else None,
            compare_entry=compare_entry if isinstance(compare_entry, dict) else None,
            source=source,
        )

    def download(self, stocks=None, start_date=None, end_date=None):
        if not self.token:
            raise ValueError("ADANOS_API_KEY is required to download Adanos sentiment data.")

        stocks = stocks if stocks else self.stocks
        start_bound = pd.to_datetime(start_date or self.start_date) if (start_date or self.start_date) else None
        end_bound = pd.to_datetime(end_date or self.end_date) if (end_date or self.end_date) else None

        for stock in stocks:
            frames = []
            for source in self.sources:
                try:
                    time.sleep(self.delay)
                    frame = self._download_source_frame(stock=stock, source=source)
                except Exception as exc:
                    with open(self.log_path, "a") as op:
                        op.write(f"{stock},{source},{exc}\n")
                    continue

                if frame.empty:
                    continue
                frames.append(frame)

            if not frames:
                continue

            output = pd.concat(frames, axis=0, ignore_index=True)
            output["timestamp"] = pd.to_datetime(output["timestamp"])
            if start_bound is not None:
                output = output[output["timestamp"] >= start_bound]
            if end_bound is not None:
                output = output[output["timestamp"] < end_bound]

            output = output.sort_values(["timestamp", "adanos_source"]).reset_index(drop=True)
            output["timestamp"] = output["timestamp"].dt.strftime("%Y-%m-%d")
            output.to_csv(os.path.join(self.workdir, f"{stock}.csv"), index=False)
