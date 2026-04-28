from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dataset_module():
    finagent_pkg = types.ModuleType("finagent")
    finagent_pkg.__path__ = []

    data_pkg = types.ModuleType("finagent.data")
    data_pkg.__path__ = []
    data_pkg.BaseDataset = object

    class _Registry:
        def register_module(self, force=False):
            def decorator(cls):
                return cls

            return decorator

    registry_module = types.ModuleType("finagent.registry")
    registry_module.DATASET = _Registry()

    original_modules = {}
    for name, module in {
        "finagent": finagent_pkg,
        "finagent.data": data_pkg,
        "finagent.registry": registry_module,
    }.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        return _load_module("test_finagent_dataset", "finagent/data/dataset.py")
    finally:
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_downloader_module():
    finagent_pkg = types.ModuleType("finagent")
    finagent_pkg.__path__ = []

    downloader_pkg = types.ModuleType("finagent.downloader")
    downloader_pkg.__path__ = []

    custom_module = types.ModuleType("finagent.downloader.custom")
    custom_module.Downloader = object

    class _Registry:
        def register_module(self, force=False):
            def decorator(cls):
                return cls

            return decorator

    registry_module = types.ModuleType("finagent.registry")
    registry_module.DOWNLOADER = _Registry()

    original_modules = {}
    for name, module in {
        "finagent": finagent_pkg,
        "finagent.downloader": downloader_pkg,
        "finagent.downloader.custom": custom_module,
        "finagent.registry": registry_module,
    }.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        return _load_module(
            "test_adanos_downloader",
            "finagent/downloader/tools/adanos_sentiment_downloader.py",
        )
    finally:
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_normalize_sentiment_dataframe_accepts_adanos_schema():
    dataset_module = _load_dataset_module()
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-28",
                "adanos_source": "reddit",
                "adanos_buzz_score": 74.3,
                "adanos_mentions": 1784,
                "adanos_sentiment_score": -0.03,
                "adanos_bullish_pct": 28.0,
                "adanos_bearish_pct": 19.0,
                "adanos_trend": "falling",
                "adanos_latest_buzz_score": 74.3,
                "adanos_latest_sentiment_score": -0.03,
                "extra": "ignored",
            }
        ]
    )

    normalized = dataset_module.normalize_sentiment_dataframe(frame)

    assert list(normalized.columns) == dataset_module.ADANOS_SENTIMENT_COLUMNS
    assert normalized.iloc[0]["adanos_source"] == "reddit"


def test_build_adanos_sentiment_frame_merges_compare_and_daily_trend():
    downloader_module = _load_downloader_module()

    frame = downloader_module.build_adanos_sentiment_frame(
        stock_payload={
            "daily_trend": [
                {"date": "2026-04-27", "mentions": 12, "sentiment_score": 0.35},
                {"date": "2026-04-28", "mentions": 18, "sentiment_score": 0.41},
            ]
        },
        compare_entry={
            "buzz_score": 71.5,
            "sentiment_score": 0.38,
            "bullish_pct": 61.0,
            "bearish_pct": 17.0,
            "trend": "rising",
            "trend_history": [
                {"date": "2026-04-27", "buzz_score": 68.0},
                {"date": "2026-04-28", "buzz_score": 71.5},
            ],
        },
        source="reddit",
    )

    assert list(frame["timestamp"]) == ["2026-04-27", "2026-04-28"]
    assert list(frame["adanos_mentions"]) == [12, 18]
    assert list(frame["adanos_buzz_score"]) == [68.0, 71.5]
    assert set(frame["adanos_trend"]) == {"rising"}
    assert set(frame["adanos_source"]) == {"reddit"}


def test_format_social_sentiment_text_renders_adanos_sources():
    sentiment_module = _load_module("test_prompt_sentiment", "finagent/prompt/sentiment.py")
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-28",
                "adanos_source": "reddit",
                "adanos_buzz_score": 74.3,
                "adanos_mentions": 1784,
                "adanos_sentiment_score": -0.03,
                "adanos_bullish_pct": 28.0,
                "adanos_bearish_pct": 19.0,
                "adanos_trend": "falling",
                "adanos_latest_buzz_score": 74.3,
                "adanos_latest_sentiment_score": -0.03,
            },
            {
                "timestamp": "2026-04-28",
                "adanos_source": "news",
                "adanos_buzz_score": 63.1,
                "adanos_mentions": 94,
                "adanos_sentiment_score": 0.12,
                "adanos_bullish_pct": 52.0,
                "adanos_bearish_pct": 13.0,
                "adanos_trend": "stable",
                "adanos_latest_buzz_score": 63.1,
                "adanos_latest_sentiment_score": 0.12,
            },
        ]
    )

    text = sentiment_module.format_social_sentiment_text(frame, "2026-04-28")

    assert "Source: reddit" in text
    assert "Source: news" in text
    assert "Trend: falling" in text
    assert "Mentions: 1784" in text
