"""
Pluggable live news providers.

All providers return a list of :class:`NewsItem` for a symbol over a date range.
Network/credential failures are swallowed and yield an empty list, so the rest
of the pipeline degrades gracefully to "no news" rather than crashing.

Providers
---------
* ``yfinance``     -- no API key, but only a handful of *recent* headlines.
* ``finnhub``      -- set ``FINNHUB_API_KEY``; returns historical company news
                      within a date range (free tier ~1 year of history).
* ``alphavantage`` -- set ``ALPHAVANTAGE_API_KEY``; NEWS_SENTIMENT endpoint with
                      time ranges (free tier is heavily rate limited).

``get_provider()`` auto-selects a keyed provider when a key is present, else
falls back to yfinance.
"""

import os
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd


def load_env_file(path=None):
    """
    Load ``KEY=VALUE`` lines from a ``.env`` file at the project root into
    ``os.environ`` (without overriding variables already set in the real
    environment). Lets you keep API keys in a gitignored file instead of the
    shell. Called automatically on import.
    """
    root = Path(__file__).resolve().parents[2]
    path = Path(path) if path else root / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


load_env_file()


@dataclass
class NewsItem:
    date: date
    headline: str
    source: str = ""

    def __post_init__(self):
        if isinstance(self.date, (datetime, pd.Timestamp)):
            self.date = self.date.date()


def _http_get_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": "stocksight-news/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class NewsProvider:
    name = "base"

    def fetch(self, symbol: str, start, end) -> list:
        raise NotImplementedError

    @staticmethod
    def _as_date(value) -> date:
        return pd.to_datetime(value).date()


class YFinanceNewsProvider(NewsProvider):
    name = "yfinance"

    def fetch(self, symbol, start, end):
        try:
            import yfinance as yf
            raw = yf.Ticker(symbol).news or []
        except Exception:
            return []
        start_d, end_d = self._as_date(start), self._as_date(end)
        items = []
        for entry in raw:
            # yfinance has used both a flat schema and a nested {'content': {...}} one.
            content = entry.get("content", entry) if isinstance(entry, dict) else {}
            title = content.get("title") or entry.get("title")
            if not title:
                continue
            ts = (entry.get("providerPublishTime")
                  or content.get("pubDate") or content.get("displayTime"))
            try:
                if isinstance(ts, (int, float)):
                    d = datetime.utcfromtimestamp(ts).date()
                elif ts:
                    d = pd.to_datetime(ts).date()
                else:
                    d = end_d
            except Exception:
                d = end_d
            if start_d <= d <= end_d:
                items.append(NewsItem(d, str(title), self.name))
        return items


class FinnhubNewsProvider(NewsProvider):
    name = "finnhub"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")

    def _company(self, symbol):
        # Finnhub expects bare tickers (US-centric). Strip the NSE suffix.
        return symbol.replace(".NS", "").replace(".BO", "")

    def fetch(self, symbol, start, end):
        if not self.api_key:
            return []
        start_d, end_d = self._as_date(start), self._as_date(end)
        items, cursor = [], start_d
        # The endpoint caps each call's range; page through in <=30-day windows.
        while cursor <= end_d:
            window_end = min(cursor + timedelta(days=30), end_d)
            params = urllib.parse.urlencode({
                "symbol": self._company(symbol),
                "from": cursor.isoformat(),
                "to": window_end.isoformat(),
                "token": self.api_key,
            })
            try:
                data = _http_get_json(f"https://finnhub.io/api/v1/company-news?{params}")
            except Exception:
                data = []
            for entry in data or []:
                headline = entry.get("headline")
                ts = entry.get("datetime")
                if not headline or not ts:
                    continue
                items.append(NewsItem(datetime.utcfromtimestamp(ts).date(),
                                      str(headline), self.name))
            cursor = window_end + timedelta(days=1)
            time.sleep(0.2)  # be polite to the free tier
        return items


class AlphaVantageNewsProvider(NewsProvider):
    name = "alphavantage"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")

    def fetch(self, symbol, start, end):
        if not self.api_key:
            return []
        start_d, end_d = self._as_date(start), self._as_date(end)
        params = urllib.parse.urlencode({
            "function": "NEWS_SENTIMENT",
            "tickers": symbol.replace(".NS", ".BSE"),
            "time_from": start_d.strftime("%Y%m%dT0000"),
            "time_to": end_d.strftime("%Y%m%dT2359"),
            "limit": "1000",
            "apikey": self.api_key,
        })
        try:
            data = _http_get_json(f"https://www.alphavantage.co/query?{params}")
        except Exception:
            return []
        items = []
        for entry in (data or {}).get("feed", []):
            title = entry.get("title")
            ts = entry.get("time_published")
            if not title or not ts:
                continue
            try:
                d = datetime.strptime(ts[:8], "%Y%m%d").date()
            except Exception:
                continue
            items.append(NewsItem(d, str(title), self.name))
        return items


class MarketauxNewsProvider(NewsProvider):
    """
    Marketaux news (https://marketaux.com) -- covers Indian (NSE/BSE) markets with
    entity tagging and sentiment. Set ``MARKETAUX_API_KEY`` (free tier ~100 req/day,
    a few articles per request). Pass the symbol as e.g. ``RELIANCE.NS``.
    """
    name = "marketaux"

    def __init__(self, api_key=None, max_pages: int = 20):
        self.api_key = api_key or os.environ.get("MARKETAUX_API_KEY", "")
        self.max_pages = max_pages

    def fetch(self, symbol, start, end):
        if not self.api_key:
            return []
        start_d, end_d = self._as_date(start), self._as_date(end)
        items, page = [], 1
        while page <= self.max_pages:
            params = urllib.parse.urlencode({
                "symbols": symbol,
                "filter_entities": "true",
                "language": "en",
                "published_after": start_d.strftime("%Y-%m-%dT00:00"),
                "published_before": end_d.strftime("%Y-%m-%dT23:59"),
                "page": page,
                "api_token": self.api_key,
            })
            try:
                data = _http_get_json(f"https://api.marketaux.com/v1/news/all?{params}")
            except Exception:
                break
            batch = (data or {}).get("data", [])
            if not batch:
                break
            for entry in batch:
                title = entry.get("title")
                ts = entry.get("published_at")
                if not title or not ts:
                    continue
                try:
                    d = pd.to_datetime(ts).date()
                except Exception:
                    continue
                items.append(NewsItem(d, str(title), self.name))
            meta = (data or {}).get("meta", {})
            if page * int(meta.get("returned", len(batch)) or len(batch)) >= int(meta.get("found", 0) or 0):
                break
            page += 1
            time.sleep(0.3)
        return items


_PROVIDERS = {
    "yfinance": YFinanceNewsProvider,
    "finnhub": FinnhubNewsProvider,
    "alphavantage": AlphaVantageNewsProvider,
    "marketaux": MarketauxNewsProvider,
}


def get_provider(name: str = None) -> NewsProvider:
    """
    Return a news provider. With ``name=None`` auto-select: Finnhub or Alpha
    Vantage if their API key is set, otherwise yfinance.
    """
    if name is None:
        if os.environ.get("FINNHUB_API_KEY"):
            name = "finnhub"
        elif os.environ.get("ALPHAVANTAGE_API_KEY"):
            name = "alphavantage"
        elif os.environ.get("MARKETAUX_API_KEY"):
            name = "marketaux"
        else:
            name = "yfinance"
    key = name.lower()
    if key not in _PROVIDERS:
        raise ValueError(f"Unknown news provider '{name}'. Choose from {list(_PROVIDERS)}.")
    return _PROVIDERS[key]()
