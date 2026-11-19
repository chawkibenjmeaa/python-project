"""Data tools: prices (yfinance / Stooq), fundamentals (SEC EDGAR), news (Finnhub).

Every number the agents see comes from this file.
"""
import json
import os
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from edgar import Company, set_identity
from openai import OpenAI

CACHE_DIR = Path(__file__).parent / ".cache"


def cached(func):
    """Cache a function's result on disk for one day, per ticker."""
    def wrapper(ticker, *args):
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"{func.__name__}_{ticker}_{date.today()}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        result = func(ticker, *args)
        path.write_bytes(pickle.dumps(result))
        return result
    return wrapper


# ---------- Prices ----------

@cached
def get_prices(ticker):
    """One year of daily closing prices. yfinance first, Stooq as a backup."""
    try:
        close = yf.Ticker(ticker).history(period="1y")["Close"]
        if close.empty:
            raise ValueError("no data from yfinance")
        close.index = close.index.tz_localize(None)
    except Exception:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        close = df["Close"].last("365D")
    return close.to_frame("Close")


# ---------- Fundamentals ----------

def _div(a, b):
    return a / b if a is not None and b else None


def _money(x):
    return "n/a" if x is None else f"${x / 1e9:.2f}B"


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _ratio(x):
    return "n/a" if x is None else f"{x:.2f}"


@cached
def get_fundamentals(ticker, price):
    """Key figures from the latest 10-K plus ratios. Returns (company name, facts dict)."""
    set_identity(os.getenv("EDGAR_IDENTITY", "Investment Memo Project memo-project@example.com"))
    company = Company(ticker)
    f = company.get_financials()
    if f is None:
        raise ValueError(f"No 10-K financials found on SEC EDGAR for {ticker}")

    revenue = f.get_revenue()
    net_income = f.get_net_income()
    op_income = f.get_operating_income()
    assets = f.get_total_assets()
    liabilities = f.get_total_liabilities()
    equity = f.get_stockholders_equity()
    if liabilities is None and assets is not None and equity is not None:
        liabilities = assets - equity  # some filings do not report the total directly
    fcf = f.get_free_cash_flow()
    shares = f.get_shares_outstanding_diluted()
    market_cap = price * shares if shares else None

    facts = {
        "Latest 10-K filed on": str(company.get_filings(form="10-K").latest().filing_date),
        "Revenue (latest 10-K)": _money(revenue),
        "Operating income": _money(op_income),
        "Net income": _money(net_income),
        "Free cash flow": _money(fcf),
        "Total assets": _money(assets),
        "Total liabilities": _money(liabilities),
        "Shareholders' equity": _money(equity),
        "Market cap": _money(market_cap),
        "Operating margin": _pct(_div(op_income, revenue)),
        "Net margin": _pct(_div(net_income, revenue)),
        "Return on equity": _pct(_div(net_income, equity)),
        "Liabilities / equity": _ratio(_div(liabilities, equity)),
        "Current ratio": _ratio(_div(f.get_current_assets(), f.get_current_liabilities())),
        "P/E": _ratio(_div(market_cap, net_income)) if net_income and net_income > 0 else "n/a (loss)",
        "P/S": _ratio(_div(market_cap, revenue)),
        "FCF yield": _pct(_div(fcf, market_cap)),
    }
    facts.update(_quarter_facts(company))
    return company.name, facts


def _quarter_facts(company):
    """Figures from the latest 10-Q. Returns {} if SEC does not answer (it happens)."""
    for _ in range(2):
        try:
            q = company.get_quarterly_financials()
            revenue, net_income = q.get_revenue(), q.get_net_income()
            return {
                "Latest 10-Q filed on": str(company.get_filings(form="10-Q").latest().filing_date),
                "Quarterly revenue (latest 10-Q)": _money(revenue),
                "Quarterly net income": _money(net_income),
                "Quarterly net margin": _pct(_div(net_income, revenue)),
            }
        except Exception:
            continue
    return {}


def price_facts(prices):
    close = prices["Close"]
    return {
        "Last price": f"${close.iloc[-1]:.2f}",
        "1-year return": _pct(close.iloc[-1] / close.iloc[0] - 1),
        "52-week high": f"${close.max():.2f}",
        "52-week low": f"${close.min():.2f}",
    }


# ---------- News and sentiment ----------

POSITIVE = {"beat", "beats", "surge", "surges", "soar", "soars", "gain", "gains", "rise", "rises",
            "record", "growth", "upgrade", "upgrades", "strong", "profit", "rally", "bullish",
            "outperform", "buy", "boost", "boosts", "win", "wins", "jump", "jumps", "raises"}
NEGATIVE = {"miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges", "loss",
            "losses", "lawsuit", "probe", "downgrade", "downgrades", "weak", "cut", "cuts",
            "bearish", "sell", "decline", "declines", "risk", "fraud", "layoffs", "recall",
            "slump", "slumps", "warning", "warns", "investigation", "fine", "sinks", "tumbles"}


def score_headline(text):
    """Simple word-list sentiment score between -1 and 1."""
    words = [w.strip(".,:;!?'\"()").lower() for w in text.split()]
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return 0.0 if pos + neg == 0 else round((pos - neg) / (pos + neg), 2)


def score_headlines(ticker, headlines):
    """Sentiment score per headline, from GPT. Falls back to the word list if that fails."""
    try:
        response = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                    f"Score each news headline for its impact on {ticker} stock, from -1 (very "
                    "negative) to 1 (very positive). Use 0 if neutral or not about the company. "
                    'Reply in JSON: {"scores": [...]} with one number per headline, in order.'},
                {"role": "user", "content": "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))},
            ],
        )
        scores = json.loads(response.choices[0].message.content)["scores"]
        if len(scores) == len(headlines):
            return [max(-1.0, min(1.0, round(float(s), 2))) for s in scores]
    except Exception:
        pass
    return [score_headline(h) for h in headlines]


@cached
def get_news(ticker):
    """Last 4 weeks of company news from Finnhub (10 headlines per week), with sentiment scores."""
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return []
    news = []
    for week in range(4):  # one request per week, because Finnhub limits each answer
        end = date.today() - timedelta(days=7 * week)
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": end - timedelta(days=6), "to": end, "token": key},
            timeout=20,
        )
        r.raise_for_status()
        items = r.json()
        step = max(1, len(items) // 10)
        for item in items[::step][:10]:
            news.append({
                "date": date.fromtimestamp(item["datetime"]).isoformat(),
                "headline": item["headline"],
                "source": item["source"],
            })
    scores = score_headlines(ticker, [n["headline"] for n in news]) if news else []
    for n, score in zip(news, scores):
        n["score"] = score
    return news
