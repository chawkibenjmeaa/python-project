"""Run the agents on several tickers and save a summary to evaluation.csv.

Usage:  python evaluate.py AAPL MSFT TSLA F INTC NVDA JPM
"""
import re
import sys
import time

import pandas as pd

from agents import build_graph

DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "F", "INTC", "NVDA", "JPM"]
tickers = sys.argv[1:] or DEFAULT_TICKERS
graph = build_graph()
rows = []

for ticker in tickers:
    start = time.time()
    try:
        result = graph.invoke({"ticker": ticker})
        view = re.search(r"\b(BUY|HOLD|SELL)\b", result["memo"])
        news = result["news"]
        rows.append({
            "ticker": ticker,
            "recommendation": view.group(1) if view else "?",
            "headlines": len(news),
            "avg_sentiment": round(sum(n["score"] for n in news) / len(news), 2) if news else None,
            "numbers_verified": len(result["check"]["verified"]),
            "numbers_unverified": len(result["check"]["unverified"]),
            "accuracy_pct": result["check"].get("accuracy_pct", 100.0),
            "unverified": " ".join(result["check"]["unverified"]),
            "seconds": round(time.time() - start),
        })
    except Exception as error:
        rows.append({"ticker": ticker, "recommendation": f"ERROR: {error}"})
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("evaluation.csv", index=False)
print("\nSaved evaluation.csv")
if "accuracy_pct" in df.columns:
    print(f"Average numerical accuracy: {df['accuracy_pct'].mean():.1f}% | Average time: {df['seconds'].mean():.1f}s")
