"""Run the agents on several tickers and save a summary to evaluation.csv.

Usage:  python evaluate.py AAPL MSFT TSLA F INTC
"""
import re
import sys
import time

import pandas as pd

from agents import build_graph

tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA", "F", "INTC"]
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
            "unverified": " ".join(result["check"]["unverified"]),
            "seconds": round(time.time() - start),
        })
    except Exception as error:
        rows.append({"ticker": ticker, "recommendation": f"ERROR: {error}"})
    print(rows[-1])

pd.DataFrame(rows).to_csv("evaluation.csv", index=False)
print("Saved evaluation.csv")
