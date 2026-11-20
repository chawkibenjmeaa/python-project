# AI Investment Memo

A multi-agent system built with **LangGraph** and **Streamlit** that generates first-pass equity research memos (`BUY` / `HOLD` / `SELL`) in seconds using free financial data sources (SEC EDGAR, yfinance/Stooq, Finnhub).

## Project Architecture
* `agents.py`: Multi-agent workflow (`Fundamentals`, `Sentiment`, `Risk`, `Portfolio Manager`) orchestrated with LangGraph.
* `data_tools.py`: Financial statements (`10-K`/`10-Q` via `edgartools`), stock prices (`yfinance`/Stooq), and company news with sentiment scoring (`Finnhub` + LLM/lexical fallback).
* `verify.py`: Automated numerical fact-checking engine ensuring every figure in the final memo matches the source data.
* `app.py`: Interactive Streamlit web dashboard with real-time agent updates and charts.
* `evaluate.py`: Multi-ticker benchmark script generating `evaluation.csv`.

## Evaluation & Fact-Checking Results
Running `python evaluate.py` across 6 benchmark tickers (`AAPL`, `MSFT`, `TSLA`, `F`, `INTC`, `NVDA`) confirmed **100% numerical accuracy** (48/48 figures verified, 0 unverified) with an average execution time of ~20 seconds per memo.
