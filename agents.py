"""The agents, their prompts and the LangGraph workflow.

fetch_data -> fundamentals + sentiment (in parallel) -> risk -> portfolio_manager -> verify

Run from the terminal:  python agents.py AAPL
"""
import json
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

import data_tools
from verify import check_memo

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RULES = (
    "Only use numbers that appear in the data you are given. Never invent or estimate a figure. "
    "If something is missing, say it is missing. Be concise."
)

FUNDAMENTALS_PROMPT = (
    "You are a fundamentals analyst. From the financial data provided (latest annual report "
    "10-K and latest quarterly report 10-Q), assess the company's "
    "profitability, balance sheet strength, cash generation and valuation. "
    "Write 4 to 6 bullet points, then one line starting with 'View:' (positive, neutral or negative). "
    + RULES
)

SENTIMENT_PROMPT = (
    "You are a news sentiment analyst. From the recent headlines and their sentiment scores "
    "(-1 = very negative, +1 = very positive), describe the tone of the news and the main themes. "
    "Write 3 to 5 bullet points, then one line starting with 'View:' (positive, neutral or negative). "
    "If there are no headlines, say so and give a neutral view. " + RULES
)

RISK_PROMPT = (
    "You are a sceptical risk analyst. Challenge the fundamentals analyst and the sentiment analyst: "
    "point out weak arguments, things they ignored and red flags in the data "
    "(leverage, margins, valuation, negative news, price behaviour). "
    "Write 3 to 6 bullet points, then one line starting with 'Risk level:' (low, medium or high). "
    + RULES
)

PM_PROMPT = (
    "You are a portfolio manager. Weigh the three analyst reports and write a short investment memo "
    "in Markdown with exactly these sections:\n"
    "## Recommendation\nBUY, HOLD or SELL in bold, plus one sentence.\n"
    "## Investment case\n3 to 4 bullet points.\n"
    "## Key risks\n2 to 3 bullet points.\n"
    "## Reasoning\nA short paragraph explaining how you weighed fundamentals, sentiment and risk.\n"
    "Take a clear position: HOLD is not a safe default. Choose BUY when quality and valuation are "
    "attractive for the risk, SELL when losses, leverage, valuation or news clearly point down, "
    "and HOLD only when the evidence is genuinely balanced. "
    "This is a first-pass screening note, not financial advice. " + RULES
)


class State(TypedDict, total=False):
    ticker: str
    company: str
    prices: object      # DataFrame with a Close column
    facts: dict         # label -> formatted figure
    news: list          # [{date, headline, source, score}]
    fundamentals: str
    sentiment: str
    risk: str
    memo: str
    check: dict         # {"verified": [...], "unverified": [...]}


def ask(system, user):
    response = OpenAI().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content


def facts_text(state):
    lines = [f"- {label}: {value}" for label, value in state["facts"].items()]
    return f"Company: {state['company']} ({state['ticker']})\n" + "\n".join(lines)


def news_text(state):
    news = state["news"]
    if not news:
        return "No recent headlines available."
    average = sum(n["score"] for n in news) / len(news)
    lines = [f"- {n['date']} | score {n['score']} | {n['headline']}" for n in news]
    return f"{len(news)} headlines, average score {average:.2f}\n" + "\n".join(lines)


# ---------- Nodes ----------

def fetch_data(state):
    ticker = state["ticker"].upper().strip()
    prices = data_tools.get_prices(ticker)
    price = float(prices["Close"].iloc[-1])
    company, fundamentals = data_tools.get_fundamentals(ticker, price)
    facts = {**data_tools.price_facts(prices), **fundamentals}
    return {"ticker": ticker, "company": company, "prices": prices,
            "facts": facts, "news": data_tools.get_news(ticker)}


def fundamentals_agent(state):
    return {"fundamentals": ask(FUNDAMENTALS_PROMPT, facts_text(state))}


def sentiment_agent(state):
    user = f"Company: {state['company']} ({state['ticker']})\n{news_text(state)}"
    return {"sentiment": ask(SENTIMENT_PROMPT, user)}


def risk_agent(state):
    user = (f"DATA\n{facts_text(state)}\n\nNEWS\n{news_text(state)}\n\n"
            f"FUNDAMENTALS ANALYST\n{state['fundamentals']}\n\n"
            f"SENTIMENT ANALYST\n{state['sentiment']}")
    return {"risk": ask(RISK_PROMPT, user)}


def portfolio_manager(state):
    user = (f"DATA\n{facts_text(state)}\n\n"
            f"FUNDAMENTALS ANALYST\n{state['fundamentals']}\n\n"
            f"SENTIMENT ANALYST\n{state['sentiment']}\n\n"
            f"RISK ANALYST\n{state['risk']}")
    return {"memo": ask(PM_PROMPT, user)}


def verify(state):
    source = facts_text(state) + "\n" + news_text(state) + "\n" + json.dumps(state["news"])
    return {"check": check_memo(state["memo"], source)}


# ---------- Graph ----------

def build_graph():
    g = StateGraph(State)
    g.add_node("fetch_data", fetch_data)
    g.add_node("fundamentals_agent", fundamentals_agent)
    g.add_node("sentiment_agent", sentiment_agent)
    g.add_node("risk_agent", risk_agent)
    g.add_node("portfolio_manager", portfolio_manager)
    g.add_node("verify", verify)

    g.add_edge(START, "fetch_data")
    g.add_edge("fetch_data", "fundamentals_agent")
    g.add_edge("fetch_data", "sentiment_agent")
    g.add_edge(["fundamentals_agent", "sentiment_agent"], "risk_agent")  # wait for both
    g.add_edge("risk_agent", "portfolio_manager")
    g.add_edge("portfolio_manager", "verify")
    g.add_edge("verify", END)
    return g.compile()


if __name__ == "__main__":
    result = build_graph().invoke({"ticker": sys.argv[1] if len(sys.argv) > 1 else "AAPL"})
    print(result["memo"])
    print("\nNumbers not found in the source data:", result["check"]["unverified"] or "none")
