"""Redesigned Streamlit app.  Run with:  python -m streamlit run app_v2.py

Same data flow as app.py (build_graph().stream), new interface:
sidebar controls, live pipeline tracker, verdict badge, metric cards, tabs.
"""
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

from agents import build_graph

try:
    import plotly.graph_objects as go
except ImportError:  # the app still works without plotly, with plainer charts
    go = None

TEAL, AMBER, RED, GREEN, MUTED = "#14b8a6", "#f59e0b", "#ef4444", "#22c55e", "#8aa4b3"
RATING_COLORS = {"BUY": GREEN, "HOLD": AMBER, "SELL": RED}

PIPELINE = [  # node name -> label shown in the tracker
    ("fetch_data", "Data"),
    ("fundamentals_agent", "Fundamentals"),
    ("sentiment_agent", "Sentiment"),
    ("risk_agent", "Risk"),
    ("portfolio_manager", "Portfolio manager"),
    ("verify", "Number check"),
]
AGENTS = {  # node name -> (title, state key, icon)
    "fundamentals_agent": ("Fundamentals agent", "fundamentals", "📊"),
    "sentiment_agent": ("Sentiment agent", "sentiment", "📰"),
    "risk_agent": ("Risk agent", "risk", "🛡️"),
}
EXAMPLES = ["AAPL", "MSFT", "TSLA", "F", "INTC", "NVDA"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&display=swap');
h1, h2, h3 {{ font-family: 'Montserrat', sans-serif; letter-spacing: -0.01em; }}
.hero h1 {{ font-size: 2.6rem; margin: 0 0 .3rem 0; padding: 0; }}
.hero p {{ color: {MUTED}; max-width: 60ch; margin: 0 0 1.2rem 0; }}
.pipeline {{ display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin: .5rem 0 1.2rem 0; }}
.step {{ padding: .35rem .8rem; border-radius: 999px; border: 1px solid #1f3a4d;
        color: {MUTED}; font-size: .85rem; }}
.step.active {{ border-color: {AMBER}; color: {AMBER}; }}
.step.done {{ border-color: {TEAL}; background: {TEAL}22; color: {TEAL}; }}
.arrow {{ color: #1f3a4d; }}
.verdict {{ text-align: center; padding: .6rem 1rem; border: 2px solid; border-radius: 12px;
           font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 2rem; line-height: 1.1; }}
.verdict small {{ display: block; font-family: sans-serif; font-weight: 400; font-size: .75rem;
                 color: {MUTED}; margin-top: .2rem; }}
[data-testid="stMetric"] {{ background: #0e2433; border: 1px solid #1f3a4d; border-radius: 10px;
                           padding: .8rem 1rem; }}
</style>
"""


def md(text):
    """Escape $ so Streamlit does not render prices as LaTeX."""
    return str(text).replace("$", "\\$")


def pipeline_html(done):
    active = next((key for key, _ in PIPELINE if key not in done), None)
    pills = []
    for key, label in PIPELINE:
        css = "done" if key in done else "active" if key == active else ""
        pills.append(f'<span class="step {css}">{label}</span>')
    return '<div class="pipeline">' + '<span class="arrow">&rsaquo;</span>'.join(pills) + "</div>"


def find_rating(state):
    """Read BUY, HOLD or SELL from the memo's '## Recommendation' section."""
    memo = str(state.get("memo", ""))
    section = re.search(r"##\s*Recommendation(.*?)(?=\n##|\Z)", memo, re.I | re.S)
    found = re.search(r"\b(BUY|HOLD|SELL)\b", (section.group(1) if section else memo).upper())
    return found.group(1) if found else "N/A"


def queue_ticker(ticker):
    """Button callback: set the ticker and ask the main script to run."""
    st.session_state.ticker = ticker
    st.session_state.run = True


# ---------------------------------------------------------------- charts
def style_figure(fig, height=340):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=height,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def price_chart(close):
    if go is None:
        st.line_chart(close)
        return
    fig = go.Figure()
    fig.add_scatter(x=close.index, y=close, name="Close", line=dict(color=TEAL, width=2))
    fig.add_scatter(x=close.index, y=close.rolling(50).mean(), name="50-day average",
                    line=dict(color=AMBER, width=1.5, dash="dot"))
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(style_figure(fig))


def sentiment_chart(news):
    daily = news.groupby("date")["score"].mean().sort_index()
    if go is None:
        st.bar_chart(daily)
        return
    fig = go.Figure()
    fig.add_bar(x=daily.index, y=daily.values, name="Average score",
                marker_color=[TEAL if v >= 0 else RED for v in daily.values])
    fig.add_hline(y=0, line_color="#64748b", line_width=1)
    st.plotly_chart(style_figure(fig))


# ---------------------------------------------------------------- analysis
def demo_stream(ticker):
    """Fake agent updates (same shape as the real graph) so the UI works without API keys."""
    rng = np.random.default_rng(sum(map(ord, ticker)))
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=250)
    prices = pd.DataFrame({"Close": 150 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(days))))}, index=days)
    news = [{"date": d.strftime("%Y-%m-%d"), "headline": "Sample headline", "source": "demo",
             "score": round(float(rng.uniform(-0.8, 0.8)), 2)} for d in days[-30:]]
    facts = {"Revenue": "demo value", "Net margin": "demo value", "P/E": "demo value"}
    yield {"fetch_data": {"ticker": ticker, "company": f"{ticker} (demo)", "prices": prices,
                          "facts": facts, "news": news}}
    time.sleep(1)
    yield {"fundamentals_agent": {"fundamentals": "- Margins look stable.\n- Valuation is above peers.\n\nView: neutral"}}
    time.sleep(1)
    yield {"sentiment_agent": {"sentiment": "- News tone is mildly positive.\n\nView: positive"}}
    time.sleep(1)
    yield {"risk_agent": {"risk": "- Valuation leaves little room for error.\n\nRisk level: medium"}}
    time.sleep(1)
    yield {"portfolio_manager": {"memo": "## Recommendation\n**HOLD**. Demo text, not a real analysis.\n"
                                         "## Investment case\n- Sample point.\n## Key risks\n- Sample risk.\n"
                                         "## Reasoning\nThis memo is fake and only shows the layout."}}
    yield {"verify": {"check": {"verified": [], "unverified": []}}}


def run_analysis(ticker, demo=False):
    """Stream the agents, show live progress, return everything needed to render the result."""
    state, messages, done = {}, [], set()
    live = st.empty()
    with live.container():
        st.subheader(f"Analyzing {ticker}")
        tracker = st.empty()
        tracker.markdown(pipeline_html(done), unsafe_allow_html=True)
        feed = st.container()
    try:
        stream = demo_stream(ticker) if demo else build_graph().stream({"ticker": ticker}, stream_mode="updates")
        for update in stream:
            for node, output in update.items():
                state.update(output)
                done.add(node)
                if node in AGENTS:
                    title, key, icon = AGENTS[node]
                    messages.append((title, icon, output[key]))
                    with feed:
                        with st.chat_message("assistant", avatar=icon):
                            st.markdown(f"**{title}**\n\n{md(output[key])}")
                tracker.markdown(pipeline_html(done), unsafe_allow_html=True)
    except Exception as error:
        live.empty()
        st.error(f"Something went wrong: {error}. Check the ticker and the keys in your .env file.")
        st.stop()
    live.empty()
    return {"state": state, "messages": messages, "done": done, "demo": demo}


# ---------------------------------------------------------------- results
def render_result(result):
    state, messages = result["state"], result["messages"]
    rating = find_rating(state)
    color = RATING_COLORS.get(rating, MUTED)

    if result.get("demo"):
        st.warning("Demo data. This is not a real analysis.")
    title_col, badge_col = st.columns([3, 1])
    with title_col:
        st.header(f"{state['company']} ({state['ticker']})")
        st.markdown(pipeline_html(result["done"]), unsafe_allow_html=True)
    with badge_col:
        st.markdown(
            f'<div class="verdict" style="border-color:{color};color:{color}">{rating}'
            f"<small>first-pass view</small></div>",
            unsafe_allow_html=True,
        )

    close = state["prices"]["Close"].dropna()
    news = pd.DataFrame(state["news"]) if state.get("news") else pd.DataFrame()
    unverified = state["check"]["unverified"]

    cards = st.columns(4)
    cards[0].metric("Last price", f"${close.iloc[-1]:,.2f}")
    cards[1].metric("Change, 12 months", f"{(close.iloc[-1] / close.iloc[0] - 1) * 100:+.1f}%")
    cards[2].metric("Average news score", f"{news['score'].mean():+.2f}" if not news.empty else "No news")
    cards[3].metric("Number check", "All verified" if not unverified else f"{len(unverified)} flagged")

    memo_tab, charts_tab, ratios_tab, agents_tab = st.tabs(["Memo", "Charts", "Ratios", "Agent notes"])

    with memo_tab:
        if unverified:
            st.warning("Numbers in the memo not found in the source data: " + ", ".join(map(str, unverified)))
        else:
            st.success("All numbers in the memo were found in the source data.")
        st.markdown(md(state["memo"]))

    with charts_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Price, last 12 months")
            price_chart(close)
        with right:
            st.subheader("News sentiment over time")
            if news.empty:
                st.info("No news. Add FINNHUB_API_KEY to the .env file to get headlines.")
            else:
                sentiment_chart(news)

    with ratios_tab:
        st.subheader("Key figures and ratios")
        facts = pd.DataFrame(state["facts"].items(), columns=["Metric", "Value"])
        facts["Value"] = facts["Value"].astype(str)
        st.dataframe(facts, hide_index=True)

    with agents_tab:
        for title, icon, text in messages:
            with st.chat_message("assistant", avatar=icon):
                st.markdown(f"**{title}**\n\n{md(text)}")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="AI Investment Memo", page_icon="📈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("ticker", "AAPL")

with st.sidebar:
    st.markdown("### Choose a company")
    st.text_input("Ticker (US-listed company)", key="ticker")
    demo = st.checkbox("Demo mode (fake data, no API keys)")
    clicked = st.button("Write memo", type="primary")
    st.caption("Or try one of these:")
    columns = st.columns(3)
    for i, name in enumerate(EXAMPLES):
        columns[i % 3].button(name, key=f"ex_{name}", on_click=queue_ticker, args=(name,))
    st.divider()
    st.caption("Educational analysis written by AI agents. Not investment advice.")

st.markdown(
    '<div class="hero"><h1>AI Investment Memo</h1>'
    "<p>Type a ticker. Four AI agents read the data, challenge each other and write a short "
    "buy, hold or sell note, then every number in it is checked against the source.</p></div>",
    unsafe_allow_html=True,
)

queued = st.session_state.pop("run", False)
ticker = st.session_state.ticker.strip().upper()
if (clicked or queued) and ticker:
    st.session_state.result = run_analysis(ticker, demo)

if "result" in st.session_state:
    render_result(st.session_state.result)
else:
    st.markdown(pipeline_html(set()), unsafe_allow_html=True)
    st.info("Pick a ticker in the sidebar and press Write memo. A first note takes about 20 seconds.")
