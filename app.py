"""Streamlit app.  Run with:  python -m streamlit run app.py"""
import pandas as pd
import streamlit as st

from agents import build_graph

AGENTS = {  # node name -> (title, state key)
    "fundamentals_agent": ("Fundamentals agent", "fundamentals"),
    "sentiment_agent": ("Sentiment agent", "sentiment"),
    "risk_agent": ("Risk agent", "risk"),
}


def md(text):
    """Escape $ so Streamlit does not render prices as LaTeX."""
    return text.replace("$", "\\$")


st.set_page_config(page_title="AI Investment Memo", layout="wide")
st.title("AI Investment Memo")
st.caption("A first-pass note written by a team of AI agents. Not financial advice.")

ticker = st.text_input("Ticker (US-listed company)", "AAPL").strip().upper()

if st.button("Write memo", type="primary") and ticker:
    state = {}
    try:
        with st.status("Agents at work...", expanded=True) as status:
            for update in build_graph().stream({"ticker": ticker}, stream_mode="updates"):
                for node, output in update.items():
                    state.update(output)
                    if node == "fetch_data":
                        st.write(f"Data loaded for **{state['company']}**")
                    elif node in AGENTS:
                        title, key = AGENTS[node]
                        with st.chat_message("assistant"):
                            st.markdown(f"**{title}**\n\n{md(output[key])}")
                    elif node == "portfolio_manager":
                        st.write("Portfolio manager wrote the memo")
            status.update(label="Done", state="complete", expanded=False)
    except Exception as error:
        st.error(f"Something went wrong: {error}")
        st.stop()

    st.header(f"Memo: {state['company']} ({state['ticker']})")
    st.markdown(md(state["memo"]))

    unverified = state["check"]["unverified"]
    if unverified:
        st.warning("Numbers in the memo not found in the source data: " + ", ".join(unverified))
    else:
        st.success("All numbers in the memo were found in the source data.")

    left, right = st.columns(2)
    with left:
        st.subheader("Price, last 12 months")
        st.line_chart(state["prices"]["Close"])
        st.subheader("News sentiment over time")
        if state["news"]:
            news = pd.DataFrame(state["news"])
            st.bar_chart(news.groupby("date")["score"].mean())
        else:
            st.info("No news. Add FINNHUB_API_KEY to the .env file to get headlines.")
    with right:
        st.subheader("Key figures and ratios")
        st.table(pd.DataFrame(state["facts"].items(), columns=["Metric", "Value"]).set_index("Metric"))
