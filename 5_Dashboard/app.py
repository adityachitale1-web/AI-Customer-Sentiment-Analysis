"""SentiLens — AI customer sentiment intelligence (capstone core requirement 5).

Run from this folder:
    streamlit run app.py

Visitors land on a marketing home page (features + live statistics) and sign in
through a login dialog (Google or email account). Signed-in users get the
analytics dashboard: live classification via the API (3_API, port 8000), with
all data read from the SQLite database (4_Database/feedback.db).
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "4_Database"))
import db  # noqa: E402

import auth  # noqa: E402
from branding import (APP_NAME, GLOBAL_CSS, LIQUID_ETHER_HTML, TAGLINE,  # noqa: E402
                      wordmark)
from landing import render_landing  # noqa: E402

API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000")
SENTIMENT_COLORS = {"Positive": "#2eb086", "Negative": "#e05656", "Neutral": "#e8b93e"}

STOPWORDS = set("""a about after all also am an and any are as at be because been but by can
could did do does for from get got had has have he her him his how i if in into is it its
just like me my no not of on or our out she so some than that the their them then there
they this to up us was we were what when which who will with would you your http user
""".split())

st.set_page_config(page_title=f"{APP_NAME} — {TAGLINE}", page_icon="🔮", layout="wide")

components.html(LIQUID_ETHER_HTML, height=1)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def transparent(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ECE8FB")
    return fig


@st.cache_data(ttl=15)
def load_data() -> pd.DataFrame:
    db.init_db()
    rows = db.fetch_feedback()
    if not rows:
        # Fresh deployment (e.g. Streamlit Cloud): bootstrap the database from
        # the committed snapshot of real, model-classified feedback.
        snapshot = PROJECT_ROOT / "4_Database" / "sample_feedback.csv"
        if snapshot.exists():
            for _, r in pd.read_csv(snapshot).iterrows():
                db.insert_feedback(r["text"], r["sentiment"], float(r["confidence"]),
                                   source=r["source"], created_at=r["created_at"])
            rows = db.fetch_feedback()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = df["created_at"].dt.date
    return df


def top_words(texts: pd.Series, n: int = 12) -> pd.DataFrame:
    counter = Counter()
    for text in texts:
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        counter.update(w for w in words if w not in STOPWORDS)
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def render_dashboard(user: dict) -> None:
    # ---------------- Top bar ----------------
    bar_l, bar_m, bar_r = st.columns([3, 3, 1.4])
    with bar_l:
        st.markdown(wordmark(32), unsafe_allow_html=True)
    with bar_m:
        st.caption(f"Signed in as **{user['name']}** ({user['email']})"
                   + (" · via Google" if user.get("provider") == "google" else ""))
    with bar_r:
        if st.button("Sign out", use_container_width=True):
            auth.sign_out()
            st.rerun()

    st.title("📊 Customer Sentiment Dashboard")
    st.caption("Transformer-based sentiment classification · live API inference · SQLite-backed trends")

    # ---------------- Live analysis box ----------------
    with st.container(border=True):
        st.subheader("Analyse new feedback (real-time API)")
        col_input, col_btn = st.columns([5, 1])
        text = col_input.text_input("Customer feedback", placeholder="Type or paste customer feedback here…",
                                    label_visibility="collapsed")
        if col_btn.button("Analyse", type="primary", use_container_width=True) and text.strip():
            try:
                resp = requests.post(f"{API_URL}/predict", json={"text": text, "source": "dashboard"}, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}[result["sentiment"]]
                st.success(f"{emoji} **{result['sentiment']}** — confidence {result['confidence']:.1%} "
                           f"(model: {result['model']})")
                load_data.clear()
            except requests.exceptions.ConnectionError:
                st.error("API is not running. Start it first:  `cd 3_API && uvicorn main:app --port 8000`")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    df = load_data()
    if df.empty:
        st.info("No feedback in the database yet. Seed real data first:  "
                "`cd 4_Database && python seed_real_data.py`")
        st.stop()

    # ---------------- Filters ----------------
    with st.sidebar:
        st.markdown(wordmark(26), unsafe_allow_html=True)
        st.header("Filters")
        sentiments = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"],
                                    default=["Positive", "Neutral", "Negative"])
        sources = st.multiselect("Source", sorted(df["source"].unique()),
                                 default=sorted(df["source"].unique()))
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

    mask = df["sentiment"].isin(sentiments) & df["source"].isin(sources)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask &= (df["date"] >= date_range[0]) & (df["date"] <= date_range[1])
    view = df[mask]

    # ---------------- KPI row ----------------
    total = len(view)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total feedback", f"{total:,}")
    for col, label in ((k2, "Positive"), (k3, "Neutral"), (k4, "Negative")):
        n = int((view["sentiment"] == label).sum())
        col.metric(label, f"{n:,}", f"{n / total:.0%}" if total else "0%", delta_color="off")

    # ---------------- Charts ----------------
    c1, c2 = st.columns([2, 3])

    with c1:
        st.subheader("Sentiment distribution")
        dist = view["sentiment"].value_counts().reset_index()
        dist.columns = ["sentiment", "count"]
        fig = px.pie(dist, values="count", names="sentiment", hole=0.45,
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS)
        st.plotly_chart(transparent(fig), use_container_width=True)

    with c2:
        st.subheader("Sentiment trend over time")
        trend = view.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig = px.line(trend, x="date", y="count", color="sentiment", markers=True,
                      color_discrete_map=SENTIMENT_COLORS)
        fig.update_layout(legend_title=None)
        st.plotly_chart(transparent(fig), use_container_width=True)

    # ---------------- Issue analysis ----------------
    st.subheader("Issue analysis — most frequent words per sentiment")
    cols = st.columns(3)
    for col, sentiment in zip(cols, ["Negative", "Neutral", "Positive"]):
        with col:
            subset = view[view["sentiment"] == sentiment]
            st.markdown(f"**{sentiment}** ({len(subset)} items)")
            if subset.empty:
                st.caption("No data for current filters.")
                continue
            words = top_words(subset["text"])
            fig = px.bar(words.sort_values("count"), x="count", y="word", orientation="h",
                         color_discrete_sequence=[SENTIMENT_COLORS[sentiment]])
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title=None, xaxis_title=None)
            st.plotly_chart(transparent(fig), use_container_width=True)

    # ---------------- Recent feedback table ----------------
    st.subheader("Recent feedback")
    st.dataframe(
        view[["created_at", "text", "sentiment", "confidence", "source"]].head(50),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------- router

auth.init_users()

# Native OIDC session (when [auth] secrets are configured on the deployment)
if auth.current_user() is None and getattr(st, "user", None) is not None:
    try:
        if getattr(st.user, "is_logged_in", False):
            st.session_state["user"] = {"name": st.user.name or st.user.email,
                                        "email": st.user.email, "provider": "google"}
    except Exception:
        pass

user = auth.current_user()
if user is None:
    db.init_db()
    total = db.total_count()
    if total == 0:
        snapshot = PROJECT_ROOT / "4_Database" / "sample_feedback.csv"
        if snapshot.exists():
            total = len(pd.read_csv(snapshot))
    render_landing(total_feedback=total)
else:
    render_dashboard(user)
