"""CXSentinel home page: hero, statistics, features, how-it-works, footer."""

import json
from pathlib import Path

import streamlit as st

from auth import open_login
from branding import APP_NAME, TAGLINE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_V2_PATH = PROJECT_ROOT / "6_Report" / "metrics_v2.json"
METRICS_PATH = PROJECT_ROOT / "6_Report" / "metrics.json"

FEATURES = [
    ("📄", "Smart Text Ingestion",
     "Analyze a single review or bulk-upload thousands of feedback items in CSV "
     "or TXT from any customer channel."),
    ("🧠", "Transformer Classification",
     "Fine-tuned DistilBERT with a TF-IDF baseline delivers accurate, "
     "context-aware sentiment scoring on every item."),
    ("📊", "Visual Analytics",
     "Interactive distribution, trend, channel, and emotion charts surface "
     "actionable insight at a glance."),
    ("🎯", "Business Insights",
     "Auto-generated insights, recommendations, and an urgent-response queue "
     "flag your most at-risk customers first."),
    ("🏷️", "Auto-Detected Metadata",
     "Automatically extracts channel, location, timestamp, and the category "
     "driver behind each piece of feedback."),
    ("📑", "CSV & PDF Export",
     "One-click download of classified results and a professionally formatted "
     "analysis report for your team."),
]

STEPS = [
    ("Connect", "Send feedback from any channel to the CXSentinel API — one call per "
                "item, batches of hundreds, or a file upload."),
    ("Classify", "The transformer model labels each item Positive, Neutral or Negative "
                 "with a confidence score."),
    ("Visualise", "Dashboards update live: distribution, trends over time and the words "
                  "driving each sentiment."),
    ("Act", "Route complaints to support, questions to sales, and ship fixes for the "
            "issues customers actually mention."),
]


def _load_model_stats() -> dict:
    # Prefer v2 (trained on tweets + Amazon/Yelp reviews), fall back to v1
    try:
        m = json.loads(METRICS_V2_PATH.read_text())
        best = m.get("yelp_test") or m.get("tweet_eval_test")
        n = sum(v.get("n", 0) for k, v in m.items() if isinstance(v, dict) and "n" in v)
        return {"accuracy": f"{best['accuracy'] * 100:.0f}%", "test_size": f"{n:,}",
                "train_size": f"{m.get('train_size', 28000):,}"}
    except Exception:
        pass
    try:
        m = json.loads(METRICS_PATH.read_text())
        return {"accuracy": f"{m['distilbert']['accuracy'] * 100:.0f}%",
                "test_size": f"{m['test_size']:,}", "train_size": "12,000"}
    except Exception:
        return {"accuracy": "68%", "test_size": "12,284", "train_size": "12,000"}


def render_home(total_feedback: int, signed_in: bool) -> None:
    # ---- Hero ----
    st.markdown(f"""
    <div class="sl-hero">
      <div class="sl-badge">✦ AI-Powered Customer Sentiment Intelligence</div>
      <h1 class="sl-grad-text">{APP_NAME}</h1>
      <div class="sl-tagline">{TAGLINE}</div>
      <p>Upload customer feedback, or paste a single review, and let our
      transformer AI classify sentiment with precision. Automatically detect
      emotion, score confidence, and surface actionable insights and
      recommendations — all in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    if not signed_in:
        _, cb1, cb2, _ = st.columns([2.4, 1.2, 1.0, 2.4])
        with cb1:
            if st.button("Start Analyzing  →", type="primary",
                         use_container_width=True, key="hero_start"):
                open_login("signup")
        with cb2:
            if st.button("View History", use_container_width=True, key="hero_history"):
                open_login("signin")

    # ---- Stats band ----
    model_stats = _load_model_stats()
    st.markdown(f"""
    <div class="sl-stats">
      <div class="sl-stat"><div class="v">{model_stats['train_size']}</div><div class="l">Texts Trained On</div></div>
      <div class="sl-stat"><div class="v">2</div><div class="l">AI Models</div></div>
      <div class="sl-stat"><div class="v">&lt;50 ms</div><div class="l">Per Prediction</div></div>
      <div class="sl-stat"><div class="v">{model_stats['accuracy']}</div><div class="l">Model Accuracy</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Features ----
    st.markdown('<h2 class="sl-section sl-grad-text">Everything You Need</h2>',
                unsafe_allow_html=True)
    st.markdown('<p class="sl-section-sub">End-to-end customer sentiment analysis '
                'powered by transformer AI, real-time scoring, and interactive '
                'analytics.</p>', unsafe_allow_html=True)
    cards = "".join(
        f'<div class="sl-card"><div class="ic">{icon}</div><h4>{title}</h4><p>{body}</p></div>'
        for icon, title, body in FEATURES
    )
    st.markdown(f'<div class="sl-cards">{cards}</div>', unsafe_allow_html=True)

    # ---- How it works ----
    st.markdown('<h2 class="sl-section sl-grad-text">How It Works</h2>', unsafe_allow_html=True)
    st.markdown('<p class="sl-section-sub">Four Steps From Feedback Chaos To Clarity.</p>',
                unsafe_allow_html=True)
    left, right = st.columns(2)
    for i, (title, body) in enumerate(STEPS):
        with (left if i % 2 == 0 else right):
            st.markdown(f"""
            <div class="sl-step">
              <div class="n">{i + 1}</div>
              <div><h5>{title}</h5><p>{body}</p></div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Bottom CTA ----
    if not signed_in:
        st.markdown('<h2 class="sl-section sl-grad-text">Ready To Hear Your Customers?</h2>',
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns([2, 1.2, 2])
        with b2:
            if st.button("Sign In To Analyze", type="primary", use_container_width=True,
                         key="cta_bottom"):
                open_login("signin")
    else:
        st.markdown('<h2 class="sl-section sl-grad-text">You Are Signed In — '
                    'Head To The Analyze Page To Begin</h2>', unsafe_allow_html=True)

    # ---- Footer ----
    linkedin_logo = (
        '<svg viewBox="0 0 24 24" width="15" height="15" fill="#0A66C2" '
        'style="vertical-align:-2px;margin-right:5px;">'
        '<path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.13 '
        '1.45-2.13 2.94v5.67H9.35V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 '
        '3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 '
        '2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.8 0 0 '
        '.78 0 1.75v20.5C0 23.22.8 24 1.77 24h20.45c.98 0 1.78-.78 '
        '1.78-1.75V1.75C24 .78 23.2 0 22.22 0z"/></svg>')
    linkedin_url = "https://www.linkedin.com/in/aditya-chitale-84237b175/"
    st.markdown(f"""
    <div class="sl-footer">
      Built By <a href="{linkedin_url}" target="_blank" class="sl-footer-link">{linkedin_logo}Aditya Chitale</a>
      — AI Major Capstone Project, 2026
    </div>
    """, unsafe_allow_html=True)
