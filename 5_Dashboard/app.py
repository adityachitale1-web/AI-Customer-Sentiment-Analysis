"""CXSentinel — AI customer sentiment intelligence (capstone core requirement 5).

Run from this folder:
    streamlit run app.py

Structure: three tabs at the top — Home Page (product overview + statistics),
Analyze Page (single feedback or bulk file upload, login required) and History
Page (stored results, trends and issue analysis, login required). Owners
(role='admin') get an extra Admin Settings tab. Sign-in happens through a
pop-up dialog (Google or email + verification code).
"""

import io
import os
import re
import subprocess
import sys
import time
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


def _bootstrap_oidc_from_env():
    """Write .streamlit/secrets.toml from environment variables so Streamlit's
    native Google login (st.login) works on hosts that provide secrets as env
    vars rather than a file — notably Hugging Face Spaces. Runs before any code
    reads st.secrets. A real committed/local secrets.toml always wins.

    Space secrets to set: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
    (optional: OAUTH_COOKIE_SECRET, OAUTH_REDIRECT_URI). The redirect URI is
    auto-derived from the Space host when not given.
    """
    cid = os.getenv("GOOGLE_CLIENT_ID")
    csec = os.getenv("GOOGLE_CLIENT_SECRET")
    if not (cid and csec):
        return
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        return  # local/committed secrets take precedence
    space_host = os.getenv("SPACE_HOST", "")
    redirect = os.getenv("OAUTH_REDIRECT_URI") or (
        f"https://{space_host}/oauth2callback" if space_host else "")
    cookie = os.getenv("OAUTH_COOKIE_SECRET",
                       "cxsentinel-oidc-cookie-set-OAUTH_COOKIE_SECRET-to-override")
    try:
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        secrets_path.write_text(
            "[auth]\n"
            f'redirect_uri = "{redirect}"\n'
            f'cookie_secret = "{cookie}"\n\n'
            "[auth.google]\n"
            f'client_id = "{cid}"\n'
            f'client_secret = "{csec}"\n'
            'server_metadata_url = '
            '"https://accounts.google.com/.well-known/openid-configuration"\n'
        )
    except Exception:
        pass  # never block startup on a secrets-write failure


_bootstrap_oidc_from_env()

import importlib  # noqa: E402
import auth  # noqa: E402
import branding  # noqa: E402
import inference  # noqa: E402
import landing  # noqa: E402


# Streamlit Cloud redeploys rerun this script but can keep OLD versions of our
# imported modules in memory, causing AttributeError on newly-added functions.
# Detect that by source-file mtime and reload any changed module — once per
# change, process-wide (the store lives in cache_resource so it survives reruns
# but is fresh per server process). Future-proof: no per-symbol checks needed.
@st.cache_resource
def _local_module_mtimes():
    return {}


def _reload_changed_local_modules():
    store = _local_module_mtimes()
    # Dependency order: branding has no local deps, auth imports branding,
    # landing imports auth + branding — reload in that order so each sees fresh
    # dependencies. Reload on first encounter too (guarantees a reused process
    # after a redeploy picks up the new source), and whenever the file changes.
    for mod in (branding, auth, inference, landing):
        try:
            path = getattr(mod, "__file__", None)
            if not path:
                continue
            mtime = os.path.getmtime(path)
            if store.get(path) != mtime:
                importlib.reload(mod)
                store[path] = mtime
        except Exception:
            pass


_reload_changed_local_modules()

# Import names AFTER the reload check so they reflect the freshest source.
from branding import (APP_NAME, GLOBAL_CSS, TAGLINE,  # noqa: E402
                      liquid_ether_html, wordmark)
from landing import render_home  # noqa: E402

API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000")
SENTIMENT_COLORS = {"Positive": "#2eb086", "Negative": "#e05656", "Neutral": "#e8b93e"}
SENTIMENT_EMOJI = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
BULK_LIMIT = 1000
MAX_UPLOAD_MB = 10  # keep in sync with [server] maxUploadSize in .streamlit/config.toml

STOPWORDS = set("""a about after all also am an and any are as at be because been but by can
could did do does for from get got had has have he her him his how i if in into is it its
just like me my no not of on or our out she so some than that the their them then there
they this to up us was we were what when which who will with would you your http user
""".split())

st.set_page_config(page_title=f"{APP_NAME} — {TAGLINE}", page_icon="🛡️", layout="wide")


def _api_healthy(timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=timeout).ok
    except Exception:
        return False


def can_manage_api() -> bool:
    """True only where we can auto-start the API: a local deployment with the
    trained model present. Cloud (no model) or an external SENTIMENT_API_URL
    are left untouched."""
    is_local = any(h in API_URL for h in ("127.0.0.1", "localhost"))
    model_present = ((PROJECT_ROOT / "2_Model" / "distilbert-sentiment-v2").exists()
                     or (PROJECT_ROOT / "2_Model" / "distilbert-sentiment").exists())
    return is_local and model_present


@st.cache_resource(show_spinner=False)
def _api_spawn_guard() -> dict:
    # Process-wide throttle so we never spawn a flood of uvicorn workers.
    return {"last_spawn": 0.0}


def ensure_api_running(wait_secs: int = 0) -> bool:
    """Make the inference API reachable, self-healing if it ever goes down.

    Returns True if the API is healthy (now or after starting it). Spawns the
    FastAPI service in its own session (so it persists independently) when it
    isn't reachable, throttled to avoid duplicate workers, and optionally waits
    up to wait_secs for it to become healthy.
    """
    if _api_healthy():
        return True
    if not can_manage_api():
        return False
    guard = _api_spawn_guard()
    now = time.time()
    if now - guard["last_spawn"] > 20:  # don't respawn more than every 20s
        try:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"],
                cwd=str(PROJECT_ROOT / "3_API"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            guard["last_spawn"] = now
        except Exception:
            return False
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if _api_healthy():
            return True
        time.sleep(1)
    return _api_healthy()


# Start the API on first load and wait for the model so Analyze is ready.
# (Where no API can run — e.g. the hosted demo — the dashboard's built-in
# engine in inference.py handles analysis instead.)
if can_manage_api():
    ensure_api_running(wait_secs=40)

components.html(liquid_ether_html(), height=1)
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
        # Fresh deployment (e.g. Streamlit Cloud): bootstrap from the committed
        # snapshot of real, model-classified feedback.
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


# -------------------------------------------------------- emotions & risk

EMOTION_LEXICON = {
    "Anger": ["angry", "furious", "outraged", "unacceptable", "disgusted", "worst",
              "hate", "scam", "ripoff", "horrible", "awful", "terrible", "disaster",
              "rude"],
    "Frustration": ["frustrating", "frustrated", "annoying", "annoyed", "useless",
                    "waste", "impossible", "confusing", "slow", "stuck", "nightmare"],
    "Disappointment": ["disappointed", "disappointing", "expected", "mediocre",
                       "unfortunately", "poor", "sadly", "letdown", "subpar"],
    "Joy": ["love", "amazing", "fantastic", "excellent", "awesome", "delighted",
            "happy", "best", "wonderful", "incredible", "perfect", "breathtaking",
            "great", "outstanding", "brilliant"],
    "Gratitude": ["thank", "thanks", "grateful", "appreciate", "helpful", "recommend",
                  "impressed"],
}
NEGATIVE_EMOTIONS = ("Anger", "Frustration", "Disappointment")
POSITIVE_EMOTIONS = ("Joy", "Gratitude")
INTENSIFIERS = {"very", "extremely", "absolutely", "totally", "completely", "never",
                "always", "utterly", "incredibly", "beyond"}


def score_emotion(text: str, sentiment: str, confidence: float):
    """Rule-scored emotional intensity (0–100) + dominant emotion label.

    Combines the model's confidence with lexicon hits, intensifiers,
    exclamation marks and shouting caps — a transparent, explainable proxy
    for emotional arousal.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    word_set = set(words)
    candidates = (NEGATIVE_EMOTIONS if sentiment == "Negative"
                  else POSITIVE_EMOTIONS if sentiment == "Positive"
                  else tuple(EMOTION_LEXICON))
    hits = {emo: sum(w in word_set for w in EMOTION_LEXICON[emo]) for emo in candidates}
    emotion, top_hits = max(hits.items(), key=lambda kv: kv[1])
    if top_hits == 0:
        emotion = ("Displeasure" if sentiment == "Negative"
                   else "Satisfaction" if sentiment == "Positive" else "Calm")
    intensifier_n = sum(w in INTENSIFIERS for w in words)
    exclaim_n = text.count("!")
    caps_n = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", text))
    base = 45 * confidence if sentiment != "Neutral" else 15 * confidence
    intensity = int(min(100, base + 14 * top_hits + 8 * intensifier_n
                        + 6 * exclaim_n + 6 * caps_n))
    return emotion, intensity


def risk_level(sentiment: str, intensity: int, confidence: float) -> str:
    if sentiment == "Negative" and (intensity >= 65 or confidence >= 0.90):
        return "High"
    if sentiment == "Negative":
        return "Medium"
    if sentiment == "Neutral":
        return "Low"
    return "None"


def category_driver(text: str) -> str:
    lowered = set(re.findall(r"[a-zA-Z']+", text.lower()))
    best, best_hits = "General Experience", 0
    for aspect, keywords in ASPECTS.items():
        n = sum(w in lowered for w in keywords)
        if n > best_hits:
            best, best_hits = aspect, n
    return best


def csat_equivalent(sentiment: str, confidence: float) -> float:
    """Map sentiment + confidence onto a 1–5 CSAT-style scale."""
    if sentiment == "Positive":
        return round(4 + confidence, 2)          # 4–5
    if sentiment == "Negative":
        return round(2 - confidence, 2)          # 1–2
    return 3.0


def enrich_results(results: pd.DataFrame) -> pd.DataFrame:
    """Add emotion, intensity, risk, category driver and CSAT columns."""
    enriched = results.copy()
    scored = [score_emotion(t, s, c) for t, s, c in
              zip(enriched["text"], enriched["sentiment"], enriched["confidence"])]
    enriched["emotion"] = [e for e, _ in scored]
    enriched["intensity"] = [i for _, i in scored]
    enriched["risk"] = [risk_level(s, i, c) for s, i, c in
                        zip(enriched["sentiment"], enriched["intensity"],
                            enriched["confidence"])]
    enriched["driver"] = enriched["text"].map(category_driver)
    enriched["csat"] = [csat_equivalent(s, c) for s, c in
                        zip(enriched["sentiment"], enriched["confidence"])]
    return enriched


# ------------------------------------------------------------- bulk parsing

METADATA_COLUMNS = {
    # NB: not bare "time" — it would match the "Sentiment" column
    "created_at": ("date", "timestamp", "created", "datetime"),
    "channel": ("source", "channel", "platform"),
    "segment": ("location", "segment", "tier", "region", "country", "city"),
}


def extract_feedback(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded .txt or .csv into a DataFrame with a `text` column
    plus any detected metadata (created_at / channel / segment).

    Handles normal CSVs, and also files where every row is one quoted string
    containing an entire CSV record (as in the provided sample file).
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue().decode("utf-8", "ignore")

    if name.endswith(".txt"):
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
        return pd.DataFrame({"text": texts})

    df = pd.read_csv(io.StringIO(raw))
    if df.shape[1] == 1 and "," in str(df.columns[0]):
        # Whole-line-quoted CSV: strip the outer quotes, unescape, re-parse
        lines = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s == '""':
                continue
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1].replace('""', '"')
            lines.append(s)
        df = pd.read_csv(io.StringIO("\n".join(lines)), skipinitialspace=True)

    text_cols = [c for c in df.columns
                 if any(k in str(c).lower() for k in ("text", "feedback", "review",
                                                      "comment", "message"))]
    text_col = text_cols[0] if text_cols else max(
        df.columns, key=lambda c: df[c].astype(str).str.len().mean())

    out = pd.DataFrame({"text": df[text_col].astype(str).str.strip().str.strip('"')})
    for target, needles in METADATA_COLUMNS.items():
        matches = [c for c in df.columns if c != text_col
                   and any(n in str(c).lower() for n in needles)]
        if matches:
            out[target] = df[matches[0]].astype(str).str.strip()
    if "created_at" in out.columns:
        parsed = pd.to_datetime(out["created_at"], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.5:
            out["created_at"] = parsed
        else:
            out = out.drop(columns=["created_at"])

    out = out[out["text"].str.len() > 0]
    out = out[out["text"].str.lower() != "nan"].reset_index(drop=True)
    return out


def extract_texts(uploaded_file) -> list:
    """Back-compat wrapper: just the feedback texts."""
    return extract_feedback(uploaded_file)["text"].tolist()


def classify_batch(texts: list) -> pd.DataFrame:
    """Send texts to the API in chunks of 100 with a progress bar."""
    results = []
    progress = st.progress(0.0, text="Classifying…")
    for start in range(0, len(texts), 100):
        chunk = texts[start:start + 100]
        resp = requests.post(
            f"{API_URL}/predict/batch",
            json={"items": [{"text": t, "source": "bulk_upload"} for t in chunk]},
            timeout=600,
        )
        resp.raise_for_status()
        results.extend(resp.json())
        progress.progress(min((start + 100) / len(texts), 1.0),
                          text=f"Classifying… {min(start + 100, len(texts))}/{len(texts)}")
    progress.empty()
    return pd.DataFrame(results)[["text", "sentiment", "confidence"]]


def require_login(feature: str) -> bool:
    if auth.current_user():
        return True
    st.info(f"Sign In To Use The {feature}.")
    if st.button("Sign In", type="primary", key=f"login_{feature}"):
        auth.open_login("signin")
    return False


# ------------------------------------------------------ analysis insights

CONFIDENCE_BANDS = [
    (0.85, "High", "The model is very sure about this classification — safe to act on it automatically."),
    (0.60, "Moderate", "The model is reasonably sure — worth a quick human glance if the decision matters."),
    (0.00, "Low", "The model is uncertain — this item should be reviewed by a person before acting."),
]

WHAT_IT_MEANS = {
    "Negative": "This feedback expresses dissatisfaction — a complaint, a frustration or an "
                "unmet expectation. Left unaddressed, feedback like this correlates with churn "
                "and negative word of mouth.",
    "Neutral": "This feedback is informational rather than emotional — often a question, a "
               "factual statement or mixed/mild opinion. Neutral items frequently contain "
               "pre-purchase questions and feature requests worth routing to sales or product.",
    "Positive": "This feedback expresses satisfaction or delight. Positive feedback identifies "
                "what to amplify in marketing, and its authors are candidates for testimonials, "
                "reviews and referral programmes.",
}

RECOMMENDED_ACTION = {
    "Negative": "Route to the support queue and respond within 24 hours — a fast, personal "
                "reply converts a surprising share of complainers into loyal customers. Tag "
                "the root cause so recurring themes surface in the History page.",
    "Neutral": "Check whether it contains a question or feature request; route questions to "
               "sales/support and log requests for the product backlog.",
    "Positive": "Thank the customer and invite them to leave a public review or testimonial. "
                "Note what they praised — it's a proven strength to feature in marketing.",
}


def confidence_band(conf: float):
    for threshold, band, explanation in CONFIDENCE_BANDS:
        if conf >= threshold:
            return band, explanation
    return "Low", CONFIDENCE_BANDS[-1][2]


RESPONSE_TEMPLATES = {
    "Negative": ("Hi {name},\n\nThank you for telling us about your experience with "
                 "{topic} — and I'm genuinely sorry we let you down. This isn't the "
                 "standard we hold ourselves to.\n\nI've flagged your case to our team "
                 "as a priority, and we'll follow up within 24 hours with a concrete "
                 "resolution. In the meantime, if there's anything urgent, reply "
                 "directly to this message.\n\nThank you for giving us the chance to "
                 "make this right.\n\n— The Customer Care Team"),
    "Neutral": ("Hi {name},\n\nThanks for reaching out about {topic}. Happy to help — "
                "here's what you need to know: [answer their question / share the "
                "requested information].\n\nIf anything is unclear or you'd like more "
                "detail, just reply to this message.\n\n— The Customer Care Team"),
    "Positive": ("Hi {name},\n\nThank you so much for the kind words about {topic} — "
                 "feedback like yours genuinely makes our day, and I've shared it with "
                 "the team.\n\nIf you have a moment, we'd be thrilled if you left a "
                 "public review — it helps others find us.\n\n— The Customer Care Team"),
}


def highlight_sentiment_words(text: str) -> str:
    """HTML-highlight the emotional language the analysis keyed on."""
    import html as html_mod
    pos_words = set(EMOTION_LEXICON["Joy"]) | set(EMOTION_LEXICON["Gratitude"])
    neg_words = (set(EMOTION_LEXICON["Anger"]) | set(EMOTION_LEXICON["Frustration"])
                 | set(EMOTION_LEXICON["Disappointment"]))
    out = []
    for token in re.split(r"(\W+)", text):
        low = token.lower()
        esc = html_mod.escape(token)
        if low in pos_words:
            out.append(f'<span style="background:rgba(46,176,134,.35);border-radius:4px;'
                       f'padding:0 3px;">{esc}</span>')
        elif low in neg_words:
            out.append(f'<span style="background:rgba(224,86,86,.4);border-radius:4px;'
                       f'padding:0 3px;">{esc}</span>')
        elif low in INTENSIFIERS:
            out.append(f'<span style="border-bottom:2px solid #e8b93e;">{esc}</span>')
        else:
            out.append(esc)
    return "".join(out)


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 4]


@st.cache_data(ttl=600, show_spinner=False)
def sentence_sentiments(text: str) -> pd.DataFrame:
    """Per-sentence classification (ephemeral — not stored in the database)."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return pd.DataFrame()
    try:
        resp = requests.post(
            f"{API_URL}/predict/batch",
            json={"items": [{"text": s, "source": "sentence", "persist": False}
                            for s in sentences[:30]]},
            timeout=120,
        )
        resp.raise_for_status()
        rows = resp.json()
    except requests.exceptions.ConnectionError:
        # No API (e.g. hosted demo): built-in engine, no persistence
        rows = inference.predict_many(sentences[:30], source="sentence",
                                      persist=False)
    return pd.DataFrame([{"sentence": r["text"], "sentiment": r["sentiment"],
                          "confidence": r["confidence"]} for r in rows])


def matched_aspects(text: str) -> list:
    lowered = set(re.findall(r"[a-zA-Z']+", text.lower()))
    return [aspect for aspect, keywords in ASPECTS.items()
            if any(w in lowered for w in keywords)]


def render_single_result(result: dict, text: str) -> None:
    sentiment = result["sentiment"]
    conf = result["confidence"]
    band, band_note = confidence_band(conf)
    emotion, intensity = score_emotion(text, sentiment, conf)
    risk = risk_level(sentiment, intensity, conf)
    driver = category_driver(text)
    csat = csat_equivalent(sentiment, conf)
    sentences = split_sentences(text)

    banner = (st.error if sentiment == "Negative"
              else st.warning if sentiment == "Neutral" else st.success)
    banner(f"{SENTIMENT_EMOJI[sentiment]} **{sentiment}** — Confidence {conf:.1%} "
           f"({band}) · Model: {result['model']}")

    # ---- Metric cards ----
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Sentiment", sentiment)
    k2.metric("AI Confidence", f"{conf:.0%}")
    k3.metric("CSAT Equivalent", f"{csat:.1f} / 5",
              help="Sentiment + confidence mapped onto a 1–5 satisfaction scale.")
    k4.metric("Primary Emotion", emotion)
    k5.metric("Emotional Intensity", f"{intensity}/100",
              help="Blends model confidence with emotion lexicon hits, intensifiers, "
                   "exclamation marks and shouting caps.")
    k6.metric("Risk Level", risk)

    # ---- Probability chart + intensity gauge + text stats ----
    c1, c2, c3 = st.columns([2, 1.6, 1.4])
    with c1:
        st.markdown("**Class Probabilities**")
        scores = result.get("scores") or {sentiment: conf}
        sdf = pd.DataFrame({"sentiment": list(scores.keys()),
                            "probability": list(scores.values())})
        fig = px.bar(sdf, x="probability", y="sentiment", orientation="h",
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                     text=[f"{v:.1%}" for v in sdf["probability"]])
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=5, b=0), showlegend=False,
                          xaxis_range=[0, 1], xaxis_title=None, yaxis_title=None)
        st.plotly_chart(transparent(fig), use_container_width=True)
    with c2:
        st.markdown("**Emotional Intensity**")
        import plotly.graph_objects as go
        gauge_color = ("#e05656" if intensity >= 65 else
                       "#e8b93e" if intensity >= 40 else "#2eb086")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=intensity,
            gauge={"axis": {"range": [0, 100], "tickcolor": "#ECE8FB"},
                   "bar": {"color": gauge_color},
                   "bgcolor": "rgba(255,255,255,0.06)"},
            number={"font": {"color": "#ECE8FB"}}))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=5))
        st.plotly_chart(transparent(fig), use_container_width=True)
    with c3:
        st.markdown("**Text Statistics**")
        s1, s2 = st.columns(2)
        s1.metric("Words", len(text.split()))
        s2.metric("Characters", len(text))
        s1.metric("Sentences", max(len(sentences), 1))
        s2.metric("Exclamations", text.count("!"))

    # ---- Highlighted language + aspects ----
    st.markdown("**Sentiment-Bearing Language** "
                "<span style='color:#8D7FBB;font-size:.8rem'>(Green = Positive Signal, "
                "Red = Negative Signal, Underlined = Intensifier)</span>",
                unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(10,4,28,.6);border:1px solid '
                f'rgba(177,158,239,.22);border-radius:12px;padding:1rem;line-height:1.7;">'
                f'{highlight_sentiment_words(text)}</div>', unsafe_allow_html=True)
    aspects = matched_aspects(text)
    if aspects:
        badges = " ".join(
            f'<span class="sl-badge" style="margin:.2rem .2rem 0 0;">{a}</span>'
            for a in aspects)
        st.markdown(f"**Business Aspects Mentioned:** {badges}", unsafe_allow_html=True)
    else:
        st.caption(f"Business Aspect: {driver}")

    # ---- Sentence-level breakdown (paragraphs) ----
    if len(sentences) >= 2:
        st.markdown("**Sentence-By-Sentence Breakdown**")
        try:
            sent_df = sentence_sentiments(text)
            if not sent_df.empty:
                counts = sent_df["sentiment"].value_counts()
                if counts.get("Positive", 0) and counts.get("Negative", 0):
                    st.warning(f"⚠️ **Mixed Sentiment Detected** — "
                               f"{counts.get('Positive', 0)} Positive And "
                               f"{counts.get('Negative', 0)} Negative Sentences. The "
                               f"overall label reflects the dominant tone; the negative "
                               f"sentences below are the actionable complaints.")
                st.dataframe(
                    sent_df, use_container_width=True, hide_index=True,
                    column_config={
                        "sentence": st.column_config.TextColumn("Sentence", width="large"),
                        "confidence": st.column_config.ProgressColumn(
                            "AI Score", min_value=0, max_value=1, format="%.0%%"),
                    })
        except Exception:
            st.caption("Sentence-Level Breakdown Unavailable (API Not Reachable).")

    # ---- Interpretation & action ----
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("### 💡 Interpretation")
        st.markdown(f"**What This Means**  \n{WHAT_IT_MEANS[sentiment]}")
        st.markdown(f"**Confidence: {band}**  \n{band_note}")
        hist = load_data()
        if not hist.empty:
            same = 100 * (hist["sentiment"] == sentiment).mean()
            st.markdown(f"**In Context**  \n{same:.0f}% of your stored feedback shares "
                        f"this sentiment ({len(hist):,} items on record).")
    with i2:
        st.markdown("### ✅ Recommended Action")
        st.markdown(RECOMMENDED_ACTION[sentiment])
        topic = (aspects[0].lower() if aspects else "your experience")
        with st.expander("Suggested Response Template"):
            st.code(RESPONSE_TEMPLATES[sentiment].format(name="[Customer Name]",
                                                         topic=topic),
                    language=None)


def generate_insights(results: pd.DataFrame, review_threshold: float = 0.60):
    """Rule-based business insights + recommendations for a batch of results."""
    total = len(results)
    pct = {s: 100 * (results["sentiment"] == s).mean()
           for s in ("Positive", "Neutral", "Negative")}
    net = pct["Positive"] - pct["Negative"]
    avg_conf = results["confidence"].mean()
    low_share = 100 * (results["confidence"] < review_threshold).mean()

    def words_for(sentiment, n=4):
        subset = results[results["sentiment"] == sentiment]["text"]
        if len(subset) < 3:
            return []
        return top_words(subset, n)["word"].tolist()

    neg_words, pos_words = words_for("Negative"), words_for("Positive")

    insights = []
    if net >= 40:
        insights.append(f"**Overall Sentiment Is Strongly Positive** (Net Sentiment Score "
                        f"{net:+.0f}). Customers in this batch are largely happy — the "
                        f"priority is protecting and amplifying what works.")
    elif net >= 10:
        insights.append(f"**Overall Sentiment Leans Positive** (Net Sentiment Score {net:+.0f}), "
                        f"but the {pct['Negative']:.0f}% negative slice is large enough to "
                        f"matter — address it before it grows.")
    elif net >= -10:
        insights.append(f"**Sentiment Is Split** (Net Sentiment Score {net:+.0f}). Positive and "
                        f"negative voices are roughly balanced — customer experience is "
                        f"inconsistent, which usually points to an operational issue affecting "
                        f"only part of your customers.")
    else:
        insights.append(f"**Overall Sentiment Is Negative** (Net Sentiment Score {net:+.0f}). "
                        f"{pct['Negative']:.0f}% of this batch is dissatisfied — treat this as "
                        f"an active customer-experience incident.")

    if neg_words:
        insights.append(f"**Complaint Themes:** negative feedback clusters around "
                        f"*{', '.join(neg_words)}* — these words point at the likely root "
                        f"causes to investigate first.")
    if pos_words:
        insights.append(f"**Praised Strengths:** positive feedback repeatedly mentions "
                        f"*{', '.join(pos_words)}* — proven strengths to feature in marketing "
                        f"and protect in any process change.")
    if pct["Neutral"] >= 25:
        insights.append(f"**Large Neutral Segment** ({pct['Neutral']:.0f}%): neutral items often "
                        f"hide questions and pre-purchase signals — mining them is a low-cost "
                        f"revenue opportunity.")
    insights.append(f"**Model Certainty:** average confidence is {avg_conf:.0%}; "
                    f"{low_share:.0f}% of items fall below your "
                    f"{review_threshold:.0%} AI-confidence weighting and should get "
                    f"human review rather than automated handling.")

    if "risk" in results.columns:
        high_risk = int((results["risk"] == "High").sum())
        if high_risk:
            top_emotion = results.loc[results["risk"] == "High", "emotion"].mode()
            emo = top_emotion.iloc[0] if len(top_emotion) else "Anger"
            insights.append(f"**{high_risk} High-Risk Items Detected** — intensely negative "
                            f"feedback (dominant emotion: {emo}). These carry the highest "
                            f"churn and public-escalation risk; see the Urgent Alert Box.")
    if "channel" in results.columns and results["channel"].nunique() > 1:
        by_ch = results.groupby("channel")["sentiment"].apply(
            lambda s: 100 * ((s == "Positive").mean() - (s == "Negative").mean()))
        worst_ch, best_ch = by_ch.idxmin(), by_ch.idxmax()
        insights.append(f"**Channel Gap:** sentiment is weakest on **{worst_ch}** "
                        f"(Net {by_ch[worst_ch]:+.0f}) and strongest on **{best_ch}** "
                        f"(Net {by_ch[best_ch]:+.0f}) — channel-specific experience "
                        f"differences are worth investigating.")
    if "segment" in results.columns and results["segment"].nunique() > 1:
        by_seg = results.groupby("segment")["sentiment"].apply(
            lambda s: 100 * ((s == "Positive").mean() - (s == "Negative").mean()))
        worst_seg = by_seg.idxmin()
        insights.append(f"**Segment Watch:** the **{worst_seg}** segment has the lowest "
                        f"Net Sentiment ({by_seg[worst_seg]:+.0f}) — localised issues "
                        f"(logistics, language, expectations) may be at play.")

    recs = []
    if "risk" in results.columns and (results["risk"] == "High").any():
        recs.append("Triage the Urgent Alert Box first — high-risk items are where churn "
                    "and public escalations start; aim for a response within hours, not days.")
    if pct["Negative"] >= 30:
        recs.append("Open a priority review of the complaint themes above; assign an owner "
                    "and re-run this analysis after the fix ships to measure the effect.")
    if neg_words:
        recs.append(f"Search the History page for '{neg_words[0]}' to see whether this theme "
                    f"is new or a long-running issue.")
    recs.append("Respond to every negative item within 24 hours — recovery speed is the "
                "single strongest lever on churn.")
    if pos_words:
        recs.append("Invite the happiest customers (high-confidence Positive) to leave public "
                    "reviews — social proof compounds.")
    if low_share > 15:
        recs.append(f"Set up a human review queue for the {low_share:.0f}% low-confidence "
                    f"items before automating actions on this data.")
    recs.append("Track the Net Sentiment Score over time in the History page — the trend "
                "matters more than any single batch.")
    return insights, recs, net, avg_conf, low_share


# Business-aspect lexicons for aspect-based analysis
ASPECTS = {
    "Delivery & Shipping": ["delivery", "shipping", "arrived", "late", "package",
                            "courier", "tracking", "dispatch", "shipment"],
    "Product Quality": ["quality", "broken", "damaged", "defective", "durable",
                        "material", "broke", "sturdy", "flimsy"],
    "Customer Support": ["support", "service", "staff", "response", "reply",
                         "agent", "rude", "helpful", "helpdesk", "resolution"],
    "Pricing & Value": ["price", "expensive", "cheap", "value", "cost", "refund",
                        "charge", "charged", "overpriced", "worth"],
    "Website & App": ["website", "app", "checkout", "login", "page", "navigation",
                      "crash", "slow", "interface", "loading"],
    "Food & Experience": ["food", "meal", "taste", "delicious", "restaurant",
                          "menu", "fresh", "hotel", "stay", "experience"],
}


def aspect_breakdown(results: pd.DataFrame) -> pd.DataFrame:
    """Count feedback per business aspect x sentiment via keyword lexicons."""
    rows = []
    lowered = results["text"].str.lower()
    for aspect, keywords in ASPECTS.items():
        pattern = r"\b(" + "|".join(keywords) + r")\b"
        hits = lowered.str.contains(pattern, regex=True)
        subset = results[hits]
        if subset.empty:
            continue
        for sentiment in ("Negative", "Neutral", "Positive"):
            n = int((subset["sentiment"] == sentiment).sum())
            if n:
                rows.append({"aspect": aspect, "sentiment": sentiment, "count": n})
    return pd.DataFrame(rows)


def top_phrases(texts: pd.Series, n: int = 6) -> pd.DataFrame:
    """Most frequent two-word phrases (both words meaningful)."""
    counter = Counter()
    for text in texts:
        words = [w for w in re.findall(r"[a-zA-Z']{3,}", text.lower())
                 if w not in STOPWORDS]
        counter.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    common = [(p, c) for p, c in counter.most_common(n * 3) if c >= 2][:n]
    return pd.DataFrame(common, columns=["phrase", "count"])


def executive_summary(results: pd.DataFrame, net: float, avg_conf: float) -> str:
    pct = {s: 100 * (results["sentiment"] == s).mean()
           for s in ("Positive", "Neutral", "Negative")}
    mood = ("strongly positive" if net >= 40 else "moderately positive" if net >= 10
            else "mixed" if net >= -10 else "negative — action needed")
    neg_words = top_words(results[results["sentiment"] == "Negative"]["text"], 3)
    themes = (f" The main complaint themes are {', '.join(neg_words['word'])}."
              if len(neg_words) else "")
    # Compare against everything previously stored
    hist_note = ""
    hist = load_data()
    if len(hist) > len(results):
        hist_net = (100 * (hist["sentiment"] == "Positive").mean()
                    - 100 * (hist["sentiment"] == "Negative").mean())
        direction = "better than" if net > hist_net + 5 else \
                    "worse than" if net < hist_net - 5 else "in line with"
        hist_note = (f" This batch is {direction} your overall history "
                     f"(Net Sentiment {net:+.0f} vs {hist_net:+.0f} all-time).")
    return (f"Analyzed **{len(results):,} feedback items**: "
            f"**{pct['Positive']:.0f}% Positive**, {pct['Neutral']:.0f}% Neutral, "
            f"**{pct['Negative']:.0f}% Negative** — overall mood is **{mood}** "
            f"(Net Sentiment Score {net:+.0f}, average model confidence {avg_conf:.0%})."
            f"{themes}{hist_note}")


def build_report_markdown(results: pd.DataFrame, summary: str,
                          insights: list, recs: list) -> str:
    lines = [f"# {APP_NAME} — Customer Feedback Analysis Report", "",
             summary.replace("**", ""), "", "## Business Insights"]
    lines += [f"- {i.replace('**', '')}" for i in insights]
    lines += ["", "## Recommendations"]
    lines += [f"{n}. {r}" for n, r in enumerate(recs, 1)]
    if "risk" in results.columns:
        counts = results["risk"].value_counts()
        lines += ["", "## Risk Profile"]
        lines += [f"- {lvl}: {counts.get(lvl, 0)} items"
                  for lvl in ("High", "Medium", "Low", "None")]
    aspects = aspect_breakdown(results)
    if not aspects.empty:
        lines += ["", "## Aspect Breakdown (Items Mentioning Each Business Area)"]
        pivot = aspects.pivot_table(index="aspect", columns="sentiment",
                                    values="count", fill_value=0)
        for aspect, row in pivot.iterrows():
            lines.append(f"- {aspect}: " + ", ".join(f"{c} {s}" for s, c in row.items() if c))
    if "channel" in results.columns and results["channel"].nunique() > 1:
        lines += ["", "## Sentiment By Channel"]
        for ch, grp in results.groupby("channel"):
            net = 100 * ((grp["sentiment"] == "Positive").mean()
                         - (grp["sentiment"] == "Negative").mean())
            lines.append(f"- {ch}: {len(grp)} items, Net Sentiment {net:+.0f}")
    if "segment" in results.columns and results["segment"].nunique() > 1:
        lines += ["", "## Sentiment By Segment"]
        for seg, grp in results.groupby("segment"):
            net = 100 * ((grp["sentiment"] == "Positive").mean()
                         - (grp["sentiment"] == "Negative").mean())
            lines.append(f"- {seg}: {len(grp)} items, Net Sentiment {net:+.0f}")
    lines += ["", "## Urgent Alert Queue (High-Risk Negative Items)"]
    if "risk" in results.columns:
        urgent = results[results["risk"] == "High"].nlargest(10, "confidence")
    else:
        urgent = results[results["sentiment"] == "Negative"].nlargest(10, "confidence")
    lines += [f"- ({r.confidence:.0%}) {r.text}" for r in urgent.itertuples()]
    return "\n".join(lines)


def render_batch_report(results: pd.DataFrame, review_threshold: float = 0.60) -> None:
    insights, recs, net, avg_conf, low_share = generate_insights(results, review_threshold)
    pct = {s: 100 * (results["sentiment"] == s).mean()
           for s in ("Positive", "Neutral", "Negative")}
    summary = executive_summary(results, net, avg_conf)
    has_time = "created_at" in results.columns
    has_channel = "channel" in results.columns and results["channel"].nunique() > 1
    has_segment = "segment" in results.columns and results["segment"].nunique() > 1

    # ---- Executive summary ----
    st.markdown("### 📋 Executive Summary")
    st.markdown(summary)

    # ---- Metric cards ----
    velocity = "—"
    if has_time:
        valid_t = results["created_at"].dropna()
        span_hours = max((valid_t.max() - valid_t.min()).total_seconds() / 3600, 1)
        velocity = (f"{len(valid_t) / (span_hours / 24):.1f}/Day" if span_hours >= 48
                    else f"{len(valid_t) / span_hours:.1f}/Hour")
    urgent_n = int((results["risk"] == "High").sum()) if "risk" in results.columns else 0
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Feedback Volume", f"{len(results):,}")
    k2.metric("Net Sentiment Score", f"{net:+.0f}",
              help="Positive % minus Negative % (−100 to +100). Above +20 is healthy.")
    k3.metric("Avg CSAT Equivalent", f"{results['csat'].mean():.2f} / 5"
              if "csat" in results.columns else "—",
              help="Sentiment + confidence mapped onto a 1–5 satisfaction scale.")
    k4.metric("Volume Velocity", velocity,
              help="Incoming feedback rate, from timestamps in the uploaded file.")
    k5.metric("Urgent Tickets", urgent_n,
              help="High-risk items: intense negative emotion or ≥90% negative confidence.")
    k6.metric("Avg Model Confidence", f"{avg_conf:.0%}")

    # ---- Urgent alert box (always visible) ----
    if urgent_n:
        st.error(f"🚨 **Urgent Alert Box — {urgent_n} High-Risk Items Need Immediate Triage**")
        urgent = results[results["risk"] == "High"].sort_values(
            ["intensity", "confidence"], ascending=False)
        cols = [c for c in ("text", "emotion", "intensity", "confidence", "driver",
                            "channel", "segment") if c in urgent.columns]
        st.dataframe(
            urgent[cols].head(10), use_container_width=True, hide_index=True,
            column_config={
                "text": st.column_config.TextColumn("Feedback", width="large"),
                "intensity": st.column_config.ProgressColumn(
                    "Emotional Intensity", min_value=0, max_value=100, format="%d"),
                "confidence": st.column_config.ProgressColumn(
                    "Model Confidence", min_value=0, max_value=1, format="%.0%%"),
                "driver": "Category Driver",
            })
    else:
        st.success("🚨 Urgent Alert Box: No High-Risk Items In This Batch.")

    st.markdown("&nbsp;")
    sec_overview, sec_aspects, sec_emotion, sec_voice, sec_insights, sec_report = st.tabs(
        ["📊 Overview & Trends", "🏷️ Aspects & Segments", "🎭 Emotions & Risk",
         "🗣️ Voice Of The Customer", "💡 Insights & Recommendations",
         "📄 Download Report"])

    # ---------------- Overview & Trends ----------------
    with sec_overview:
        if has_time:
            st.markdown("**Volume & Sentiment Over Time**")
            valid = results.dropna(subset=["created_at"])
            span_days = (valid["created_at"].max() - valid["created_at"].min()).days
            freq = "D" if span_days >= 3 else "h"
            trend = (valid.groupby([pd.Grouper(key="created_at", freq=freq), "sentiment"])
                     .size().reset_index(name="count"))
            fig = px.area(trend, x="created_at", y="count", color="sentiment",
                          color_discrete_map=SENTIMENT_COLORS)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              legend_title=None, xaxis_title=None,
                              yaxis_title="Feedback Items")
            st.plotly_chart(transparent(fig), use_container_width=True)
        else:
            st.caption("No Timestamp Column Detected In The Upload — Time-Series "
                       "Trends Are Shown On The History Page Instead.")

        c1, c2 = st.columns(2)
        with c1:
            if has_channel:
                st.markdown("**Sentiment Distribution By Channel**")
                fig = px.sunburst(results, path=["channel", "sentiment"],
                                  color="sentiment",
                                  color_discrete_map={**SENTIMENT_COLORS,
                                                      "(?)": "#5227FF"})
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(transparent(fig), use_container_width=True)
            else:
                st.markdown("**Sentiment Distribution**")
                dist = results["sentiment"].value_counts().reset_index()
                dist.columns = ["sentiment", "count"]
                fig = px.pie(dist, values="count", names="sentiment", hole=0.55,
                             color="sentiment", color_discrete_map=SENTIMENT_COLORS)
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(transparent(fig), use_container_width=True)
        with c2:
            st.markdown("**Confidence Distribution**")
            fig = px.histogram(results, x="confidence", nbins=20, color="sentiment",
                               color_discrete_map=SENTIMENT_COLORS)
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_range=[0.3, 1.0], legend_title=None,
                              yaxis_title="Items", xaxis_title="Model Confidence")
            st.plotly_chart(transparent(fig), use_container_width=True)

        if has_channel:
            st.markdown("**Volume Share By Channel (Donut)**")
            ch = results["channel"].value_counts().reset_index()
            ch.columns = ["channel", "count"]
            fig = px.pie(ch, values="count", names="channel", hole=0.55)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(transparent(fig), use_container_width=True)

        t1, t2, t3, t4 = st.columns(4)
        wl = results["text"].str.split().str.len()
        t1.metric("Avg Length", f"{wl.mean():.0f} Words")
        t2.metric("Longest Item", f"{wl.max()} Words")
        t3.metric("High Confidence (≥85%)", f"{100 * (results['confidence'] >= 0.85).mean():.0f}%")
        t4.metric(f"Needs Human Review (<{review_threshold:.0%})",
                  f"{int((results['confidence'] < review_threshold).sum())}")

    # ---------------- Aspects & Segments ----------------
    with sec_aspects:
        st.caption("Which Business Areas Drive Each Sentiment — Detected Via Keyword "
                   "Lexicons Per Aspect.")
        aspects = aspect_breakdown(results)
        if aspects.empty:
            st.info("No Recognisable Business Aspects Found In This Batch.")
        else:
            fig = px.bar(aspects, x="count", y="aspect", color="sentiment",
                         orientation="h", barmode="group",
                         color_discrete_map=SENTIMENT_COLORS)
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title=None, xaxis_title="Feedback Items",
                              legend_title=None)
            st.plotly_chart(transparent(fig), use_container_width=True)

            neg_aspects = (aspects[aspects["sentiment"] == "Negative"]
                           .sort_values("count", ascending=False))
            if not neg_aspects.empty:
                worst = neg_aspects.iloc[0]
                st.error(f"**Biggest Problem Area: {worst['aspect']}** — "
                         f"{worst['count']} Negative Items Mention It. Start The "
                         f"Root-Cause Investigation Here.")
            pos_aspects = (aspects[aspects["sentiment"] == "Positive"]
                           .sort_values("count", ascending=False))
            if not pos_aspects.empty:
                best = pos_aspects.iloc[0]
                st.success(f"**Strongest Area: {best['aspect']}** — "
                           f"{best['count']} Positive Items Praise It.")

        st.markdown("**Top Phrases Per Sentiment**")
        cols = st.columns(3)
        for col, sentiment in zip(cols, ["Negative", "Neutral", "Positive"]):
            with col:
                st.caption(sentiment)
                phrases = top_phrases(results[results["sentiment"] == sentiment]["text"])
                if phrases.empty:
                    st.caption("No Recurring Phrases.")
                else:
                    st.dataframe(phrases, use_container_width=True, hide_index=True)

        if has_segment:
            st.markdown("**Customer Segment Sentiment** (By Location / Tier)")
            seg = (results.groupby("segment")
                   .agg(items=("sentiment", "size"),
                        net=("sentiment", lambda s: 100 * ((s == "Positive").mean()
                                                           - (s == "Negative").mean())))
                   .sort_values("items", ascending=False).head(10).reset_index())
            fig = px.bar(seg, x="net", y="segment", orientation="h", text="items",
                         color="net", color_continuous_scale=["#e05656", "#e8b93e", "#2eb086"],
                         range_color=[-100, 100])
            fig.update_traces(texttemplate="%{text} Items", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="Net Sentiment Score", yaxis_title=None,
                              coloraxis_showscale=False)
            st.plotly_chart(transparent(fig), use_container_width=True)

    # ---------------- Emotions & Risk ----------------
    with sec_emotion:
        if "emotion" not in results.columns:
            st.info("Emotion Scoring Not Available For This Batch.")
        else:
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Primary Emotions Detected**")
                emo = (results.groupby(["emotion", "sentiment"]).size()
                       .reset_index(name="count"))
                fig = px.bar(emo, x="count", y="emotion", color="sentiment",
                             orientation="h", color_discrete_map=SENTIMENT_COLORS)
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title=None, xaxis_title="Items",
                                  legend_title=None)
                st.plotly_chart(transparent(fig), use_container_width=True)
            with e2:
                st.markdown("**Risk Level Distribution**")
                risk_counts = results["risk"].value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]
                fig = px.pie(risk_counts, values="count", names="risk", hole=0.55,
                             color="risk",
                             color_discrete_map={"High": "#e05656", "Medium": "#e8934e",
                                                 "Low": "#e8b93e", "None": "#2eb086"})
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(transparent(fig), use_container_width=True)

            st.markdown("**Emotional Intensity Distribution**")
            st.caption("Intensity Blends Model Confidence With Emotion Lexicon Hits, "
                       "Intensifiers, Exclamation Marks And Shouting Caps (0–100).")
            fig = px.histogram(results, x="intensity", nbins=20, color="sentiment",
                               color_discrete_map=SENTIMENT_COLORS)
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                              legend_title=None, yaxis_title="Items",
                              xaxis_title="Emotional Intensity")
            st.plotly_chart(transparent(fig), use_container_width=True)

            st.markdown("**Negative Feedback Word Cloud**")
            neg_text = " ".join(results.loc[results["sentiment"] == "Negative", "text"])
            if len(neg_text) < 30:
                st.caption("Not Enough Negative Feedback For A Word Cloud.")
            else:
                try:
                    from wordcloud import WordCloud
                    wc = WordCloud(width=1100, height=360, mode="RGBA",
                                   background_color=None, colormap="RdPu",
                                   stopwords=STOPWORDS).generate(neg_text)
                    st.image(wc.to_array(), use_container_width=True)
                except Exception:
                    words = top_words(results.loc[results["sentiment"] == "Negative",
                                                  "text"], 15)
                    fig = px.bar(words.sort_values("count"), x="count", y="word",
                                 orientation="h",
                                 color_discrete_sequence=[SENTIMENT_COLORS["Negative"]])
                    fig.update_layout(height=340, margin=dict(l=0, r=0, t=5, b=0),
                                      yaxis_title=None, xaxis_title=None)
                    st.plotly_chart(transparent(fig), use_container_width=True)

    # ---------------- Voice of the customer ----------------
    with sec_voice:
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("#### 😠 Most Confident Negative Feedback")
            worst_items = results[results["sentiment"] == "Negative"].nlargest(5, "confidence")
            for r in worst_items.itertuples():
                st.markdown(f"> {r.text}  \n> — *Confidence {r.confidence:.0%}*")
            if worst_items.empty:
                st.caption("No Negative Items In This Batch.")
        with q2:
            st.markdown("#### 😊 Most Confident Positive Feedback")
            best_items = results[results["sentiment"] == "Positive"].nlargest(5, "confidence")
            for r in best_items.itertuples():
                st.markdown(f"> {r.text}  \n> — *Confidence {r.confidence:.0%}*")
            if best_items.empty:
                st.caption("No Positive Items In This Batch.")

        st.markdown("#### 🚨 Urgent Response Queue")
        st.caption("High-Confidence Negative Items — Respond To These First.")
        urgent = results[(results["sentiment"] == "Negative")
                         & (results["confidence"] >= 0.70)].sort_values(
            "confidence", ascending=False)
        if urgent.empty:
            st.success("Nothing Urgent — No High-Confidence Negative Items.")
        else:
            st.dataframe(urgent.head(15), use_container_width=True, hide_index=True)

    # ---------------- Insights & recommendations ----------------
    with sec_insights:
        i_col, r_col = st.columns(2)
        with i_col:
            st.markdown("### 💡 Business Insights")
            for line in insights:
                st.markdown(f"- {line}")
        with r_col:
            st.markdown("### ✅ Recommendations")
            for i, line in enumerate(recs, 1):
                st.markdown(f"{i}. {line}")

    # ---------------- Downloadable PDF report ----------------
    with sec_report:
        st.markdown("### 📄 Professional Analysis Report (PDF)")
        st.markdown(
            "A formatted, shareable document containing the **entire analysis**: "
            "executive summary, key-metrics table, numbered figures (sentiment "
            "distribution, confidence, trends, aspects, emotions), channel / "
            "segment tables, the urgent response queue, **business insights**, "
            "**recommendations**, and a methodology note.")
        if st.button("Generate PDF Report", type="primary", key="gen_pdf"):
            with st.spinner("Building The Report — Rendering Figures And Tables…"):
                from report_pdf import build_pdf
                try:
                    st.session_state["pdf_report"] = build_pdf(
                        results, summary, insights, recs,
                        aspect_breakdown(results), review_threshold,
                        batch_name=st.session_state.get("batch_name", "Uploaded Dataset"))
                except Exception as exc:
                    st.error(f"Report Generation Failed: {exc}")
        if st.session_state.get("pdf_report"):
            st.download_button("⬇️ Download PDF Report",
                               st.session_state["pdf_report"],
                               "CXSentinel_Analysis_Report.pdf", "application/pdf",
                               use_container_width=True)
            st.caption("The Raw Data (CSV) And A Markdown Version Are Available "
                       "At The Bottom Of The Page.")

    # ---- Interactive data table + exports ----
    st.markdown("### 🔎 Interactive Data Table")
    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        sel_sent = st.multiselect("Sentiment Filter", ["Positive", "Neutral", "Negative"],
                                  default=["Positive", "Neutral", "Negative"],
                                  key="tbl_sent")
    with f2:
        risk_opts = (sorted(results["risk"].unique())
                     if "risk" in results.columns else [])
        sel_risk = st.multiselect("Risk Filter", risk_opts, default=risk_opts,
                                  key="tbl_risk") if risk_opts else []
    with f3:
        search = st.text_input("Search Feedback Text", key="tbl_search",
                               placeholder="e.g. delivery, refund, crash…")

    table = results[results["sentiment"].isin(sel_sent)]
    if sel_risk:
        table = table[table["risk"].isin(sel_risk)]
    if search.strip():
        table = table[table["text"].str.contains(re.escape(search.strip()), case=False)]
    show_cols = [c for c in ("text", "sentiment", "confidence", "driver", "emotion",
                             "intensity", "risk", "csat", "channel", "segment",
                             "created_at") if c in table.columns]
    st.dataframe(
        table[show_cols], use_container_width=True, hide_index=True, height=360,
        column_config={
            "text": st.column_config.TextColumn("Feedback Snippet", width="large"),
            "confidence": st.column_config.ProgressColumn(
                "AI Score", min_value=0, max_value=1, format="%.0%%"),
            "intensity": st.column_config.ProgressColumn(
                "Intensity", min_value=0, max_value=100, format="%d"),
            "driver": "Category Driver",
            "csat": st.column_config.NumberColumn("CSAT Eq.", format="%.1f ⭐"),
        })
    st.caption(f"{len(table):,} Of {len(results):,} Items Shown.")
    d1, d2 = st.columns(2)
    d1.download_button("Download Results As CSV",
                       results.to_csv(index=False).encode(),
                       "cxsentinel_results.csv", "text/csv",
                       use_container_width=True)
    d2.download_button("Download Written Report (Markdown)",
                       build_report_markdown(results, summary, insights, recs).encode(),
                       "cxsentinel_report.md", "text/markdown",
                       use_container_width=True)


# ------------------------------------------------------------------- tabs

def render_analyze() -> None:
    st.markdown('<h2 class="sl-section sl-grad-text">Analyze Customer Feedback</h2>',
                unsafe_allow_html=True)
    if not require_login("Analyze Page"):
        return

    mode = st.radio("Input Mode", ["Single Feedback", "Bulk Upload"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Single Feedback":
        with st.container(border=True):
            st.subheader("Single Feedback — Line Or Paragraph")
            text = st.text_area("Customer Feedback",
                                placeholder="Type or paste customer feedback here — a single "
                                            "line or a whole paragraph…",
                                height=120, label_visibility="collapsed")
            if st.button("Analyze", type="primary") and text.strip():
                try:
                    with st.spinner("Analyzing…"):
                        if can_manage_api() and not _api_healthy(1):
                            ensure_api_running(wait_secs=40)
                        if _api_healthy(1):
                            resp = requests.post(f"{API_URL}/predict",
                                                 json={"text": text, "source": "dashboard"},
                                                 timeout=60)
                            resp.raise_for_status()
                            result = resp.json()
                        else:
                            # No API (e.g. hosted demo): built-in engine
                            result = inference.predict_many([text],
                                                            source="dashboard")[0]
                    st.session_state["single_result"] = (result, text)
                    load_data.clear()
                except requests.exceptions.ConnectionError:
                    result = inference.predict_many([text], source="dashboard")[0]
                    st.session_state["single_result"] = (result, text)
                    load_data.clear()
                except Exception as exc:
                    st.error(f"Prediction Failed: {exc}")

        # Persist across reruns so expanders/downloads don't wipe the analysis
        if "single_result" in st.session_state:
            res, analyzed_text = st.session_state["single_result"]
            head_l, head_r = st.columns([5, 1.2])
            head_l.markdown("### 📋 Feedback Analysis Report")
            if head_r.button("Clear Analysis", use_container_width=True):
                st.session_state.pop("single_result", None)
                st.rerun()
            else:
                render_single_result(res, analyzed_text)
    else:
        with st.container(border=True):
            st.subheader("Bulk Upload — CSV Or TXT")
            st.caption(f"**Maximum Upload Size: {MAX_UPLOAD_MB} MB Per File** · CSV: the "
                       "text column is detected automatically (e.g. Text / Feedback / "
                       "Review), with timestamps, channels and locations picked up when "
                       "present. TXT: one feedback item per line. "
                       f"Up to {BULK_LIMIT:,} items analyzed per upload.")
            uploaded = st.file_uploader("Upload Feedback File", type=["csv", "txt"],
                                        label_visibility="collapsed")
            if uploaded is not None:
                try:
                    feed = extract_feedback(uploaded)
                except Exception as exc:
                    st.error(f"Could Not Parse The File: {exc}")
                    feed = pd.DataFrame()
                if not feed.empty:
                    if len(feed) > BULK_LIMIT:
                        st.warning(f"File Has {len(feed):,} Items — Analyzing The First "
                                   f"{BULK_LIMIT:,}.")
                        feed = feed.head(BULK_LIMIT)
                    meta_found = [c for c in ("created_at", "channel", "segment")
                                  if c in feed.columns]
                    meta_note = (f" Metadata Detected: {', '.join(meta_found)} — "
                                 f"Trends, Channel And Segment Analysis Enabled."
                                 if meta_found else "")
                    st.write(f"**{len(feed):,} Feedback Items Detected.**{meta_note} Preview:")
                    st.dataframe(feed.head(5), use_container_width=True, hide_index=True)
                    if st.button(f"Analyze {len(feed):,} Items", type="primary"):
                        try:
                            texts = feed["text"].tolist()
                            with st.spinner("Analyzing…"):
                                if can_manage_api() and not _api_healthy(1):
                                    ensure_api_running(wait_secs=40)
                                if _api_healthy(1):
                                    results = classify_batch(texts)
                                else:
                                    # No API (e.g. hosted demo): built-in engine
                                    rows = inference.predict_many(
                                        texts, source="bulk_upload")
                                    results = pd.DataFrame(rows)[
                                        ["text", "sentiment", "confidence"]]
                            for col in ("created_at", "channel", "segment"):
                                if col in feed.columns:
                                    results[col] = feed[col].values[:len(results)]
                            results = enrich_results(results)
                            load_data.clear()
                            st.session_state["batch_results"] = results
                            st.session_state["batch_name"] = uploaded.name
                        except Exception as exc:
                            st.error(f"Bulk Analysis Failed: {exc}")

        # The report lives in session state so downloads, expanders and tab
        # switches don't wipe it (every widget interaction reruns the script).
        if "batch_results" in st.session_state:
            review_threshold = st.slider(
                "🎚️ AI Confidence Weighting", min_value=0.50, max_value=0.95,
                value=0.60, step=0.05,
                help="How Much Weight To Give The AI's Own Certainty: Items Below "
                     "This Confidence Are Flagged For Human Review Instead Of "
                     "Automated Handling. Raising It Makes The Analysis More "
                     "Conservative — Fewer Automated Decisions, More Human Checks. "
                     "It Recalculates The Review Counts, Insights And Report Live.")
            head_l, head_r = st.columns([5, 1.2])
            head_l.success(f"Report For **{st.session_state.get('batch_name', 'Upload')}** — "
                           f"{len(st.session_state['batch_results']):,} Items Classified And Stored.")
            if head_r.button("Clear Report", use_container_width=True):
                for key in ("batch_results", "batch_name", "pdf_report"):
                    st.session_state.pop(key, None)
                st.rerun()
            else:
                render_batch_report(st.session_state["batch_results"], review_threshold)


def render_history() -> None:
    st.markdown('<h2 class="sl-section sl-grad-text">Feedback History & Trends</h2>',
                unsafe_allow_html=True)
    if not require_login("History Page"):
        return

    df = load_data()
    if df.empty:
        st.info("No Feedback In The Database Yet — Analyze Something First.")
        return

    # ---------------- Filters ----------------
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        sentiments = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"],
                                    default=["Positive", "Neutral", "Negative"])
    with f2:
        sources = st.multiselect("Source", sorted(df["source"].unique()),
                                 default=sorted(df["source"].unique()))
    with f3:
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date Range", (min_d, max_d),
                                   min_value=min_d, max_value=max_d)

    mask = df["sentiment"].isin(sentiments) & df["source"].isin(sources)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask &= (df["date"] >= date_range[0]) & (df["date"] <= date_range[1])
    view = df[mask]

    # ---------------- KPI row ----------------
    total = len(view)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Feedback", f"{total:,}")
    for col, label in ((k2, "Positive"), (k3, "Neutral"), (k4, "Negative")):
        n = int((view["sentiment"] == label).sum())
        col.metric(label, f"{n:,}", f"{n / total:.0%}" if total else "0%", delta_color="off")

    # ---------------- Charts ----------------
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Sentiment Distribution")
        dist = view["sentiment"].value_counts().reset_index()
        dist.columns = ["sentiment", "count"]
        fig = px.pie(dist, values="count", names="sentiment", hole=0.45,
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS)
        st.plotly_chart(transparent(fig), use_container_width=True)
    with c2:
        st.subheader("Sentiment Trend Over Time")
        trend = view.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig = px.line(trend, x="date", y="count", color="sentiment", markers=True,
                      color_discrete_map=SENTIMENT_COLORS)
        fig.update_layout(legend_title=None)
        st.plotly_chart(transparent(fig), use_container_width=True)

    # ---------------- Issue analysis ----------------
    st.subheader("Issue Analysis — Most Frequent Words Per Sentiment")
    cols = st.columns(3)
    for col, sentiment in zip(cols, ["Negative", "Neutral", "Positive"]):
        with col:
            subset = view[view["sentiment"] == sentiment]
            st.markdown(f"**{sentiment}** ({len(subset)} Items)")
            if subset.empty:
                st.caption("No Data For Current Filters.")
                continue
            words = top_words(subset["text"])
            fig = px.bar(words.sort_values("count"), x="count", y="word", orientation="h",
                         color_discrete_sequence=[SENTIMENT_COLORS[sentiment]])
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title=None, xaxis_title=None)
            st.plotly_chart(transparent(fig), use_container_width=True)

    # ---------------- Table + export ----------------
    st.subheader("Recent Feedback")
    st.dataframe(
        view[["created_at", "text", "sentiment", "confidence", "source"]].head(100),
        use_container_width=True, hide_index=True,
    )
    st.download_button("Download Filtered History As CSV",
                       view.drop(columns=["date"]).to_csv(index=False).encode(),
                       "cxsentinel_history.csv", "text/csv")


def render_admin(user: dict) -> None:
    st.markdown('<h2 class="sl-section sl-grad-text">Admin Settings</h2>',
                unsafe_allow_html=True)
    st.caption(f"Master Controls — Signed In As Owner ({user['email']})")

    # ---------------- Overview ----------------
    users = auth.list_users()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Registered Users", len(users))
    a2.metric("Stored Feedback", f"{db.total_count():,}")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        a3.metric("API Status", "Healthy")
        a4.metric("Active Model", health.get("model", "?"))
    except Exception:
        a3.metric("API Status", "Offline")
        a4.metric("Active Model", "—")

    st.info("Sign-Up Is Instant — Email Addresses Are Validated (Format, Domain "
            "Typos And A DNS Deliverability Check) With No OTP / Verification Step.")

    # ---------------- User management ----------------
    st.subheader("Registered Users")
    if users:
        udf = pd.DataFrame(users)
        udf["verified"] = udf["verified"].map({0: "No", 1: "Yes"})
        st.dataframe(udf, use_container_width=True, hide_index=True)

    # ---------------- Change password ----------------
    with st.expander("Change Master Password"):
        cur = st.text_input("Current Password", type="password", key="adm_cur")
        new1 = st.text_input("New Password (Min 8 Characters)", type="password", key="adm_new1")
        new2 = st.text_input("Confirm New Password", type="password", key="adm_new2")
        if st.button("Update Password", type="primary"):
            if new1 != new2:
                st.error("New Passwords Do Not Match.")
            else:
                ok, err = auth.change_password(user["email"], cur, new1)
                st.success("Password Updated.") if ok else st.error(err)

    # ---------------- Danger zone ----------------
    with st.expander("Danger Zone"):
        st.warning("These Actions Affect All Users And Cannot Be Undone.")
        confirm = st.checkbox("I Understand — Clear ALL Stored Feedback")
        if st.button("Clear All Feedback Data", disabled=not confirm):
            removed = db.delete_all()
            load_data.clear()
            st.success(f"Removed {removed:,} Feedback Rows.")
        if st.button("Restore Sample Snapshot (400+ Real Classified Items)"):
            snapshot = PROJECT_ROOT / "4_Database" / "sample_feedback.csv"
            if snapshot.exists():
                count = 0
                for _, r in pd.read_csv(snapshot).iterrows():
                    db.insert_feedback(r["text"], r["sentiment"], float(r["confidence"]),
                                       source=r["source"], created_at=r["created_at"])
                    count += 1
                load_data.clear()
                st.success(f"Restored {count:,} Rows From The Snapshot.")
            else:
                st.error("Snapshot File Not Found.")


# ---------------------------------------------------------------- router

auth.init_users()
auth.seed_master()

# Native OIDC session (when [auth] secrets are configured on the deployment).
# After Google redirects back, mirror the account locally (so it appears in the
# admin panel) and preserve the owner's admin role if the email matches.
if auth.current_user() is None and getattr(st, "user", None) is not None:
    try:
        if getattr(st.user, "is_logged_in", False):
            g_user, _ = auth.get_or_create_google_user(
                st.user.email, st.user.name or "")
            st.session_state["user"] = g_user or {
                "name": st.user.name or st.user.email,
                "email": st.user.email, "provider": "google", "role": "user"}
    except Exception:
        pass

user = auth.current_user()

# Keep the signed-in session in sync with the database (name/role can change,
# e.g. the owner account's name migration) without requiring a re-login.
if user:
    fresh = auth.get_profile(user["email"])
    if fresh:
        user = {**user, **fresh}
        st.session_state["user"] = user

# ---------------- Top navigation bar ----------------
bar_l, bar_gap, bar_r1, bar_r2 = st.columns([3.4, 2.6, 0.95, 1.25],
                                            vertical_alignment="center")
with bar_l:
    st.markdown(
        wordmark(34)
        + f'<div style="letter-spacing:.28em;font-size:.6rem;color:#9A8CC9;'
          f'text-transform:uppercase;margin:.2rem 0 0 3.1rem;">{TAGLINE.rstrip(".")}'
          f'</div>',
        unsafe_allow_html=True)
if user:
    with bar_r2:
        # Profile avatar (top right): circular initials → opens the profile modal
        if st.button(auth.initials(user["name"]), key="profile_btn",
                     help="View your profile"):
            auth.profile_dialog(user)
else:
    with bar_r1:
        if st.button("Sign In", use_container_width=True, key="nav_signin"):
            auth.open_login("signin")
    with bar_r2:
        if st.button("Get Started", type="primary", use_container_width=True,
                     key="nav_getstarted"):
            auth.open_login("signup")

# ---------------- Tabs ----------------
db.init_db()
tab_names = ["🏠 Home Page", "🔬 Analyze Page", "🕘 History Page"]
if auth.is_admin(user):
    tab_names.append("⚙️ Admin Settings")
tabs = st.tabs(tab_names)

with tabs[0]:
    total = db.total_count()
    if total == 0:
        snapshot = PROJECT_ROOT / "4_Database" / "sample_feedback.csv"
        if snapshot.exists():
            total = len(pd.read_csv(snapshot))
    render_home(total_feedback=total, signed_in=user is not None)
with tabs[1]:
    render_analyze()
with tabs[2]:
    render_history()
if auth.is_admin(user):
    with tabs[3]:
        render_admin(user)
