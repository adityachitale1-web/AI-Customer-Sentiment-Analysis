"""Interactive sentiment dashboard (core requirement 5).

Run from this folder:
    streamlit run app.py

Reads all analysed feedback from the SQLite database (4_Database/feedback.db)
and lets you analyse new feedback live through the API (3_API, port 8000).
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

API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000")
SENTIMENT_COLORS = {"Positive": "#2eb086", "Negative": "#e05656", "Neutral": "#e8b93e"}

STOPWORDS = set("""a about after all also am an and any are as at be because been but by can
could did do does for from get got had has have he her him his how i if in into is it its
just like me my no not of on or our out she so some than that the their them then there
they this to up us was we were what when which who will with would you your http user
""".split())

st.set_page_config(page_title="Customer Sentiment Dashboard", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Liquid Ether background (ReactBits-style WebGL fluid, self-contained shader)
# palette: #5227FF / #FF9FFC / #B19EEF on #060010
# ---------------------------------------------------------------------------
LIQUID_ETHER_HTML = """
<canvas id="ether"></canvas>
<style>html,body{margin:0;padding:0;overflow:hidden;background:#060010}
#ether{position:fixed;inset:0;width:100vw;height:100vh;display:block}</style>
<script>
const canvas = document.getElementById('ether');
const gl = canvas.getContext('webgl', {antialias:false, alpha:false});
const VSH = `attribute vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
const FSH = `
precision highp float;
uniform vec2 u_res; uniform float u_t;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453123); }
float noise(vec2 p){
  vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1.,0.)),u.x),
             mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),u.x),u.y);
}
float fbm(vec2 p){
  float v=0., a=.5;
  for(int i=0;i<5;i++){ v+=a*noise(p); p=p*2.03+vec2(13.7,7.1); a*=.5; }
  return v;
}
void main(){
  vec2 uv=(gl_FragCoord.xy*2.-u_res)/min(u_res.x,u_res.y);
  float t=u_t*.12;
  vec2 drift=vec2(sin(t*.43)*.8, cos(t*.31)*.8);
  vec2 q=vec2(fbm(uv+drift+t*.5), fbm(uv-drift-t*.35));
  vec2 r=vec2(fbm(uv+3.2*q+vec2(1.7,9.2)+t*.9), fbm(uv+3.2*q+vec2(8.3,2.8)-t*.6));
  float f=fbm(uv+3.6*r);
  vec3 c1=vec3(.322,.153,1.);    // #5227FF
  vec3 c2=vec3(1.,.624,.988);    // #FF9FFC
  vec3 c3=vec3(.694,.62,.937);   // #B19EEF
  vec3 col=mix(vec3(.024,.0,.063), c1, smoothstep(.15,.6,f));
  col=mix(col, c3, smoothstep(.35,.8,length(q))*.75);
  col=mix(col, c2, smoothstep(.5,.95,length(r)*f)*.65);
  col*=.35+.65*(1.-.4*dot(uv*.55,uv*.55));  // dim + vignette so UI stays readable
  gl_FragColor=vec4(col,1.);
}`;
function shader(type, src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); return s; }
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VSH));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FSH));
gl.linkProgram(prog); gl.useProgram(prog);
const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
const loc = gl.getAttribLocation(prog,'p');
gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
const uRes = gl.getUniformLocation(prog,'u_res'), uT = gl.getUniformLocation(prog,'u_t');
function resize(){
  const dpr = Math.min(window.devicePixelRatio||1, 1.5);
  canvas.width = innerWidth*dpr; canvas.height = innerHeight*dpr;
  gl.viewport(0,0,canvas.width,canvas.height);
}
addEventListener('resize', resize); resize();
function frame(ms){
  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1f(uT, ms*0.001);
  gl.drawArrays(gl.TRIANGLES,0,3);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
"""

def liquid_ether_background() -> None:
    components.html(LIQUID_ETHER_HTML, height=1)
    st.markdown("""
    <style>
    .stApp { background: #060010; }
    /* Pin the WebGL iframe as a fixed, non-interactive backdrop */
    iframe[title="st.components.v1.html"], iframe[title="st.iframe"] {
        position: fixed; inset: 0; width: 100vw; height: 100vh;
        z-index: 0; border: none; pointer-events: none;
    }
    div[data-testid="stElementContainer"]:has(> iframe[title="st.components.v1.html"]),
    div[data-testid="stElementContainer"]:has(> iframe[title="st.iframe"]),
    div[data-testid="element-container"]:has(> iframe[title="st.iframe"]) {
        height: 0; min-height: 0;
    }
    /* Lift the app content above the backdrop and let it show through */
    [data-testid="stAppViewContainer"] { position: relative; z-index: 1; background: transparent; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: rgba(13, 6, 34, 0.88); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(10, 4, 28, 0.55); backdrop-filter: blur(6px);
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


def transparent(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ECE8FB")
    return fig


liquid_ether_background()


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


st.title("📊 AI Customer Sentiment Analysis Dashboard")
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
    st.info("No feedback in the database yet. Seed sample data first:  "
            "`cd 4_Database && python seed_sample_data.py`")
    st.stop()

# ---------------- Filters ----------------
with st.sidebar:
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
