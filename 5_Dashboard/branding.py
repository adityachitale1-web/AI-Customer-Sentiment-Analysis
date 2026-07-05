"""SentiLens branding: name, logo, Liquid Ether background and global CSS."""

APP_NAME = "SentiLens"
TAGLINE = "See what your customers feel."

# Lens ring with a sentiment waveform — Liquid Ether palette
LOGO_SVG = """
<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="sl-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#5227FF"/>
      <stop offset="55%" stop-color="#B19EEF"/>
      <stop offset="100%" stop-color="#FF9FFC"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="26" stroke="url(#sl-grad)" stroke-width="5" fill="rgba(82,39,255,0.10)"/>
  <path d="M15 36 H22 L26 27 L32 44 L38 29 L42 36 H49"
        stroke="url(#sl-grad)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def logo_html(size: int = 40) -> str:
    return LOGO_SVG.format(size=size)


def wordmark(size: int = 34) -> str:
    return f"""
    <div style="display:flex;align-items:center;gap:.6rem;">
      {logo_html(size)}
      <span style="font-size:{size * 0.72}px;font-weight:800;letter-spacing:-.02em;
                   background:linear-gradient(90deg,#F2EFFD 30%,#B19EEF 70%,#FF9FFC 100%);
                   -webkit-background-clip:text;background-clip:text;color:transparent;">
        {APP_NAME}
      </span>
    </div>
    """


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
  vec3 c1=vec3(.322,.153,1.);
  vec3 c2=vec3(1.,.624,.988);
  vec3 c3=vec3(.694,.62,.937);
  vec3 col=mix(vec3(.024,.0,.063), c1, smoothstep(.15,.6,f));
  col=mix(col, c3, smoothstep(.35,.8,length(q))*.75);
  col=mix(col, c2, smoothstep(.5,.95,length(r)*f)*.65);
  col*=.35+.65*(1.-.4*dot(uv*.55,uv*.55));
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

GLOBAL_CSS = """
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
/* Buttons: gradient pill for primary */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #5227FF 0%, #7A4DFF 100%);
    border: none; border-radius: 999px; font-weight: 700;
    padding: .55rem 1.6rem;
    box-shadow: 0 4px 24px rgba(82, 39, 255, .45);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #6A3DFF 0%, #9A6DFF 100%);
    box-shadow: 0 4px 32px rgba(122, 77, 255, .65);
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(177,158,239,.45);
    border-radius: 999px; font-weight: 600;
}
/* Landing building blocks */
.sl-hero { text-align: center; padding: 3.2rem 1rem 1rem; }
.sl-hero h1 {
    font-size: clamp(2.4rem, 6vw, 4.2rem); font-weight: 800; line-height: 1.08;
    letter-spacing: -.02em; margin: 0 0 1rem;
}
.sl-grad-text {
    background: linear-gradient(90deg, #F2EFFD 20%, #B19EEF 60%, #FF9FFC 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sl-hero p { font-size: 1.15rem; color: #C9C2E8; max-width: 620px; margin: 0 auto 1.6rem; }
.sl-badge {
    display: inline-block; padding: .35rem 1rem; border-radius: 999px;
    border: 1px solid rgba(177,158,239,.4); background: rgba(82,39,255,.15);
    color: #D8CFF7; font-size: .85rem; font-weight: 600; margin-bottom: 1.3rem;
}
.sl-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem; margin: 2.2rem 0;
}
.sl-stat {
    background: rgba(10,4,28,.6); backdrop-filter: blur(8px);
    border: 1px solid rgba(177,158,239,.22); border-radius: 16px;
    padding: 1.2rem 1rem; text-align: center;
}
.sl-stat .v { font-size: 1.9rem; font-weight: 800; color: #F2EFFD; }
.sl-stat .l { font-size: .82rem; color: #A99BD6; margin-top: .25rem; }
.sl-cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem; margin: 1rem 0 2rem;
}
.sl-card {
    background: rgba(10,4,28,.6); backdrop-filter: blur(8px);
    border: 1px solid rgba(177,158,239,.22); border-radius: 16px;
    padding: 1.4rem 1.3rem; transition: transform .15s, border-color .15s;
}
.sl-card:hover { transform: translateY(-3px); border-color: rgba(255,159,252,.55); }
.sl-card .ic { font-size: 1.7rem; }
.sl-card h4 { margin: .55rem 0 .35rem; color: #F2EFFD; font-size: 1.05rem; }
.sl-card p { margin: 0; color: #B4A9DC; font-size: .9rem; line-height: 1.45; }
.sl-step { display: flex; gap: .9rem; align-items: flex-start; margin-bottom: 1rem; }
.sl-step .n {
    min-width: 2rem; height: 2rem; border-radius: 999px; display: flex;
    align-items: center; justify-content: center; font-weight: 800; color: white;
    background: linear-gradient(135deg, #5227FF, #FF9FFC);
}
.sl-step h5 { margin: 0 0 .15rem; color: #F2EFFD; font-size: 1rem; }
.sl-step p { margin: 0; color: #B4A9DC; font-size: .88rem; }
.sl-footer {
    margin-top: 2.5rem; padding: 1.6rem 0 .8rem; text-align: center;
    border-top: 1px solid rgba(177,158,239,.18); color: #8D7FBB; font-size: .85rem;
}
h2.sl-section { text-align: center; font-weight: 800; letter-spacing: -.01em; margin: 1.2rem 0 .4rem; }
p.sl-section-sub { text-align: center; color: #A99BD6; margin: 0 0 1.4rem; }
.sl-divider { display:flex; align-items:center; gap:.8rem; color:#8D7FBB; font-size:.8rem; margin:.6rem 0; }
.sl-divider::before, .sl-divider::after { content:""; flex:1; height:1px; background:rgba(177,158,239,.25); }
</style>
"""
