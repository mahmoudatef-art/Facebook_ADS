# AdPulse - Click Prediction Studio
# Run with:  streamlit run app.py

import os
import json
import pickle
import re
import time as _time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from fpdf import FPDF
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

st.set_page_config(page_title="AdPulse | Click Prediction Studio", page_icon="📈", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
T, S = "Time Spent on Site", "Salary"
FEATS = [T, S]
BLUE, GREY = "#1F4BFF", "#C3C9D6"
CMAP = {"Yes": BLUE, "No": GREY}

lang = st.sidebar.radio("Language / اللغة", ["English", "العربية"], horizontal=True)
AR = lang == "العربية"


def L(en, ar):
    return ar if AR else en


# ------------------------------------------------------------
# Look and feel — Tableau-style motion, hover, and soft welcomes
# ------------------------------------------------------------
def _anim_suffix():
    # Unique suffix so CSS animations replay every time the page re-runs
    return str(int(_time.time() * 1000) % 10_000_000)


_AS = _anim_suffix()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; color: #0E1630; }}
.stApp {{ background: #EEF1F6; }}
[data-testid="stSidebar"] {{ background: #0E1630; }}
[data-testid="stSidebar"] * {{ color: #E8ECF8 !important; }}

/* Hero */
.hero {{
  background: linear-gradient(120deg, #0E1630 0%, #1A2F86 50%, #0E1630 100%);
  background-size: 200% 100%;
  animation: drift 14s ease-in-out infinite alternate;
  border-radius: 18px; padding: 34px 38px; margin-bottom: 22px;
}}
.hero h1 {{ color: #fff !important; font-size: 2.5rem; margin: 0 0 6px 0; font-weight: 800; }}
.hero p {{ color: #AEB8D6; font-size: 1.05rem; margin: 0; max-width: 68ch; }}
@keyframes drift {{ to {{ background-position: 100% 0; }} }}

/* Soft welcome strip — gentle reveal + soft underline */
.welcome {{
  background: #fff;
  border-radius: 14px;
  padding: 16px 22px;
  margin-bottom: 18px;
  border-left: 4px solid #1F4BFF;
  box-shadow: 0 1px 0 rgba(14,22,48,.06);
  overflow: hidden;
  animation: welcomeIn{_AS} .7s cubic-bezier(.2,.8,.2,1) both;
  position: relative;
}}
.welcome .w-label {{ color: #5B6785; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 6px; }}
.welcome .w-text {{
  color: #0E1630;
  font-size: 1.05rem;
  font-weight: 500;
  line-height: 1.6;
  max-width: 72ch;
  opacity: 0;
  animation: textIn{_AS} 1.1s cubic-bezier(.2,.8,.2,1) .15s forwards;
}}
.welcome::after {{
  content: "";
  position: absolute; left: 0; bottom: 0; height: 2px; width: 0;
  background: linear-gradient(90deg, #1F4BFF, transparent);
  animation: underFill{_AS} 1.4s cubic-bezier(.2,.8,.2,1) .2s forwards;
}}
@keyframes welcomeIn{_AS} {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes textIn{_AS} {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes underFill{_AS} {{
  from {{ width: 0; }}
  to   {{ width: 100%; }}
}}

/* KPI cards — fill & lift like Tableau */
.kpi {{
  background: #fff;
  border-radius: 14px;
  padding: 18px 20px;
  border-left: 5px solid #1F4BFF;
  transition: transform .25s ease, box-shadow .25s ease;
  animation: kpiIn{_AS} .65s cubic-bezier(.2,.8,.2,1) both;
  position: relative;
  overflow: hidden;
}}
.kpi::after {{
  content: "";
  position: absolute; left: 0; bottom: 0; height: 3px; width: 0;
  background: linear-gradient(90deg, #1F4BFF, #5B7BFF);
  animation: fillBar{_AS} 1.1s cubic-bezier(.2,.8,.2,1) .2s forwards;
}}
.kpi:hover {{
  transform: translateY(-5px);
  box-shadow: 0 14px 28px rgba(14,22,48,.14);
}}
.kpi .kl {{ color: #5B6785; font-size: .85rem; }}
.kpi .kv {{
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 2rem; font-weight: 800; color: #0E1630;
  animation: countPop{_AS} .5s ease .3s both;
}}
.kpi .ks {{ color: #5B6785; font-size: .8rem; }}
@keyframes kpiIn{_AS} {{
  from {{ opacity: 0; transform: translateY(18px) scale(.97); }}
  to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes fillBar{_AS} {{
  from {{ width: 0; }}
  to   {{ width: 100%; }}
}}
@keyframes countPop{_AS} {{
  from {{ opacity: 0; transform: scale(.85); }}
  to   {{ opacity: 1; transform: scale(1); }}
}}

/* Stagger children so columns fill one after another */
.stColumns > div:nth-child(1) .kpi {{ animation-delay: .05s; }}
.stColumns > div:nth-child(2) .kpi {{ animation-delay: .15s; }}
.stColumns > div:nth-child(3) .kpi {{ animation-delay: .25s; }}
.stColumns > div:nth-child(4) .kpi {{ animation-delay: .35s; }}

/* Notes & verdicts */
.verdict {{ border-radius: 14px; padding: 18px 22px; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.5rem; font-weight: 700;
  animation: kpiIn{_AS} .6s cubic-bezier(.2,.8,.2,1) both; }}
.yes {{ background: #DCE5FF; color: #0F2FB3; }}
.no  {{ background: #F6DFD6; color: #9A3412; }}
.note {{
  background: #fff; border-radius: 12px; padding: 14px 18px; color: #25304F; margin-bottom: 10px;
  animation: kpiIn{_AS} .55s cubic-bezier(.2,.8,.2,1) both;
  transition: box-shadow .2s, transform .2s;
}}
.note:hover {{ box-shadow: 0 8px 18px rgba(14,22,48,.08); transform: translateY(-2px); }}

/* Chart containers — soft glow on hover, smooth appear */
[data-testid="stPlotlyChart"], .js-plotly-plot {{
  border-radius: 14px;
  overflow: hidden;
  background: #fff;
  box-shadow: 0 1px 0 rgba(14,22,48,.05);
  transition: box-shadow .3s ease, transform .3s ease;
  animation: chartIn{_AS} .75s cubic-bezier(.2,.8,.2,1) both;
}}
[data-testid="stPlotlyChart"]:hover, .js-plotly-plot:hover {{
  box-shadow: 0 12px 32px rgba(31,75,255,.12);
  transform: translateY(-3px);
}}
@keyframes chartIn{_AS} {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
  animation: kpiIn{_AS} .6s cubic-bezier(.2,.8,.2,1) both;
  border-radius: 12px;
  overflow: hidden;
}}

/* Tabs & general blocks */
.stTabs [data-baseweb="tab-panel"] {{ animation: kpiIn{_AS} .5s ease both; }}

@media (prefers-reduced-motion: reduce) {{
  .hero, .welcome, .welcome .w-text, .welcome::after, .kpi, .kpi::after, .kpi .kv,
  .verdict, .note, [data-testid="stPlotlyChart"], .js-plotly-plot {{
    animation: none !important;
  }}
  .welcome .w-text {{ opacity: 1; }}
  .welcome::after, .kpi::after {{ width: 100%; }}
  .kpi {{ transition: none; }}
}}
</style>
""", unsafe_allow_html=True)

if AR:
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
html, body, [class*="css"], h1, h2, h3, .kpi .kv, .welcome .w-text {{ font-family: 'Cairo', sans-serif !important; }}
.block-container, [data-testid="stSidebar"] {{ direction: rtl; text-align: right; }}
.welcome {{ border-left: none; border-right: 4px solid #1F4BFF; }}
.js-plotly-plot, .plotly {{ direction: ltr; }}
</style>""", unsafe_allow_html=True)


def welcome(page_key: str):
    """Typewriter welcome that writes character by character in front of the user."""
    msgs = {
        "Dashboard": (
            "Welcome. Your campaign pulse is ready — hover any chart, click a column, and watch the story unfold.",
            "أهلاً بيك. نبض الحملة جاهز — مرّر على أي رسمة، اضغط على عمود، وشوف القصة تتفتح قدامك."
        ),
        "Overview": (
            "Good to see you. Here is a calm view of who clicks and why — take your time exploring the audience.",
            "منور. هنا نظرة هادية على مين بيدوس وليه — خد وقتك واستكشف الجمهور براحتك."
        ),
        "Explorer": (
            "Feel free to dig in. Every distribution and segment is interactive — hover to reveal the numbers that matter.",
            "اتفضل استكشف. كل توزيع وشريحة تفاعلية — مرّر عشان تشوف الأرقام اللي تهمك."
        ),
        "Model Lab": (
            "The models are trained and waiting. Compare them side by side — every metric reacts when you hover.",
            "الموديلات متدربة ومستنية. قارن بينهم جنب بعض — كل مقياس بيتحرك لما تمرّر."
        ),
        "Predictor": (
            "Try any visitor. Move the sliders and the probability will answer you in real time — no pressure, just insight.",
            "جرّب أي زائر. حرّك السلايدرز والاحتمال هيرد عليك فوراً — من غير ضغط، بس رؤية واضحة."
        ),
        "Campaign Simulator": (
            "Set a budget and an audience. Watch expected clicks and return fill in as you refine the plan.",
            "حط ميزانية وجمهور. شوف الضغطات المتوقعة والعائد وهم بيتملوا وأنت بتعدّل الخطة."
        ),
        "Batch Scoring": (
            "Upload a list and we will score every row for you. Soft, fast, and ready to download.",
            "ارفع قائمة وهنقيّم كل صف ليك. ناعم، سريع، وجاهز للتحميل."
        ),
        "Insights": (
            "A quiet summary of what the data is telling you. Download the report whenever you are ready.",
            "ملخص هادي لما بتقوله الداتا. نزّل التقرير لما تكون جاهز."
        ),
    }
    en, ar = msgs.get(page_key, (
        "Welcome. Everything is prepared for you — explore at your own pace.",
        "أهلاً بيك. كل حاجة جاهزة ليك — استكشف على مهلك."
    ))
    text = ar if AR else en
    label = "رسالة ترحيب" if AR else "A soft welcome"
    dir_attr = "rtl" if AR else "ltr"
    # Real typewriter via JS so the user sees letters appear one by one
    html = f"""<!doctype html><html dir="{dir_attr}"><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600&family=IBM+Plex+Sans:wght@400;500&family=Cairo:wght@400;600&display=swap');
*{{box-sizing:border-box}}
body{{margin:0;font-family:'IBM Plex Sans','Cairo',sans-serif;background:transparent;color:#0E1630}}
.box{{background:#fff;border-radius:14px;padding:16px 22px;border-{"right" if AR else "left"}:4px solid #1F4BFF;
  box-shadow:0 1px 0 rgba(14,22,48,.06);position:relative;overflow:hidden;
  animation:in .55s cubic-bezier(.2,.8,.2,1) both}}
.lbl{{color:#5B6785;font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;margin-bottom:8px}}
.txt{{font-size:1.05rem;font-weight:500;line-height:1.65;min-height:2.6em;max-width:72ch}}
.cur{{display:inline-block;width:2px;height:1.1em;background:#1F4BFF;margin-inline-start:2px;
  vertical-align:-2px;animation:blink .75s step-end infinite}}
.bar{{position:absolute;left:0;bottom:0;height:2px;width:0;background:linear-gradient(90deg,#1F4BFF,transparent);
  animation:fill 1.6s cubic-bezier(.2,.8,.2,1) .15s forwards}}
@keyframes in{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes blink{{50%{{opacity:0}}}}
@keyframes fill{{to{{width:100%}}}}
@media (prefers-reduced-motion:reduce){{.box,.bar,.cur{{animation:none}}.bar{{width:100%}}.cur{{display:none}}}}
</style></head><body>
<div class="box"><div class="lbl">{label}</div><div class="txt" id="t"></div><div class="bar"></div></div>
<script>
const full={json.dumps(text, ensure_ascii=False)};
const el=document.getElementById('t');
let i=0;
const ms=Math.max(18, Math.min(42, 2200/Math.max(full.length,1)));
function tick(){{
  if(i<=full.length){{
    el.innerHTML=full.slice(0,i)+'<span class="cur"></span>';
    i++;
    if(i<=full.length) setTimeout(tick, ms);
    else el.innerHTML=full;
  }}
}}
setTimeout(tick, 280);
</script></body></html>"""
    import streamlit.components.v1 as components
    components.html(html, height=110, scrolling=False)


def _parse_kpi_value(value):
    """Extract numeric target and display prefix/suffix for count-up."""
    s = str(value).strip()
    if s in ("-", ""):
        return None, s, "", 0
    # Keep original string for non-numeric fallback
    import re as _re
    m = _re.match(r'^([^0-9\-+]*)([+\-]?[0-9][0-9,]*(?:\.[0-9]+)?)(.*)$', s)
    if not m:
        return None, s, "", 0
    pre, num, suf = m.group(1), m.group(2), m.group(3)
    try:
        target = float(num.replace(",", ""))
    except ValueError:
        return None, s, "", 0
    decimals = len(num.split(".")[1]) if "." in num else 0
    return target, pre, suf, decimals


def kpi(label, value, sub=""):
    """Tableau-style KPI with count-up number and filling bar."""
    target, pre, suf, decimals = _parse_kpi_value(value)
    uid = f"k{_anim_suffix()}{abs(hash((label, str(value)))) % 10000}"
    if target is None:
        st.markdown(
            f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{value}</div>'
            f'<div class="ks">{sub}</div></div>',
            unsafe_allow_html=True,
        )
        return
    # Live count-up via small HTML component
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@800&family=IBM+Plex+Sans:wght@400;500&family=Cairo:wght@400;700&display=swap');
*{{box-sizing:border-box}}
body{{margin:0;font-family:'IBM Plex Sans','Cairo',sans-serif;background:transparent}}
.kpi{{background:#fff;border-radius:14px;padding:16px 18px;border-left:5px solid #1F4BFF;
  position:relative;overflow:hidden;height:100%;
  animation:in .65s cubic-bezier(.2,.8,.2,1) both;
  transition:transform .25s,box-shadow .25s}}
.kpi:hover{{transform:translateY(-4px);box-shadow:0 12px 26px rgba(14,22,48,.12)}}
.kl{{color:#5B6785;font-size:.85rem;margin-bottom:2px}}
.kv{{font-family:'Bricolage Grotesque','Cairo',sans-serif;font-size:1.85rem;font-weight:800;color:#0E1630;line-height:1.2}}
.ks{{color:#5B6785;font-size:.78rem;margin-top:4px}}
.fill{{position:absolute;left:0;bottom:0;height:3px;width:0;
  background:linear-gradient(90deg,#1F4BFF,#5B7BFF);
  animation:bar 1.15s cubic-bezier(.2,.8,.2,1) .15s forwards}}
@keyframes in{{from{{opacity:0;transform:translateY(14px) scale(.97)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}
@keyframes bar{{to{{width:100%}}}}
@media (prefers-reduced-motion:reduce){{.kpi,.fill{{animation:none}}.fill{{width:100%}}}}
</style></head><body>
<div class="kpi">
  <div class="kl">{label}</div>
  <div class="kv" id="{uid}">{pre}0{suf}</div>
  <div class="ks">{sub}</div>
  <div class="fill"></div>
</div>
<script>
(function(){{
  const el=document.getElementById('{uid}');
  const to={target}, pre={json.dumps(pre)}, suf={json.dumps(suf)}, d={decimals};
  const t0=performance.now(), dur=1100;
  const fmt=v=>pre+v.toLocaleString(undefined,{{minimumFractionDigits:d,maximumFractionDigits:d}})+suf;
  function frame(t){{
    const p=Math.min(1,(t-t0)/dur);
    const e=1-Math.pow(1-p,3);
    el.textContent=fmt(to*e);
    if(p<1) requestAnimationFrame(frame);
    else el.textContent=fmt(to);
  }}
  requestAnimationFrame(frame);
}})();
</script></body></html>"""
    import streamlit.components.v1 as components
    components.html(html, height=118, scrolling=False)


def kpi_row(items):
    """Render a full row of animated KPIs side by side (Dashboard-style).
    items: list of (label, value, sub)
    """
    n = max(len(items), 1)
    cols = st.columns(n)
    for col, (label, value, sub) in zip(cols, items):
        with col:
            kpi(label, value, sub)


def animated_bars(labels, values, title="", y_label="", height=360, horizontal=False,
                  value_suffix="", value_decimals=1):
    """Column/bar chart that fills from zero in front of the user (Chart.js, Dashboard-style)."""
    import streamlit.components.v1 as components
    labels = [str(x) for x in labels]
    values = [float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
              for v in values]
    dir_attr = "rtl" if AR else "ltr"
    data_json = json.dumps({"labels": labels, "values": values, "title": title,
                            "yLabel": y_label, "horizontal": horizontal,
                            "suffix": value_suffix, "decimals": value_decimals},
                           ensure_ascii=False)
    uid = f"ab{_anim_suffix()}{abs(hash(tuple(labels) + tuple(values))) % 99999}"
    html = f"""<!doctype html><html dir="{dir_attr}"><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700;800&family=IBM+Plex+Sans:wght@400;500&family=Cairo:wght@400;700&display=swap');
*{{box-sizing:border-box}}
body{{margin:0;font-family:'IBM Plex Sans','Cairo',sans-serif;background:transparent;color:#0E1630}}
.wrap{{background:#fff;border-radius:16px;padding:16px 18px 12px;box-shadow:0 1px 0 rgba(14,22,48,.06);
  animation:in .55s cubic-bezier(.2,.8,.2,1) both;transition:transform .25s,box-shadow .25s}}
.wrap:hover{{transform:translateY(-3px);box-shadow:0 12px 28px rgba(14,22,48,.12)}}
h4{{margin:0 0 10px;font:700 1rem 'Bricolage Grotesque','Cairo',sans-serif;color:#0E1630}}
.cv{{position:relative;height:{height - 70}px}}
@keyframes in{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
@media (prefers-reduced-motion:reduce){{.wrap{{animation:none}}}}
</style></head><body>
<div class="wrap"><h4 id="ttl"></h4><div class="cv"><canvas id="{uid}"></canvas></div></div>
<script>
const D={data_json};
document.getElementById('ttl').textContent=D.title||'';
const BLUE='#1F4BFF', AMB='#FFB020', INK='#0E1630';
Chart.defaults.font.family="'IBM Plex Sans','Cairo',sans-serif";
Chart.defaults.color='#5B6785';
const fmt=v=>Number(v).toLocaleString(undefined,{{minimumFractionDigits:D.decimals,maximumFractionDigits:D.decimals}})+D.suffix;
const isH=!!D.horizontal;
new Chart(document.getElementById('{uid}'),{{
  type:'bar',
  data:{{
    labels:D.labels,
    datasets:[{{
      data:D.values,
      backgroundColor:BLUE,
      hoverBackgroundColor:AMB,
      borderRadius:8,
      borderSkipped:false,
      maxBarThickness:isH?28:64
    }}]
  }},
  options:{{
    indexAxis:isH?'y':'x',
    responsive:true,
    maintainAspectRatio:false,
    animation:{{
      duration:1200,
      easing:'easeOutCubic',
      delay:ctx=>ctx.type==='data'&&ctx.mode==='default'?ctx.dataIndex*90:0
    }},
    plugins:{{
      legend:{{display:false}},
      tooltip:{{
        backgroundColor:INK,padding:10,cornerRadius:10,displayColors:false,
        callbacks:{{
          title:i=>i[0].label,
          label:c=>fmt(c.parsed[isH?'x':'y'])
        }}
      }}
    }},
    scales:{{
      x:{{
        beginAtZero:true,
        grid:{{display:isH,color:'#EEF1F6'}},
        ticks:isH?{{callback:v=>fmt(v)}}:{{maxRotation:45,minRotation:0}},
        title:D.yLabel&&isH?{{display:true,text:D.yLabel}}:undefined
      }},
      y:{{
        beginAtZero:true,
        grid:{{display:!isH,color:'#EEF1F6'}},
        ticks:!isH?{{callback:v=>fmt(v)}}:{{}},
        title:D.yLabel&&!isH?{{display:true,text:D.yLabel}}:undefined
      }}
    }}
  }}
}});
</script></body></html>"""
    components.html(html, height=height, scrolling=False)


def style(fig, h=420):
    """Polished Plotly style with rich hover and smooth feel."""
    fig.update_layout(
        template="plotly_white",
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fff",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="IBM Plex Sans, Cairo, sans-serif", color="#0E1630"),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#0E1630",
            font_size=13,
            font_family="IBM Plex Sans, Cairo, sans-serif",
            font_color="#E8ECF8",
            bordercolor="#1F4BFF",
        ),
        transition=dict(duration=800, easing="cubic-in-out"),
        bargap=0.22,
    )
    # Soften markers and make bars feel more interactive on hover
    for tr in fig.data:
        t = getattr(tr, "type", None)
        if t == "scatter":
            tr.update(marker=dict(
                line=dict(width=0.6, color="white"),
                opacity=0.88,
            ))
            if not getattr(tr, "hovertemplate", None):
                tr.hovertemplate = "%{x}<br>%{y}<extra></extra>"
        elif t in ("bar", "histogram"):
            try:
                tr.update(marker=dict(line=dict(width=0), opacity=0.92, cornerradius=7))
            except Exception:
                tr.update(marker=dict(line=dict(width=0), opacity=0.92))
            if not getattr(tr, "hovertemplate", None):
                tr.hovertemplate = "<b>%{x}</b><br>%{y}<extra></extra>"
        elif t == "contour":
            if not getattr(tr, "hovertemplate", None):
                tr.hovertemplate = "Time: %{x}<br>Salary: %{y}<br>Click %: %{z:.1f}<extra></extra>"
    return fig


PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "adpulse_chart"},
}


# ------------------------------------------------------------
# Data + models
# ------------------------------------------------------------
@st.cache_data
def load_data():
    d = pd.read_csv(os.path.join(BASE, "facebook_ads.csv"), encoding="ISO-8859-1")
    d["Clicked Text"] = d["Clicked"].map({0: "No", 1: "Yes"})
    return d


@st.cache_resource(show_spinner="Training the model lab...")
def build_lab(_df):
    X, y = _df[FEATS], _df["Clicked"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    cands = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    pairs, rows, roc, probs = {}, [], {}, {}
    for name, m in cands.items():
        m.fit(Xtr_s, ytr)
        p = m.predict_proba(Xte_s)[:, 1]
        pred = (p >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, p)
        roc[name], probs[name], pairs[name] = (fpr, tpr), pred, (m, sc)
        rows.append({"Model": name,
                     "Train Accuracy": accuracy_score(ytr, m.predict(Xtr_s)),
                     "Test Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred), "Recall": recall_score(yte, pred),
                     "F1": f1_score(yte, pred), "ROC AUC": roc_auc_score(yte, p)})
    metrics = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    return pairs, metrics, roc, probs, (Xte_s, yte.values), sc


def proba(pair, X):
    m, s = pair
    return m.predict_proba(s.transform(pd.DataFrame(X, columns=FEATS)))[:, 1]


if not os.path.exists(os.path.join(BASE, "facebook_ads.csv")):
    st.error("facebook_ads.csv not found. Put it in the same folder as app.py.")
    st.stop()

df = load_data()
pairs, metrics, roc, preds, (Xte_s, yte), sc = build_lab(df)

# Optional: the model saved by the notebook shows up as an extra choice
nb_path, sc_path = os.path.join(BASE, "best_model.pkl"), os.path.join(BASE, "scaler.pkl")
if os.path.exists(nb_path) and os.path.exists(sc_path):
    try:
        pairs = {"Notebook best model": (pickle.load(open(nb_path, "rb")),
                                         pickle.load(open(sc_path, "rb"))), **pairs}
    except Exception:
        pass

tmin, tmax = int(df[T].min()), int(df[T].max())
smin, smax = int(df[S].min()), int(df[S].max())

# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
NAV = {"Dashboard": ("Dashboard", "لوحة المتابعة"), "Overview": ("Overview", "نظرة عامة"), "Explorer": ("Explorer", "الاستكشاف"),
       "Model Lab": ("Model Lab", "معمل الموديلات"), "Predictor": ("Predictor", "التوقّع"),
       "Campaign Simulator": ("Campaign Simulator", "محاكي الحملات"),
       "Batch Scoring": ("Batch Scoring", "تقييم دفعة"), "Insights": ("Insights & Report", "التقرير والرؤى")}
st.sidebar.title("AdPulse")
st.sidebar.caption(L("Click Prediction Studio", "استوديو توقّع الضغطات"))
page = st.sidebar.radio(L("Go to", "اذهب إلى"), list(NAV), format_func=lambda k: NAV[k][1 if AR else 0])
st.sidebar.divider()
st.sidebar.caption(L(f"{len(df):,} users · {len(pairs)} models loaded",
                     f"{len(df):,} مستخدم · {len(pairs)} موديلات"))

# ------------------------------------------------------------
# Dashboard (Tableau-style)
# ------------------------------------------------------------
DASH_CSS = """<style>
.vc{background:#fff;border-radius:16px;padding:16px 18px 8px;box-shadow:0 1px 0 rgba(14,22,48,.06);transition:transform .2s,box-shadow .2s}
.vc:hover{transform:translateY(-4px);box-shadow:0 12px 26px rgba(14,22,48,.14)}
.vl{color:#5B6785;font-size:.85rem}
.vv{font-family:'Bricolage Grotesque',sans-serif;font-size:2.1rem;font-weight:800;color:#0E1630;line-height:1.15}
.vb{display:inline-block;font-size:.78rem;font-weight:600;border-radius:20px;padding:2px 10px;margin:2px 0 4px}
.vb.up{background:#DCE5FF;color:#0F2FB3}.vb.down{background:#F6DFD6;color:#9A3412}
.ln{fill:none;stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.4s ease-out .15s forwards}
.ar{opacity:0;animation:fadein .8s ease-out 1s forwards}
.dt{opacity:0;animation:fadein .3s ease-out 1.5s forwards}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes fadein{to{opacity:1}}
@property --num{syntax:'<integer>';initial-value:0;inherits:false}
.cnt::after{content:attr(data-v)}
@supports (animation-timeline: scroll()){
 .cnt{animation:cnt 1.3s ease-out forwards;counter-reset:num var(--num)}
 .cnt::after{content:counter(num)}
}
@keyframes cnt{from{--num:0}to{--num:var(--n)}}
@media (prefers-reduced-motion: reduce){
 .ln,.ar,.dt{animation:none;opacity:1;stroke-dashoffset:0}
 .cnt{animation:none}.cnt::after{content:attr(data-v) !important}
}
.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:#3DDC97;margin-right:8px;animation:pulse 1.8s ease-out infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(61,220,151,.7)}100%{box-shadow:0 0 0 12px rgba(61,220,151,0)}}
.chip{display:inline-block;background:rgba(255,255,255,.1);color:#DDE6FF;border-radius:20px;padding:4px 14px;font-size:.82rem;margin-bottom:14px}
.rg{stroke-dashoffset:var(--o);animation:ringin 1.5s cubic-bezier(.2,.8,.2,1)}
@keyframes ringin{from{stroke-dashoffset:var(--c)}}
.bars{display:flex;align-items:flex-end;gap:6px;padding-top:22px}
.bc{flex:1;height:var(--h);background:linear-gradient(180deg,#5B7BFF,#1F4BFF);border-radius:8px 8px 0 0;position:relative;transform-origin:bottom;animation:rise .9s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d)}
.bc.top{background:linear-gradient(180deg,#FFB020,#E8890C)}
.bc span{position:absolute;top:-18px;width:100%;text-align:center;font-size:.7rem;color:#5B6785}
@keyframes rise{from{transform:scaleY(0)}}
.bx{color:#5B6785;font-size:.78rem;margin-top:6px}
.vc2{background:#fff;border-radius:16px;padding:16px 18px;height:100%;box-shadow:0 1px 0 rgba(14,22,48,.06)}
.vc2 h4{font-family:'Bricolage Grotesque',sans-serif;margin:0 0 6px;color:#0E1630;font-size:1rem}
.mrow{margin:10px 0}.mrow .mt{height:10px;background:#E3E7F1;border-radius:6px;overflow:hidden}
.mrow .mt i{display:block;height:100%;width:var(--w);background:linear-gradient(90deg,#1F4BFF,#5B7BFF);border-radius:6px;animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d)}
@keyframes grow{from{width:0}}
.mrow .ml{display:flex;justify-content:space-between;font-size:.82rem;color:#25304F;margin-bottom:3px}
@media (prefers-reduced-motion: reduce){.rg,.bc,.mrow .mt i,.pulse{animation:none}}
</style>"""


def ring(pct, color, sub="", size=170):
    r = 54
    c = 2 * np.pi * r
    return (f'<svg viewBox="0 0 140 140" width="{size}" style="display:block;margin:auto">'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="#E3E7F1" stroke-width="14"/>'
            f'<circle class="rg" cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="14" '
            f'stroke-linecap="round" stroke-dasharray="{c:.1f}" style="--c:{c:.1f};--o:{c * (1 - pct / 100):.1f}" '
            f'transform="rotate(-90 70 70)"/>'
            f'<text x="70" y="69" text-anchor="middle" font-size="26" font-weight="800" fill="#0E1630" '
            f'font-family="Bricolage Grotesque,sans-serif">{pct:.1f}%</text>'
            f'<text x="70" y="88" text-anchor="middle" font-size="10" fill="#5B6785">{sub}</text></svg>')


def bars_html(vals, first, last):
    vals = list(np.nan_to_num(vals))
    m = max(vals) or 1
    top = int(np.argmax(vals))
    cols = "".join(f'<div class="bc{" top" if i == top else ""}" style="--h:{v / m * 100:.0f}%;--d:{i * 70}ms">'
                   f'<span>{v:.0f}</span></div>' for i, v in enumerate(vals))
    return f'<div class="bars" style="height:190px">{cols}</div><div class="bx">{first} → {last}</div>'


def meters_html(rows):
    out = ""
    for i, (lab, v) in enumerate(rows):
        out += (f'<div class="mrow"><div class="ml"><span>{lab}</span><b>{v:.0f}%</b></div>'
                f'<div class="mt"><i style="--w:{v:.0f}%;--d:{i * 120}ms"></i></div></div>')
    return out


def spark(vals, uid, color=BLUE, w=240, h=60):
    v = np.nan_to_num(np.asarray(vals, float))
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1.0
    x = np.linspace(4, w - 4, len(v))
    y = (h - 8) - (v - lo) / span * (h - 16)
    d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    area = d + f" L{x[-1]:.1f},{h} L{x[0]:.1f},{h} Z"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%"><defs><linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".28"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#g{uid})" class="ar"/>'
            f'<path d="{d}" class="ln" pathLength="1" stroke="{color}"/>'
            f'<circle cx="{x[-1]:.1f}" cy="{y[-1]:.1f}" r="4" fill="{color}" class="dt"/></svg>')


def vcard(label, value_html, badge, good, series, uid):
    return (f'<div class="vc"><div class="vl">{label}</div><div class="vv">{value_html}</div>'
            f'<div class="vb {"up" if good else "down"}">{"▲" if good else "▼"} {badge}</div>'
            f'{spark(series, uid, BLUE if good else "#D9622B")}</div>')


def counter(n):
    n = int(n)
    return f'<span class="cnt" style="--n:{n}" data-v="{n:,}"></span>'


def band_stats(d, col, edges):
    g = d.groupby(pd.cut(d[col], edges, include_lowest=True), observed=False)
    out = pd.DataFrame({"users": g.size(), "clicks": g["Clicked"].sum(), "rate": g["Clicked"].mean() * 100,
                        "avg_s": g[S].mean()}).reset_index(drop=True)
    out[["rate", "avg_s"]] = out[["rate", "avg_s"]].ffill().bfill().fillna(0)
    out["mid"] = (edges[:-1] + edges[1:]) / 2
    return out


BOARD_HTML = r"""<!doctype html><html dir="__DIR__"><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700;800&family=IBM+Plex+Sans:wght@400;500&family=Cairo:wght@400;700&display=swap');
*{box-sizing:border-box}body{margin:0;font-family:'IBM Plex Sans','Cairo',sans-serif;color:#0E1630;background:transparent}
.bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.chip{background:#0E1630;color:#DDE6FF;border-radius:20px;padding:5px 14px;font-size:.82rem}
.chip i{display:inline-block;width:9px;height:9px;border-radius:50%;background:#3DDC97;margin-inline-end:8px;animation:pl 1.8s infinite}
@keyframes pl{0%{box-shadow:0 0 0 0 rgba(61,220,151,.7)}100%{box-shadow:0 0 0 12px rgba(61,220,151,0)}}
button{font:inherit;cursor:pointer;border:0;border-radius:20px;padding:6px 14px;background:#1F4BFF;color:#fff;transition:transform .15s,background .15s}
button:hover{transform:scale(1.06);background:#0F2FB3}button.ghost{background:#E3E7F1;color:#0E1630}
.grid{display:grid;gap:14px;margin-bottom:14px}.k{grid-template-columns:repeat(4,1fr)}.r2{grid-template-columns:1.5fr 1fr}
.card{background:#fff;border-radius:16px;padding:16px 18px;box-shadow:0 1px 0 rgba(14,22,48,.06);transition:transform .25s,box-shadow .25s}
.card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(14,22,48,.14)}
h4{margin:0 0 10px;font:700 1rem 'Bricolage Grotesque','Cairo',sans-serif}.hd{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.cv{position:relative;height:270px}.sp{position:relative;height:60px}
.kl{color:#5B6785;font-size:.85rem}.kv{font:800 2.1rem 'Bricolage Grotesque',sans-serif;line-height:1.2}
.kb{display:inline-block;font-size:.78rem;font-weight:600;border-radius:20px;padding:2px 10px;margin:2px 0 6px}
.up{background:#DCE5FF;color:#0F2FB3}.dn{background:#F6DFD6;color:#9A3412}
.seg button{background:#E3E7F1;color:#25304F;padding:4px 12px;margin-inline-start:4px;font-size:.8rem}.seg button.on{background:#1F4BFF;color:#fff}
.mrow{margin:12px 0;cursor:default}.ml{display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:4px}
.mt{height:12px;background:#E3E7F1;border-radius:8px;overflow:hidden}
.mt i{display:block;height:100%;background:linear-gradient(90deg,#1F4BFF,#5B7BFF);border-radius:8px;transition:width 1.1s cubic-bezier(.2,.8,.2,1),filter .2s}
.mrow:hover .mt i{filter:brightness(1.25) saturate(1.3)}
.big{font:800 3rem 'Bricolage Grotesque',sans-serif;color:#1F4BFF;line-height:1.1}.band{color:#5B6785;margin-top:6px}
.st{display:flex;gap:18px;margin-top:10px}.st span{display:block;color:#5B6785;font-size:.78rem}.st b{font-size:1.1rem}
.hint{color:#5B6785;margin:30px 0}
@media(max-width:900px){.k{grid-template-columns:1fr 1fr}.r2{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){.chip i{animation:none}}
</style></head><body>
<div class="bar"><span class="chip"><i></i><span id="live"></span></span><button id="rp"></button></div>
<div class="grid k" id="kp" data-b="k"></div>
<div class="grid r2">
 <div class="card" data-b="col"><div class="hd"><h4 id="t1"></h4><div class="seg" id="mt"></div></div><div class="cv"><canvas id="col"></canvas></div></div>
 <div class="card" data-b="don"><h4 id="t2"></h4><div class="cv"><canvas id="don"></canvas></div></div>
</div>
<div class="grid r2">
 <div class="card" data-b="line"><h4 id="t3"></h4><div class="cv"><canvas id="line"></canvas></div></div>
 <div class="card" data-b="sal"><h4 id="t4"></h4><div class="cv"><canvas id="sal"></canvas></div></div>
</div>
<div class="grid" style="grid-template-columns:1fr 1fr">
 <div class="card" data-b="seg"><h4 id="t5"></h4><div id="seg"></div></div>
 <div class="card"><div class="hd"><h4 id="t6"></h4><button id="clr" class="ghost"></button></div><div id="det"></div></div>
</div>
<script>
const D=__DATA__,T=D.tx,K=D.K,$=id=>document.getElementById(id);
const BLUE='#1F4BFF',GREY='#C3C9D6',AMB='#FFB020',INK='#0E1630';
Chart.defaults.font.family="'IBM Plex Sans','Cairo',sans-serif";Chart.defaults.color='#5B6785';
let sel=null,metric='rate',CH={},built={},io=null,fired=false;
const fmt=(v,m)=>m==='rate'?(+v).toFixed(1)+'%':Math.round(v).toLocaleString();
const mk=(id,cfg)=>{if(CH[id])CH[id].destroy();return CH[id]=new Chart($(id),cfg)};
const tip=(f,u='')=>({backgroundColor:INK,padding:10,cornerRadius:10,displayColors:false,callbacks:{title:i=>i[0].label+u,label:f}});
function count(el,to,d,pre,suf){const t0=performance.now();(function f(t){const p=Math.min(1,(t-t0)/1100),e=1-Math.pow(1-p,3);
 el.textContent=pre+(to*e).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})+suf;if(p<1)requestAnimationFrame(f)})(t0)}
const colColors=()=>D.T.map((_,i)=>sel===i?AMB:(sel===null?BLUE:BLUE+'55'));
const donData=()=>sel===null?[K.clicks,K.users-K.clicks]:[D.clicks[sel],D.users[sel]-D.clicks[sel]];
const axisY=v=>metric==='rate'?v+'%':v;

function buildK(){
 $('kp').innerHTML=D.kp.map((k,i)=>`<div class="card"><div class="kl">${k.label}</div><div class="kv" id="kv${i}">${k.pre}0${k.suf}</div><div class="kb ${k.up?'up':'dn'}">${k.up?'▲':'▼'} ${k.badge}</div><div class="sp"><canvas id="sp${i}"></canvas></div></div>`).join('');
 D.kp.forEach((k,i)=>{count($('kv'+i),k.v,k.d,k.pre,k.suf);const c=k.up?BLUE:'#D9622B';
  mk('sp'+i,{type:'line',data:{labels:D.T,datasets:[{data:k.s,borderColor:c,borderWidth:2.5,tension:.4,pointRadius:0,pointHoverRadius:6,pointHoverBackgroundColor:c,fill:true,
   backgroundColor:cx=>{const a=cx.chart.chartArea;if(!a)return null;const g=cx.chart.ctx.createLinearGradient(0,a.top,0,a.bottom);g.addColorStop(0,c+'55');g.addColorStop(1,c+'00');return g}}]},
   options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},layout:{padding:{top:6}},
   plugins:{legend:{display:false},tooltip:tip(x=>x.parsed.y.toLocaleString(undefined,{maximumFractionDigits:1}),' '+T.min)},scales:{x:{display:false},y:{display:false}}}})})}

function buildCol(){$('t1').textContent=T.colTitle;
 mk('col',{type:'bar',data:{labels:D.T,datasets:[{data:D[metric],backgroundColor:colColors(),hoverBackgroundColor:AMB,borderRadius:8,borderSkipped:false}]},
 options:{responsive:true,maintainAspectRatio:false,animation:{duration:1100,delay:c=>c.type==='data'&&c.mode==='default'?c.dataIndex*60:0},
  plugins:{legend:{display:false},tooltip:tip(c=>fmt(c.parsed.y,metric),' '+T.min)},
  scales:{x:{grid:{display:false},title:{display:true,text:T.axisT}},y:{beginAtZero:true,grid:{color:'#EEF1F6'},ticks:{callback:axisY}}},
  onClick:(e,els)=>select(els.length?els[0].index:null),onHover:(e,els)=>{e.native.target.style.cursor=els.length?'pointer':'default'}}})}

const center={id:'ct',afterDraw(ch){const a=ch.chartArea,x=(a.left+a.right)/2,y=(a.top+a.bottom)/2,ds=ch.data.datasets[0].data,tot=ds[0]+ds[1]||1,
 ac=ch.getActiveElements(),i=ac.length?ac[0].index:0,c=ch.ctx;c.save();c.textAlign='center';c.fillStyle=INK;c.font="800 30px 'Bricolage Grotesque',sans-serif";
 c.fillText((ds[i]/tot*100).toFixed(1)+'%',x,y+6);c.fillStyle='#5B6785';c.font="13px 'IBM Plex Sans',sans-serif";c.fillText(ch.data.labels[i],x,y+28);c.restore()}};
function buildDon(){$('t2').textContent=T.donTitle;
 mk('don',{type:'doughnut',data:{labels:[T.clicked,T.notClicked],datasets:[{data:donData(),backgroundColor:[BLUE,GREY],borderWidth:3,borderColor:'#fff',hoverOffset:16}]},
 options:{cutout:'72%',maintainAspectRatio:false,animation:{animateRotate:true,animateScale:true,duration:1500},plugins:{legend:{position:'bottom'},tooltip:tip(c=>c.parsed.toLocaleString())}},plugins:[center]})}

function buildLine(){$('t3').textContent=T.lineTitle;const n=D.T.length,dl=1400/n,
 py=c=>c.index===0?c.chart.scales.y.getPixelForValue(0):c.chart.getDatasetMeta(c.datasetIndex).data[c.index-1].getProps(['y'],true).y;
 const cross={id:'cr',afterDatasetsDraw(ch){const a=ch.tooltip&&ch.tooltip.getActiveElements();if(a&&a.length){const x=a[0].element.x,c=ch.ctx;c.save();c.strokeStyle='#0E163355';c.setLineDash([4,4]);c.beginPath();c.moveTo(x,ch.chartArea.top);c.lineTo(x,ch.chartArea.bottom);c.stroke();c.restore()}}};
 mk('line',{type:'line',data:{labels:D.T,datasets:[{data:D[metric],borderColor:BLUE,borderWidth:3,tension:.4,fill:true,pointBackgroundColor:c=>sel===c.dataIndex?AMB:'#fff',pointBorderColor:c=>sel===c.dataIndex?AMB:BLUE,
  pointBorderWidth:2,pointRadius:c=>sel===c.dataIndex?9:4,pointHoverRadius:9,
  backgroundColor:cx=>{const a=cx.chart.chartArea;if(!a)return null;const g=cx.chart.ctx.createLinearGradient(0,a.top,0,a.bottom);g.addColorStop(0,BLUE+'44');g.addColorStop(1,BLUE+'00');return g}}]},
 options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
  animation:{x:{type:'number',easing:'linear',duration:dl,from:NaN,delay(c){if(c.type!=='data'||c.xS)return 0;c.xS=true;return c.index*dl}},
   y:{type:'number',easing:'linear',duration:dl,from:py,delay(c){if(c.type!=='data'||c.yS)return 0;c.yS=true;return c.index*dl}}},
  plugins:{legend:{display:false},tooltip:tip(c=>fmt(c.parsed.y,metric),' '+T.min)},
  scales:{x:{grid:{display:false},title:{display:true,text:T.axisT}},y:{beginAtZero:true,grid:{color:'#EEF1F6'},ticks:{callback:axisY}}},
  onClick:(e,els)=>select(els.length?els[0].index:null)},plugins:[cross]})}

function buildSal(){$('t4').textContent=T.salTitle;
 mk('sal',{type:'bar',data:{labels:D.sL,datasets:[{data:D.sRate,backgroundColor:BLUE,hoverBackgroundColor:AMB,borderRadius:8,borderSkipped:false}]},
 options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:{duration:1100,delay:c=>c.type==='data'&&c.mode==='default'?c.dataIndex*90:0},
  plugins:{legend:{display:false},tooltip:tip(c=>c.parsed.x.toFixed(1)+'%')},scales:{x:{beginAtZero:true,grid:{color:'#EEF1F6'},ticks:{callback:v=>v+'%'}},y:{grid:{display:false}}}}})}

function buildSeg(){$('t5').textContent=T.segTitle;
 $('seg').innerHTML=D.seg.map((s,i)=>`<div class="mrow"><div class="ml"><span>${s[0]}</span><b>${s[1].toFixed(0)}%</b></div><div class="mt"><i style="width:0;transition-delay:${i*130}ms"></i></div></div>`).join('');
 requestAnimationFrame(()=>requestAnimationFrame(()=>document.querySelectorAll('#seg i').forEach((e,i)=>e.style.width=D.seg[i][1]+'%')))}

function det(){const el=$('det');$('t6').textContent=T.detTitle;$('clr').textContent=T.clear;
 if(sel===null){el.innerHTML=`<p class="hint">${T.hint}</p>`;$('clr').style.visibility='hidden';return}
 $('clr').style.visibility='visible';const r=D.rate[sel],df=r-K.rate;
 el.innerHTML=`<div class="band">${D.T[sel]} ${T.min}</div><div class="big" id="dv">0%</div><div class="kb ${df>=0?'up':'dn'}">${df>=0?'▲ +':'▼ '}${df.toFixed(1)} ${T.pts}</div>
 <div class="st"><div><span>${T.usersL}</span><b>${D.users[sel].toLocaleString()}</b></div><div><span>${T.clicksL}</span><b>${D.clicks[sel].toLocaleString()}</b></div><div><span>${T.avgSal}</span><b>$${Math.round(D.sal[sel]).toLocaleString()}</b></div></div>`;
 count($('dv'),r,1,'','%')}

function select(i){sel=(i===null||i===sel)?null:i;
 if(CH.col){CH.col.data.datasets[0].backgroundColor=colColors();CH.col.update()}
 if(CH.don){CH.don.data.datasets[0].data=donData();CH.don.update()}
 if(CH.line)CH.line.update();det()}
function buildMt(){$('mt').innerHTML=[['rate',T.mRate],['users',T.mUsers],['clicks',T.mClicks]].map(m=>`<button data-m="${m[0]}" class="${m[0]===metric?'on':''}">${m[1]}</button>`).join('')}
$('mt').onclick=e=>{const m=e.target.dataset.m;if(!m)return;metric=m;buildMt();['col','line'].forEach(k=>{if(CH[k]){CH[k].data.datasets[0].data=D[m];CH[k].update()}})};
$('clr').onclick=()=>select(null);

const B={k:buildK,col:buildCol,don:buildDon,line:buildLine,sal:buildSal,seg:buildSeg};
const go=el=>{const b=el.dataset.b;if(built[b])return;built[b]=1;B[b]()};
function run(){if(io)io.disconnect();fired=false;
 io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){fired=true;go(e.target)}}),{threshold:.15});
 document.querySelectorAll('[data-b]').forEach(el=>io.observe(el));
 setTimeout(()=>{if(!fired)document.querySelectorAll('[data-b]').forEach(go)},2500)}
$('rp').onclick=()=>{Object.values(CH).forEach(c=>c.destroy());CH={};built={};sel=null;det();run()};
$('live').textContent=T.live;$('rp').textContent='↻ '+T.replay;buildMt();det();run();
</script></body></html>
"""


def embed_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def dash_css():
    # A new keyframe name on every run makes the browser replay the CSS animations
    n = int(_time.time() * 1000) % 10_000_000
    return re.sub(r"(@keyframes |animation:)(draw|fadein|ringin|rise|grow|cnt)\b",
                  lambda m: f"{m.group(1)}{m.group(2)}{n}", DASH_CSS)


if page == "Dashboard":
    welcome("Dashboard")
    st.markdown(dash_css(), unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h1>{L("Campaign dashboard", "لوحة متابعة الحملة")}</h1>'
                f'<p>{L("Hover any visual, click a column to drill down, press Replay to watch it animate again.", "مرّر على أي رسمة، واضغط على عمود عشان تشوف تفاصيله، واضغط Replay عشان الحركة تتعاد.")}</p></div>',
                unsafe_allow_html=True)
    fa, fb = st.columns(2)
    with fa:
        lo_t, hi_t = st.slider(L("Time on site (min)", "الوقت على الموقع (دقيقة)"), tmin, tmax, (tmin, tmax), key="dash_t")
    with fb:
        lo_s, hi_s = st.slider(L("Salary", "المرتب"), smin, smax, (smin, smax), step=500, key="dash_s")
    f = df[df[T].between(lo_t, hi_t) & df[S].between(lo_s, hi_s)]
    if f.empty:
        st.warning(L("No users match these filters. Widen the ranges.", "مفيش مستخدمين بالفلاتر دي. وسّع النطاق."))
        st.stop()

    rate_all, rate_f = df["Clicked"].mean() * 100, f["Clicked"].mean() * 100
    sal_pct = (f[S].mean() / df[S].mean() - 1) * 100
    ed = np.linspace(tmin, tmax, 13)
    sp = band_stats(f, T, ed)
    sed = np.linspace(smin, smax, 7)
    ss = band_stats(f, S, sed)
    qq = f.assign(tb=pd.qcut(f[T], 4, labels=False, duplicates="drop") + 1,
                  sb=pd.qcut(f[S], 4, labels=False, duplicates="drop") + 1)
    sg = qq.groupby(["tb", "sb"])["Clicked"].agg(["mean", "size"]).reset_index()
    sg = sg[sg["size"] >= 5].sort_values("mean", ascending=False).head(5)
    seg_rows = [[L(f"Time Q{int(r.tb)} + Salary Q{int(r.sb)}", f"وقت Q{int(r.tb)} + مرتب Q{int(r.sb)}"),
                 round(float(r["mean"]) * 100, 1)] for _, r in sg.iterrows()]
    n_clicks = int(f["Clicked"].sum())
    kp = [
        dict(label=L("Users", "المستخدمين"), v=len(f), d=0, pre="", suf="", up=True, s=sp["users"].round(1).tolist(),
             badge=L(f"{len(f) / len(df) * 100:.0f}% of all users", f"{len(f) / len(df) * 100:.0f}% من الكل")),
        dict(label=L("Clicks", "الضغطات"), v=n_clicks, d=0, pre="", suf="", up=True, s=sp["clicks"].round(1).tolist(),
             badge=L(f"{n_clicks / df['Clicked'].sum() * 100:.0f}% of all clicks", f"{n_clicks / df['Clicked'].sum() * 100:.0f}% من كل الضغطات")),
        dict(label=L("Click rate", "معدل الضغط"), v=round(rate_f, 1), d=1, pre="", suf="%", up=bool(rate_f >= rate_all),
             s=sp["rate"].round(1).tolist(),
             badge=L(f"{rate_f - rate_all:+.1f} pts vs overall", f"{rate_f - rate_all:+.1f} نقطة عن الإجمالي")),
        dict(label=L("Avg salary", "متوسط المرتب"), v=round(float(f[S].mean())), d=0, pre="$", suf="", up=bool(sal_pct >= 0),
             s=sp["avg_s"].round(0).tolist(), badge=L(f"{sal_pct:+.1f}% vs overall", f"{sal_pct:+.1f}% عن الإجمالي")),
    ]
    data = dict(
        T=[f"{ed[i]:.0f}-{ed[i + 1]:.0f}" for i in range(12)], users=sp["users"].tolist(), clicks=sp["clicks"].tolist(),
        rate=sp["rate"].round(1).tolist(), sal=sp["avg_s"].round(0).tolist(),
        sL=[f"${sed[i] / 1000:.0f}k-{sed[i + 1] / 1000:.0f}k" for i in range(6)], sRate=ss["rate"].round(1).tolist(),
        kp=kp, K=dict(users=len(f), clicks=n_clicks, rate=round(rate_f, 1)), seg=seg_rows,
        tx=dict(live=L(f"{len(pairs)} models online", f"{len(pairs)} موديلات شغالة"), replay=L("Replay", "إعادة الحركة"),
                colTitle=L("By time on site (click a column)", "حسب الوقت على الموقع (اضغط على عمود)"),
                donTitle=L("Share of clicks", "نسبة الضغط"), lineTitle=L("Trend line", "خط الاتجاه"),
                salTitle=L("Click rate by salary band", "معدل الضغط حسب شريحة المرتب"),
                segTitle=L("Best audience segments", "أفضل شرايح الجمهور"), detTitle=L("Drill-down", "التفاصيل"),
                hint=L("Click a column or a point on the line to see that time band here.", "اضغط على عمود أو نقطة على الخط عشان تشوف الشريحة دي هنا."),
                clear=L("Clear", "مسح"), mRate=L("Click rate", "معدل الضغط"), mUsers=L("Users", "المستخدمين"),
                mClicks=L("Clicks", "الضغطات"), clicked=L("Clicked", "داس"), notClicked=L("Did not click", "ما داسش"),
                axisT=L("Time on site (min)", "الوقت على الموقع (دقيقة)"), min=L("min", "دقيقة"),
                pts=L("pts vs overall", "نقطة عن الإجمالي"), usersL=L("Users", "المستخدمين"),
                clicksL=L("Clicks", "الضغطات"), avgSal=L("Avg salary", "متوسط المرتب")))
    html = (BOARD_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__DIR__", "rtl" if AR else "ltr"))
    embed_html(html, 1380)

    steps = np.linspace(lo_t, hi_t, 9)[1:]
    parts = [x for x in (f[f[T] <= v].assign(Step=f"≤ {v:.0f} min") for v in steps) if len(x)]
    an = px.scatter(pd.concat(parts), x=T, y=S, color="Clicked Text", color_discrete_map=CMAP, animation_frame="Step",
                    range_x=[lo_t - 1, hi_t + 1], range_y=[lo_s - 2000, hi_s + 2000], opacity=0.8,
                    title=L("Audience build-up (press Play)", "تكوّن الجمهور (اضغط Play)"))
    if an.layout.updatemenus:
        an.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
        an.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400
    st.plotly_chart(style(an, 440), width="stretch", config=PLOT_CONFIG)


# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    welcome("Overview")
    st.markdown(f'<div class="hero"><h1>{L("Who clicks your ad, and why?", "مين بيدوس على إعلانك، وليه؟")}</h1>'
                f'<p>{L("Explore the audience, compare five models, and test any visitor before you spend a dollar on them.", "استكشف الجمهور، قارن بين خمس موديلات، وجرّب أي زائر قبل ما تصرف عليه ولا دولار.")}</p></div>',
                unsafe_allow_html=True)

    lo_t, hi_t = st.slider(L("Time on site (min)", "الوقت على الموقع (دقيقة)"), tmin, tmax, (tmin, tmax))
    lo_s, hi_s = st.slider(L("Salary", "المرتب"), smin, smax, (smin, smax), step=500)
    f = df[df[T].between(lo_t, hi_t) & df[S].between(lo_s, hi_s)]

    kpi_row([
        (L("Users in view", "المستخدمين المعروضين"), f"{len(f):,}",
         L(f"of {len(df):,} total", f"من {len(df):,}")),
        (L("Click rate", "معدل الضغط"),
         f"{f['Clicked'].mean() * 100:.1f}%" if len(f) else "-",
         L(f"overall {df['Clicked'].mean() * 100:.1f}%", f"الإجمالي {df['Clicked'].mean() * 100:.1f}%")),
        (L("Avg time on site", "متوسط الوقت"),
         f"{f[T].mean():.1f}" if len(f) else "-", "min"),
        (L("Avg salary", "متوسط المرتب"),
         f"${f[S].mean():,.0f}" if len(f) else "-", ""),
    ])

    st.write("")
    yes, no = df[df.Clicked == 1], df[df.Clicked == 0]
    st.markdown(L(f'<div class="note"><b>What the data says.</b> Users who clicked spent '
                  f'{yes[T].mean():.1f} minutes on the site on average, against {no[T].mean():.1f} '
                  f'for those who did not. Their average salary was ${yes[S].mean():,.0f} versus '
                  f'${no[S].mean():,.0f}.</div>',
                  f'<div class="note"><b>الداتا بتقول إيه؟</b> اللي داسوا قضّوا في المتوسط '
                  f'{yes[T].mean():.1f} دقيقة على الموقع مقابل {no[T].mean():.1f} للي ما داسوش. '
                  f'ومتوسط مرتبهم ${yes[S].mean():,.0f} مقابل ${no[S].mean():,.0f}.</div>'),
                unsafe_allow_html=True)

    if len(f):
        st.plotly_chart(style(px.scatter(f, x=T, y=S, color="Clicked Text", color_discrete_map=CMAP,
                                         title="Your selection"), 380), width="stretch", config=PLOT_CONFIG)
    st.dataframe(f.drop(columns="Clicked Text"), width="stretch", height=260)
    st.download_button(L("Download this selection (CSV)", "نزّل الاختيار (CSV)"), f.drop(columns="Clicked Text").to_csv(index=False),
                       "selection.csv", "text/csv")

# ------------------------------------------------------------
# Explorer
# ------------------------------------------------------------
if page == "Explorer":
    welcome("Explorer")
    st.markdown(
        f'<div class="hero"><h1>{L("Explorer", "الاستكشاف")}</h1>'
        f'<p>{L("Every chart is interactive. Hover to reveal values, switch views, and find the segments that work hardest.", "كل رسمة تفاعلية. مرّر عشان تشوف القيم، غيّر العرض، واكتشف الشرايح اللي بتشتغل أقوى.")}</p></div>',
        unsafe_allow_html=True,
    )
    t1, t2, t3, t4 = st.tabs(["Distributions", "Click-rate map", "Relationships", "Segments"])

    with t1:
        col = st.selectbox("Feature", FEATS)
        kind = st.radio("View", ["Histogram", "Box", "Violin"], horizontal=True)
        if kind == "Histogram":
            counts, edges = np.histogram(df[col], bins=12)
            bin_labels = [f"{edges[i]:.0f}" for i in range(len(counts))]
            animated_bars(
                labels=bin_labels,
                values=counts.tolist(),
                title=L(f"Distribution of {col}", f"توزيع {col}"),
                y_label=L("Users", "المستخدمين"),
                height=400,
                value_decimals=0,
            )
            st.caption(L("Bars fill as the page opens. Hover any column to see the exact count.",
                         "الأعمدة بتتملي لما الصفحة تفتح. مرّر على أي عمود عشان تشوف العدد."))
        elif kind == "Box":
            fig = px.box(df, x="Clicked Text", y=col, color="Clicked Text", color_discrete_map=CMAP, points="all")
            st.plotly_chart(style(fig), width="stretch", config=PLOT_CONFIG)
        else:
            fig = px.violin(df, x="Clicked Text", y=col, color="Clicked Text", color_discrete_map=CMAP, box=True)
            st.plotly_chart(style(fig), width="stretch", config=PLOT_CONFIG)

    with t2:
        bins = st.slider("Grid size", 4, 20, 10)
        fig = px.density_heatmap(df, x=T, y=S, z="Clicked", histfunc="avg", nbinsx=bins, nbinsy=bins,
                                 color_continuous_scale="Blues", title="Average click rate per cell")
        st.plotly_chart(style(fig, 460), width="stretch", config=PLOT_CONFIG)
        st.caption("Dark cells are where most users click. Targeting those cells is where budget works hardest.")

    with t3:
        a, b = st.columns(2)
        with a:
            st.plotly_chart(style(px.scatter(df, x=T, y=S, color="Clicked Text", color_discrete_map=CMAP,
                                             marginal_x="histogram", marginal_y="histogram"), 460),
                            width="stretch", config=PLOT_CONFIG)
        with b:
            corr = df[FEATS + ["Clicked"]].corr().round(2)
            st.plotly_chart(style(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                            zmin=-1, zmax=1, title="Correlation"), 460),
                            width="stretch", config=PLOT_CONFIG)

    with t4:
        q = st.slider("Number of segments per feature", 2, 6, 4)
        seg = df.copy()
        seg["Time band"] = pd.qcut(seg[T], q, duplicates="drop").astype(str)
        seg["Salary band"] = pd.qcut(seg[S], q, duplicates="drop").astype(str)
        pv = seg.pivot_table(index="Salary band", columns="Time band", values="Clicked", aggfunc="mean")
        pv = pv.reindex(index=sorted(pv.index, key=lambda s: float(s.split(",")[0].strip("([")))[::-1],
                        columns=sorted(pv.columns, key=lambda s: float(s.split(",")[0].strip("(["))))
        st.plotly_chart(style(px.imshow((pv * 100).round(1), text_auto=True, color_continuous_scale="Blues",
                                        title="Click rate % by segment"), 460),
                        width="stretch", config=PLOT_CONFIG)

# ------------------------------------------------------------
# Model Lab
# ------------------------------------------------------------
if page == "Model Lab":
    welcome("Model Lab")
    st.markdown(
        f'<div class="hero"><h1>{L("Model Lab", "معمل الموديلات")}</h1>'
        f'<p>{L("Five models trained on the same split. Compare metrics, ROC curves, and what each model listens to.", "خمس موديلات اتدربوا على نفس التقسيمة. قارن المقاييس ومنحنيات ROC وإيه اللي كل موديل بيهتم بيه.")}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="note">Best by test accuracy: <b>{metrics["Model"][0]}</b> '
                f'({metrics["Test Accuracy"][0] * 100:.1f}%).</div>', unsafe_allow_html=True)
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns[1:]})
                 .background_gradient(cmap="Blues", subset=metrics.columns[1:]), width="stretch")

    a, b = st.columns(2)
    with a:
        metric = st.selectbox("Compare by", list(metrics.columns[1:]), index=1)
        animated_bars(
            labels=metrics["Model"].tolist(),
            values=metrics[metric].tolist(),
            title=L(f"Compare by {metric}", f"مقارنة حسب {metric}"),
            y_label=metric,
            height=400,
            value_decimals=3,
        )
    with b:
        fig = go.Figure()
        for n, (fpr, tpr) in roc.items():
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=n)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="grey"))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style(fig, 400), width="stretch", config=PLOT_CONFIG)

    pick = st.selectbox("Inspect a model", list(roc.keys()))
    c, d = st.columns(2)
    with c:
        cm = confusion_matrix(yte, preds[pick])
        st.plotly_chart(style(px.imshow(cm, text_auto=True, x=["No", "Yes"], y=["No", "Yes"],
                                        color_continuous_scale="Blues", title=f"Confusion matrix: {pick}",
                                        labels=dict(x="Predicted", y="Actual")), 380),
                        width="stretch", config=PLOT_CONFIG)
    with d:
        with st.spinner("Measuring feature importance..."):
            imp = permutation_importance(pairs[pick][0], Xte_s, yte, n_repeats=10, random_state=42)
        animated_bars(
            labels=FEATS,
            values=imp.importances_mean.tolist(),
            title=L("What drives the prediction", "إيه اللي بيحرك التوقّع"),
            y_label=L("Accuracy lost when shuffled", "الدقة اللي بتضيع لما نخلط"),
            height=380,
            value_decimals=3,
        )

# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------
if page == "Predictor":
    welcome("Predictor")
    st.markdown(
        f'<div class="hero"><h1>{L("Predictor", "التوقّع")}</h1>'
        f'<p>{L("Set time and salary for any visitor. The probability fills in live, with a clear reason behind the number.", "حط الوقت والمرتب لأي زائر. الاحتمال بيتملي مباشرة، مع سبب واضح ورا الرقم.")}</p></div>',
        unsafe_allow_html=True,
    )
    if "history" not in st.session_state:
        st.session_state.history = []

    left, right = st.columns([1, 2])
    with left:
        name = st.selectbox("Model", list(pairs.keys()))
        time = st.slider("Time on site (min)", tmin, tmax, int(df[T].median()))
        salary = st.slider("Salary", smin, smax, int(df[S].median()), step=500)
        pair = pairs[name]
        p = float(proba(pair, [[time, salary]])[0])
        likely = p >= 0.5

        st.markdown(f'<div class="verdict {"yes" if likely else "no"}">'
                    f'{L("Likely to click", "غالباً هيدوس") if likely else L("Unlikely to click", "غالباً مش هيدوس")}</div>', unsafe_allow_html=True)
        g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": BLUE if likely else "#D9622B"},
                                          "threshold": {"line": {"color": "#0E1630", "width": 4}, "value": 50}}))
        st.markdown(dash_css(), unsafe_allow_html=True)
        st.markdown(f'<div class="vc2">{ring(p * 100, BLUE if likely else "#D9622B", L("click probability", "احتمال الضغط"), 190)}</div>',
                    unsafe_allow_html=True)

        base = df["Clicked"].mean()
        d = np.hypot((df[T] - time) / df[T].std(), (df[S] - salary) / df[S].std())
        near = df.loc[d.nsmallest(15).index]
        st.markdown(L(f'<div class="note">This visitor is <b>{p / base:.1f}x</b> as likely as the average user '
                      f'({base * 100:.0f}%). Of the 15 most similar users in your data, '
                      f'<b>{int(near["Clicked"].sum())}</b> clicked.</div>',
                      f'<div class="note">الزائر ده احتمال ضغطه <b>{p / base:.1f}x</b> من المستخدم المتوسط '
                      f'({base * 100:.0f}%). ومن أقرب 15 مستخدم شبهه في الداتا، '
                      f'<b>{int(near["Clicked"].sum())}</b> داسوا.</div>'), unsafe_allow_html=True)
        if st.button(L("Save scenario", "احفظ السيناريو")):
            st.session_state.history.append({"Model": name, "Time": time, "Salary": salary,
                                             "Click probability %": round(p * 100, 1)})

    with right:
        t0, t1, t2, t3 = st.tabs([L("Why this result", "ليه النتيجة دي"), L("Decision map", "خريطة القرار"),
                                   L("What-if curves", "لو غيّرت إيه"), L("Saved scenarios", "السيناريوهات")])
        with t0:
            # Exact Shapley values for two features (the same idea SHAP uses)
            bg = df[FEATS].sample(min(300, len(df)), random_state=0).copy()
            base_p = proba(pair, bg.values).mean()
            bt, bs = bg.copy(), bg.copy()
            bt[T], bs[S] = time, salary
            v_t, v_s = proba(pair, bt.values).mean(), proba(pair, bs.values).mean()
            phi_t = 0.5 * ((v_t - base_p) + (p - v_s))
            phi_s = 0.5 * ((v_s - base_p) + (p - v_t))
            w = go.Figure(go.Waterfall(
                x=[L("Average user", "المستخدم المتوسط"), T, S, L("This visitor", "الزائر ده")],
                measure=["absolute", "relative", "relative", "total"],
                y=[base_p * 100, phi_t * 100, phi_s * 100, 0],
                increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color="#D9622B")),
                totals=dict(marker=dict(color="#0E1630")), connector=dict(line=dict(color=GREY))))
            w.update_layout(title=L("How each feature moved the click probability (percentage points)",
                                    "كل عامل حرّك احتمال الضغط قد إيه (نقطة مئوية)"), showlegend=False)
            st.plotly_chart(style(w, 380), width="stretch", config=PLOT_CONFIG)
            main, ph = (T, phi_t) if abs(phi_t) >= abs(phi_s) else (S, phi_s)
            st.markdown(L(f'<div class="note"><b>{main}</b> is the biggest factor here. It '
                          f'{"raises" if ph > 0 else "lowers"} the click probability by {abs(ph) * 100:.1f} points '
                          f'compared with the average user.</div>',
                          f'<div class="note"><b>{main}</b> هو أكبر عامل هنا. بيـ{"زوّد" if ph > 0 else "قلّل"} '
                          f'احتمال الضغط بمقدار {abs(ph) * 100:.1f} نقطة مقارنة بالمستخدم المتوسط.</div>'),
                        unsafe_allow_html=True)
        with t1:
            tg, sg = np.linspace(tmin, tmax, 50), np.linspace(smin, smax, 50)
            tt, ss = np.meshgrid(tg, sg)
            Z = proba(pair, np.c_[tt.ravel(), ss.ravel()]).reshape(tt.shape)
            fig = go.Figure(go.Contour(x=tg, y=sg, z=Z * 100, colorscale="Blues", contours_coloring="heatmap",
                                       colorbar=dict(title="Click %"), opacity=0.85))
            for lab, colr in (("No", GREY), ("Yes", "#0E1630")):
                sub = df[df["Clicked Text"] == lab].sample(min(250, (df["Clicked Text"] == lab).sum()), random_state=1)
                fig.add_scatter(x=sub[T], y=sub[S], mode="markers", name=f"Clicked: {lab}",
                                marker=dict(size=6, color=colr, line=dict(width=1, color="white")))
            fig.add_scatter(x=[time], y=[salary], mode="markers", name="This visitor",
                            marker=dict(size=22, color="#FFB020", symbol="star", line=dict(width=2, color="#0E1630")))
            fig.update_layout(xaxis_title=T, yaxis_title=S, title=f"Where the model says yes ({name})")
            st.plotly_chart(style(fig, 520), width="stretch", config=PLOT_CONFIG)
        with t2:
            ts, ssal = np.linspace(tmin, tmax, 60), np.linspace(smin, smax, 60)
            a = px.line(x=ts, y=proba(pair, np.c_[ts, np.full(60, salary)]) * 100,
                        title=f"Click % as time changes (salary fixed at ${salary:,})",
                        labels=dict(x=T, y="Click %"))
            a.add_scatter(x=[time], y=[p * 100], mode="markers", marker=dict(size=14, color="#FFB020"), name="Now")
            b = px.line(x=ssal, y=proba(pair, np.c_[np.full(60, time), ssal]) * 100,
                        title=f"Click % as salary changes (time fixed at {time} min)",
                        labels=dict(x=S, y="Click %"))
            b.add_scatter(x=[salary], y=[p * 100], mode="markers", marker=dict(size=14, color="#FFB020"), name="Now")
            a.update_traces(line_color=BLUE, selector=dict(mode="lines"))
            b.update_traces(line_color=BLUE, selector=dict(mode="lines"))
            st.plotly_chart(style(a, 280), width="stretch", config=PLOT_CONFIG)
            st.plotly_chart(style(b, 280), width="stretch", config=PLOT_CONFIG)
        with t3:
            if st.session_state.history:
                h = pd.DataFrame(st.session_state.history)
                st.dataframe(h, width="stretch")
                animated_bars(
                    labels=[f"S{i+1}" for i in range(len(h))],
                    values=h["Click probability %"].tolist(),
                    title=L("Saved scenarios", "السيناريوهات المحفوظة"),
                    y_label=L("Click probability %", "احتمال الضغط %"),
                    height=320,
                    value_suffix="%",
                    value_decimals=1,
                )
                if st.button("Clear scenarios"):
                    st.session_state.history = []
                    st.rerun()
            else:
                st.info("No scenarios yet. Set the sliders, then choose Save scenario to compare visitors.")

# ------------------------------------------------------------
# Campaign Simulator
# ------------------------------------------------------------
if page == "Campaign Simulator":
    welcome("Campaign Simulator")
    st.markdown(
        f'<div class="hero"><h1>{L("Campaign Simulator", "محاكي الحملات")}</h1>'
        f'<p>{L("Pick an audience, set your budget, and watch expected clicks and return fill in as you refine the plan.", "اختار جمهور، حط ميزانية، وشوف الضغطات المتوقعة والعائد وهم بيتملوا وأنت بتعدّل الخطة.")}</p></div>',
        unsafe_allow_html=True,
    )
    name = st.selectbox("Model", list(pairs.keys()))
    a, b = st.columns(2)
    with a:
        lo_t, hi_t = st.slider("Target: time on site", tmin, tmax, (tmin, tmax))
        lo_s, hi_s = st.slider("Target: salary", smin, smax, (smin, smax), step=500)
    with b:
        budget = st.number_input("Budget ($)", 100, 1_000_000, 5000, step=500)
        cpm = st.number_input("Cost per 1,000 impressions ($)", 1.0, 100.0, 12.0, step=0.5)
        value = st.number_input("Value of one click ($)", 0.1, 500.0, 2.5, step=0.1)

    aud = df[df[T].between(lo_t, hi_t) & df[S].between(lo_s, hi_s)]
    if aud.empty:
        st.warning("No users match this audience. Widen the ranges.")
    else:
        ctr = proba(pairs[name], aud[FEATS].values).mean()
        everyone = proba(pairs[name], df[FEATS].values).mean()
        imps = budget / cpm * 1000
        clicks, rev = imps * ctr, imps * ctr * value
        kpi_row([
            (L("Expected click rate", "معدل الضغط المتوقع"), f"{ctr * 100:.1f}%",
             f"vs {everyone * 100:.1f}% untargeted"),
            (L("Expected clicks", "الضغطات المتوقعة"), f"{clicks:,.0f}",
             f"from {imps:,.0f} impressions"),
            (L("Cost per click", "تكلفة الضغطة"),
             f"${budget / clicks:,.2f}" if clicks else "-", ""),
            (L("Return on spend", "العائد على الإنفاق"),
             f"{(rev - budget) / budget * 100:+.0f}%", f"revenue ${rev:,.0f}"),
        ])

        st.write("")
        sizes = np.linspace(0, 1, 6)[1:]
        curve_labels, curve_vals = [], []
        for share in sizes:
            part = aud.assign(p=proba(pairs[name], aud[FEATS].values)).nlargest(max(1, int(len(aud) * share)), "p")
            curve_labels.append(f"{int(share * 100)}%")
            curve_vals.append(float(part["p"].mean() * 100))
        animated_bars(
            labels=curve_labels,
            values=curve_vals,
            title=L("What happens if you only target the most likely users",
                    "لو استهدفت أكتر الناس احتمال يضغطوا بس"),
            y_label=L("Expected click rate %", "معدل الضغط المتوقع %"),
            height=400,
            value_suffix="%",
            value_decimals=1,
        )

# ------------------------------------------------------------
# Batch scoring
# ------------------------------------------------------------
if page == "Batch Scoring":
    welcome("Batch Scoring")
    st.markdown(
        f'<div class="hero"><h1>{L("Batch Scoring", "تقييم دفعة")}</h1>'
        f'<p>{L("Upload a CSV with Time Spent on Site and Salary. Every row gets a click probability you can download.", "ارفع CSV فيه الوقت على الموقع والمرتب. كل صف هياخد احتمال ضغط تقدر تنزّله.")}</p></div>',
        unsafe_allow_html=True,
    )
    name = st.selectbox("Model", list(pairs.keys()))
    st.download_button("Download a template", df[FEATS].head(5).to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("Your CSV", type="csv")
    if up:
        new = pd.read_csv(up)
        missing = [c for c in FEATS if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}. Rename them to match the template and upload again.")
        else:
            new["Click probability %"] = (proba(pairs[name], new[FEATS].values) * 100).round(1)
            new["Prediction"] = np.where(new["Click probability %"] >= 50, "Will click", "Will not click")
            c1, c2 = st.columns(2)
            with c1: kpi("Rows scored", f"{len(new):,}")
            with c2: kpi("Predicted to click", f"{(new['Prediction'] == 'Will click').mean() * 100:.1f}%")
            st.dataframe(new, width="stretch")
            counts, edges = np.histogram(new["Click probability %"], bins=10, range=(0, 100))
            bin_labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(counts))]
            animated_bars(
                labels=bin_labels,
                values=counts.tolist(),
                title=L("Click probability distribution", "توزيع احتمال الضغط"),
                y_label=L("Rows", "عدد الصفوف"),
                height=340,
                value_decimals=0,
            )
            st.download_button("Download scored file", new.to_csv(index=False), "scored.csv", "text/csv")


# ------------------------------------------------------------
# Insights and report
# ------------------------------------------------------------
def make_pdf(title, lines):
    # Works with both fpdf2 and the old fpdf package
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 10, title)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 11)
    for ln in lines:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, "- " + ln.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(1)
    try:
        out = pdf.output(dest="S")
    except TypeError:
        out = pdf.output()
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")


if page == "Insights":
    welcome("Insights")
    st.markdown(
        f'<div class="hero"><h1>{L("Insights & Report", "التقرير والرؤى")}</h1>'
        f'<p>{L("A quiet summary of what the data is telling you. Download the report when you are ready.", "ملخص هادي لما بتقوله الداتا. نزّل التقرير لما تكون جاهز.")}</p></div>',
        unsafe_allow_html=True,
    )
    best = metrics.iloc[0]
    rate = df["Clicked"].mean() * 100
    yes, no = df[df.Clicked == 1], df[df.Clicked == 0]
    q = df.assign(tb=pd.qcut(df[T], 4, labels=False, duplicates="drop") + 1,
                  sb=pd.qcut(df[S], 4, labels=False, duplicates="drop") + 1)
    seg = (q.groupby(["tb", "sb"])["Clicked"].agg(["mean", "size"]).reset_index()
           .sort_values("mean", ascending=False))
    top = seg.iloc[0]
    tq, sq = f"Q{int(top['tb'])}", f"Q{int(top['sb'])}"
    imp = permutation_importance(pairs[best["Model"]][0], Xte_s, yte, n_repeats=10, random_state=42)
    driver = FEATS[int(np.argmax(imp.importances_mean))]

    en = [
        f"The dataset has {len(df):,} users and an overall click rate of {rate:.1f}%.",
        f"Users who clicked spent {yes[T].mean():.1f} min on the site on average (non-clickers: {no[T].mean():.1f} min) "
        f"and earned ${yes[S].mean():,.0f} on average (non-clickers: ${no[S].mean():,.0f}).",
        f"The strongest segment is time-on-site quartile {tq} with salary quartile {sq} (Q4 = highest): "
        f"{top['mean'] * 100:.0f}% click rate across {int(top['size'])} users.",
        f"The best model is {best['Model']} with {best['Test Accuracy'] * 100:.1f}% test accuracy "
        f"and ROC AUC {best['ROC AUC']:.3f}.",
        f"The feature that matters most to that model is {driver}.",
        "Recommendation: concentrate budget on the strongest segment and check cost per click "
        "in the Campaign Simulator before launch.",
    ]
    ar = [
        f"الداتا فيها {len(df):,} مستخدم ومعدل الضغط الإجمالي {rate:.1f}%.",
        f"اللي داسوا قضّوا في المتوسط {yes[T].mean():.1f} دقيقة (اللي ما داسوش: {no[T].mean():.1f}) "
        f"ومتوسط مرتبهم ${yes[S].mean():,.0f} (اللي ما داسوش: ${no[S].mean():,.0f}).",
        f"أقوى شريحة هي الربع {tq} في الوقت مع الربع {sq} في المرتب (Q4 = الأعلى): "
        f"معدل ضغط {top['mean'] * 100:.0f}% على {int(top['size'])} مستخدم.",
        f"أفضل موديل هو {best['Model']} بدقة اختبار {best['Test Accuracy'] * 100:.1f}% "
        f"و ROC AUC {best['ROC AUC']:.3f}.",
        f"أهم عامل عند الموديل ده هو {driver}.",
        "التوصية: ركّز الميزانية على أقوى شريحة، وراجع تكلفة الضغطة في محاكي الحملات قبل الإطلاق.",
    ]
    for ln in (ar if AR else en):
        st.markdown(f'<div class="note">{ln}</div>', unsafe_allow_html=True)

    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns[1:]}), width="stretch")
    model_lines = [f"{r.Model}: accuracy {r._3:.3f}, F1 {r.F1:.3f}, ROC AUC {r._7:.3f}"
                   for r in metrics.itertuples()]
    pdf_bytes = make_pdf("AdPulse - Click Prediction Report", en + ["Model results:"] + model_lines)
    md = "# AdPulse\n\n" + "\n".join("- " + x for x in (ar if AR else en))
    a, b = st.columns(2)
    with a:
        st.download_button(L("Download report (PDF)", "نزّل التقرير (PDF)"), pdf_bytes, "adpulse_report.pdf",
                           "application/pdf")
    with b:
        st.download_button(L("Download report (Markdown)", "نزّل التقرير (Markdown)"), md, "adpulse_report.md")
    if AR:
        st.caption("تقرير الـ PDF بالإنجليزي حالياً.")
