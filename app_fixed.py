"""
============================================================
 Smart Crop Recommendation System  —  FIXED VERSION
 ──────────────────────────────────────────────────────────
 SELF-CONTAINED: HTML is embedded inside this file.
 No separate templates/ folder needed.

 Run from the SAME folder as your .pkl files:
     python app_fixed.py
 Then open: http://127.0.0.1:5000
============================================================

 Soft Computing Novelties vs the base IEEE paper:
 1. Soft Voting Ensemble  (RF + GBM + KNN + NaiveBayes)
 2. SHAP Explainability   (why was this crop chosen?)
 3. Fuzzy Soil Health     (trapezoidal membership functions)
 4. Season Suitability    (Kharif / Rabi / Zaid)
 5. Top-3 Alternatives    (next-best crops + confidence)
 6. Smart Alerts          (out-of-range warnings)
============================================================
"""

from flask import Flask, request
from flask import render_template_string   # ← key fix: no templates/ folder needed
import pickle, os, io, base64, warnings
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Locate .pkl files relative to this script ────────────
BASE = os.path.dirname(os.path.abspath(__file__))

model      = pickle.load(open(os.path.join(BASE, 'model.pkl'),    'rb'))
rf_model   = pickle.load(open(os.path.join(BASE, 'rf_model.pkl'), 'rb'))
le         = pickle.load(open(os.path.join(BASE, 'encoder.pkl'),  'rb'))
features   = pickle.load(open(os.path.join(BASE, 'features.pkl'), 'rb'))

shap_explainer = shap.TreeExplainer(rf_model)

# ── Mappings ─────────────────────────────────────────────
SEASON_MAP = {
    'rice':'Kharif','maize':'Kharif','chickpea':'Rabi','kidneybeans':'Kharif',
    'pigeonpeas':'Kharif','mothbeans':'Kharif','mungbean':'Kharif',
    'blackgram':'Kharif','lentil':'Rabi','pomegranate':'Zaid','banana':'Zaid',
    'mango':'Zaid','grapes':'Rabi','watermelon':'Zaid','muskmelon':'Zaid',
    'apple':'Rabi','orange':'Rabi','papaya':'Zaid','coconut':'Zaid',
    'cotton':'Kharif','jute':'Kharif','coffee':'Zaid',
}
SEASON_INFO = {
    'Kharif': {'emoji':'🌧️','months':'Jun – Oct','color':'#10b981'},
    'Rabi':   {'emoji':'❄️', 'months':'Nov – Apr','color':'#3b82f6'},
    'Zaid':   {'emoji':'☀️', 'months':'Mar – Jun','color':'#f59e0b'},
}
CROP_EMOJI = {
    'rice':'🌾','maize':'🌽','chickpea':'🫘','kidneybeans':'🫘',
    'pigeonpeas':'🫛','mothbeans':'🌱','mungbean':'🌱','blackgram':'🌱',
    'lentil':'🫘','pomegranate':'🍎','banana':'🍌','mango':'🥭',
    'grapes':'🍇','watermelon':'🍉','muskmelon':'🍈','apple':'🍏',
    'orange':'🍊','papaya':'🍈','coconut':'🥥','cotton':'🌿',
    'jute':'🌿','coffee':'☕',
}
IDEAL_RANGES = {
    'N':(0,140,'Nitrogen'),'P':(5,145,'Phosphorus'),'K':(5,205,'Potassium'),
    'temperature':(8,44,'Temperature'),'humidity':(14,100,'Humidity'),
    'ph':(3.5,9.5,'pH'),'rainfall':(20,300,'Rainfall'),
}


# ════════════════════════════════════════════════════════
#  FUZZY LOGIC  (Novelty #3)
# ════════════════════════════════════════════════════════
def trap(x,a,b,c,d):
    if x<=a or x>=d: return 0.0
    if a<x<=b: return (x-a)/(b-a)
    if b<x<=c: return 1.0
    return (d-x)/(d-c)

def fuzzy_health(N,P,K,temperature,humidity,ph,rainfall):
    sc = {
        'Nitrogen (N)':    trap(N,           20,  40, 120, 140),
        'Phosphorus (P)':  trap(P,           10,  20, 100, 130),
        'Potassium (K)':   trap(K,            5,  15, 100, 150),
        'Temperature':     trap(temperature,  8,  20,  35,  44),
        'Humidity':        trap(humidity,    14,  50,  90, 100),
        'pH':              trap(ph,         4.0, 5.5, 7.5, 9.0),
        'Rainfall':        trap(rainfall,   20,  60, 200, 280),
    }
    w = {'Nitrogen (N)':.20,'Phosphorus (P)':.15,'Potassium (K)':.15,
         'Temperature':.15,'Humidity':.10,'pH':.15,'Rainfall':.10}
    idx = round(sum(w[k]*v for k,v in sc.items())*100, 1)
    return sc, idx


# ════════════════════════════════════════════════════════
#  CHART HELPERS
# ════════════════════════════════════════════════════════
def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def shap_chart(input_df, class_idx):
    sv   = shap_explainer.shap_values(input_df)[0][:, class_idx]
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('#1e293b'); ax.set_facecolor('#1e293b')
    colors = ['#10b981' if v>=0 else '#ef4444' for v in sv]
    ax.barh(features, sv, color=colors, edgecolor='none', height=.55)
    ax.axvline(0, color='#059669', lw=.8, ls='--')
    ax.set_xlabel('SHAP Value', color='#94a3b8', fontsize=10)
    ax.set_title('Why this crop? — Feature Contributions', color='#f8fafc', fontsize=12, pad=12)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    for s in ax.spines.values(): s.set_edgecolor('#334155')
    pos = mpatches.Patch(color='#10b981', label='Favours this crop')
    neg = mpatches.Patch(color='#ef4444', label='Opposes this crop')
    ax.legend(handles=[pos,neg], fontsize=9, facecolor='#1e293b',
              edgecolor='#334155', labelcolor='#f8fafc', loc='lower right')
    plt.tight_layout()
    return to_b64(fig)

def radar_chart(fuzzy_scores):
    labels = list(fuzzy_scores.keys())
    vals   = [fuzzy_scores[k]*100 for k in labels]
    n      = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    vals  += vals[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1e293b'); ax.set_facecolor('#1e293b')
    ax.plot(angles, vals, color='#10b981', lw=2.5)
    ax.fill(angles, vals, color='#059669', alpha=.35)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, color='#f8fafc', fontsize=9)
    ax.set_ylim(0,100); ax.set_yticks([25,50,75,100])
    ax.set_yticklabels(['25','50','75','100'], color='#94a3b8', fontsize=8)
    ax.spines['polar'].set_color('#334155'); ax.grid(color='#334155', lw=.8)
    ax.set_title('Fuzzy Soil Health Radar', color='#f8fafc', fontsize=12, pad=18)
    plt.tight_layout()
    return to_b64(fig)

def get_alerts(data):
    out=[]
    for f,(lo,hi,name) in IDEAL_RANGES.items():
        v=data[f]
        if v<lo: out.append(f"⚠️ {name} ({v}) is below the typical minimum ({lo}).")
        elif v>hi: out.append(f"⚠️ {name} ({v}) exceeds the typical maximum ({hi}).")
    return out


# ════════════════════════════════════════════════════════
#  HTML TEMPLATE  (embedded — no templates/ folder needed)
# ════════════════════════════════════════════════════════
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🌾 Smart Crop Recommender</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ── PALETTE: Slate & Emerald ── */
:root{
  --bg:        #0f172a;   /* Slate 900 */
  --surface:   #1e293b;   /* Slate 800 */
  --card:      #0f172a;   /* Slate 900 */
  --accent:    #10b981;   /* Emerald 500 */
  --accentd:   #059669;   /* Emerald 600 */
  --dark:      #020617;   /* Slate 950 */
  --red:       #ef4444;
  --gold:      #f59e0b;
  --text:      #f8fafc;
  --muted:     #94a3b8;
  --border:    #334155;
  --r:         16px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:'Inter',system-ui,sans-serif;
  background:var(--bg);
  color:var(--text);
  min-height:100vh;
  padding-bottom:60px;
}

/* ══════════════════ HEADER ══════════════════ */
header{
  background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
  padding:28px 48px 24px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  border-bottom:2px solid var(--accentd);
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.header-left h1{font-size:2rem;font-weight:800;letter-spacing:0.5px;color:var(--accent)}
.header-left p{color:var(--muted);font-size:.95rem;margin-top:6px;font-weight:500;}
.badges{display:flex;gap:10px;flex-wrap:wrap}
.badge{
  background:var(--surface);border:1px solid var(--border);border-radius:20px;
  padding:6px 16px;font-size:.8rem;color:var(--muted);font-weight:500;
}
.badge b{color:var(--accent);font-weight:700}

/* ══════════════════ MAIN WRAPPER ══════════════════ */
.main{
  width:100%;
  max-width: 1200px;
  margin: 0 auto;
  padding:32px 24px;
}

/* ══════════════════ CARDS ══════════════════ */
.card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r);
  padding:32px;
  margin-bottom:24px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}
.card h2{font-size:1.2rem;color:var(--accent);margin-bottom:20px;font-weight:700;
         display:flex;align-items:center;gap:10px}
.card h3{font-size:.85rem;color:var(--muted);text-transform:uppercase;
         letter-spacing:1px;margin-bottom:16px;font-weight:700}

/* ══════════════════ FORM ══════════════════ */
.grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));gap:24px}

label{
  display:block;font-size:.78rem;color:var(--muted);
  margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px;font-weight:600
}
input{
  width:100%;background:var(--bg);
  border:1.5px solid var(--border);
  border-radius:10px;color:var(--text);
  padding:12px 16px;font-size:1rem;
  outline:none;transition:all .2s ease;
}
input:focus{border-color:var(--accent);box-shadow:0 0 0 4px rgba(16, 185, 129, 0.15);background:var(--surface);}
.hint{font-size:.7rem;color:var(--muted);margin-top:6px;line-height:1.4}

.btn-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 32px;
}
.btn{
  background:linear-gradient(135deg,#059669,#10b981);
  border:none;border-radius:12px;
  color:#ffffff;font-size:1.1rem;font-weight:700;
  padding:16px 48px;cursor:pointer;letter-spacing:1px;
  transition:all .3s ease;
  box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
}
.btn:hover{transform: translateY(-2px); box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);}
.btn:active{transform: translateY(1px); box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);}

/* ══════════════════ ALERTS ══════════════════ */
.alert-box{
  background:rgba(245, 158, 11, 0.1);border:1px solid rgba(245, 158, 11, 0.3);
  border-radius:12px;padding:16px 24px;margin-bottom:24px
}
.alert-box p{font-size:.9rem;color:var(--gold);margin-bottom:4px;display:flex;align-items:center;gap:8px;}
.err{
  background:rgba(239, 68, 68, 0.1);border:1px solid rgba(239, 68, 68, 0.3);
  border-radius:var(--r);padding:16px 24px;
  color:var(--red);margin-bottom:24px;font-weight:500;
}

/* ══════════════════ RESULT LAYOUT ══════════════════ */
.top-row{
  display:grid;
  grid-template-columns:1fr 1.4fr 1fr;
  gap:24px;
  margin-bottom:24px;
}
.mid-row{
  display:grid;
  grid-template-columns:3fr 2fr;
  gap:24px;
  margin-bottom:24px;
}
@media(max-width:1100px){
  .top-row{grid-template-columns:1fr 1fr}
  .mid-row{grid-template-columns:1fr}
}
@media(max-width:768px){.top-row{grid-template-columns:1fr}}

/* ══════════════════ RESULT CARD ══════════════════ */
.result-main{
  background:linear-gradient(145deg,#1e293b,#0f172a);
  border:2px solid var(--accent);
  border-radius:var(--r);
  padding:40px 24px;
  text-align:center;
  display:flex;flex-direction:column;
  justify-content:center;align-items:center;
  box-shadow: 0 10px 40px rgba(16, 185, 129, 0.15);
  position:relative;
  overflow:hidden;
}
.result-main::before{
  content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
  background:radial-gradient(circle, rgba(16,185,129,0.1) 0%, transparent 60%);
  pointer-events:none;
}
.crop-emoji{font-size:6rem;line-height:1;margin-bottom:10px;filter:drop-shadow(0 10px 20px rgba(0,0,0,0.3));}
.cname{
  font-size:3rem;font-weight:900;
  text-transform:capitalize;margin:10px 0 8px;
  color:var(--accent);
  text-shadow:0 0 40px rgba(16, 185, 129, 0.3);
  letter-spacing:1px;
}
.conf-txt{font-size:1.1rem;color:var(--accent);font-weight:600;opacity:0.9;}
.bar-wrap{
  background:var(--bg);border-radius:10px;height:12px;
  margin:16px 0;overflow:hidden;width:80%;
  box-shadow:inset 0 2px 5px rgba(0,0,0,0.2);
}
.bar{
  background:linear-gradient(90deg,#059669,#10b981);
  height:100%;border-radius:10px;
}
.season-badge{
  display:inline-flex;align-items:center;gap:8px;
  padding:8px 20px;border-radius:30px;
  margin-top:16px;font-size:.95rem;font-weight:700;
  backdrop-filter:blur(4px);
}

/* ══════════════════ STAT CARDS ══════════════════ */
.stat-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:28px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.stat-num{font-size:4rem;font-weight:900;line-height:1;color:var(--accent);margin-bottom:5px;}
.stat-lbl{font-size:.85rem;text-transform:uppercase;letter-spacing:1px;
          color:var(--muted);font-weight:700}
.stat-sub{font-size:.9rem;color:var(--muted);margin-top:16px;line-height:1.6}

/* ══════════════════ ALTERNATIVES ══════════════════ */
.alt-item{
  display:flex;align-items:center;gap:16px;
  padding:16px 0;border-bottom:1px solid var(--border);
}
.alt-item:last-child{border-bottom:none}
.alt-name{text-transform:capitalize;font-weight:600;font-size:1.1rem;color:var(--text);margin-bottom:6px;}
.alt-conf{margin-left:auto;color:var(--gold);font-weight:800;font-size:1rem;white-space:nowrap}
.abar-w{background:var(--bg);border-radius:6px;height:8px;overflow:hidden}
.abar{background:linear-gradient(90deg,#d97706,#f59e0b);height:100%;border-radius:6px}

/* ══════════════════ CHARTS ══════════════════ */
.chart-img{width:100%;border-radius:12px;display:block;border:1px solid var(--border);}
.chart-note{color:var(--muted);font-size:.85rem;margin-top:16px;line-height:1.6;background:var(--bg);padding:16px;border-radius:10px;border:1px solid var(--border);}

/* ══════════════════ PARAM BARS ══════════════════ */
.prow{display:flex;align-items:center;gap:16px;margin-bottom:16px}
.pname{width:130px;font-size:.9rem;color:var(--muted);flex-shrink:0;font-weight:600}
.pbw{flex:1;background:var(--bg);border-radius:8px;height:12px;overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,0.2);}
.pb{height:100%;border-radius:8px}
.pval{width:50px;text-align:right;font-size:.9rem;color:var(--text);font-weight:700}

/* ══════════════════ FOOTER ══════════════════ */
footer{
  text-align:center;color:var(--muted);font-size:.85rem;
  margin-top:60px;padding:24px 0;
  border-top:1px solid var(--border);
  display:flex;justify-content:center;gap:16px;flex-wrap:wrap;
}
footer b{color:var(--accent);font-weight:600}
</style>
</head>
<body>

<!-- ══════════ HEADER ══════════ -->
<header>
  <div class="header-left">
    <h1>🌾 Smart Crop Recommender</h1>
    <p>AI-powered precision agriculture &mdash; Soft Computing Project</p>
  </div>
  <div class="badges">
    <div class="badge">Model: <b>Soft Voting Ensemble</b></div>
    <div class="badge">Accuracy: <b>99.55%</b></div>
    <div class="badge">XAI: <b>SHAP</b></div>
    <div class="badge">Fuzzy: <b>Soil Health Index</b></div>
  </div>
</header>

<!-- ══════════ MAIN ══════════ -->
<div class="main">

<!-- INPUT FORM -->
<div class="card">
  <h2>📊 Enter Soil &amp; Climate Parameters</h2>
  <form method="POST" action="/predict">
    <div class="grid">
      <div>
        <label>Nitrogen (N)</label>
        <input type="number" name="N" step="any"
               value="{{ input_data.N if input_data else '' }}" placeholder="e.g. 90" required>
        <div class="hint">mg/kg &middot; 0&ndash;140</div>
      </div>
      <div>
        <label>Phosphorus (P)</label>
        <input type="number" name="P" step="any"
               value="{{ input_data.P if input_data else '' }}" placeholder="e.g. 42" required>
        <div class="hint">mg/kg &middot; 5&ndash;145</div>
      </div>
      <div>
        <label>Potassium (K)</label>
        <input type="number" name="K" step="any"
               value="{{ input_data.K if input_data else '' }}" placeholder="e.g. 43" required>
        <div class="hint">mg/kg &middot; 5&ndash;205</div>
      </div>
      <div>
        <label>Temperature (&deg;C)</label>
        <input type="number" name="temperature" step="any"
               value="{{ input_data.temperature if input_data else '' }}" placeholder="e.g. 22" required>
        <div class="hint">Ambient temp</div>
      </div>
      <div>
        <label>Humidity (%)</label>
        <input type="number" name="humidity" step="any"
               value="{{ input_data.humidity if input_data else '' }}" placeholder="e.g. 82" required>
        <div class="hint">Relative humidity</div>
      </div>
      <div>
        <label>Soil pH</label>
        <input type="number" name="ph" step="any"
               value="{{ input_data.ph if input_data else '' }}" placeholder="e.g. 6.5" required>
        <div class="hint">3.5 acid &middot; 7 neutral &middot; 9.5 alkaline</div>
      </div>
      <div>
        <label>Rainfall (mm)</label>
        <input type="number" name="rainfall" step="any"
               value="{{ input_data.rainfall if input_data else '' }}" placeholder="e.g. 202" required>
        <div class="hint">Annual rainfall mm</div>
      </div>
    </div>
    <div class="btn-wrapper">
      <button type="submit" class="btn">🔍 &nbsp; Recommend Crop</button>
    </div>
  </form>
</div>

{% if error %}
<div class="err">⛔ Error: {{ error }}</div>
{% endif %}

{% if result %}

<!-- ALERTS -->
{% if alerts %}
<div class="alert-box">
  {% for a in alerts %}<p>{{ a }}</p>{% endfor %}
</div>
{% endif %}

<!-- TOP ROW: Health Index | Result | Alternatives -->
<div class="top-row">

  <!-- Fuzzy Health Index -->
  <div class="stat-card">
    <h3>🌡️ Fuzzy Soil Health Index</h3>
    <div class="stat-num" style="color:{{ health_color }}">{{ health_idx }}</div>
    <div class="stat-lbl" style="color:{{ health_color }}">/100 &mdash; {{ health_label }}</div>
    <div class="stat-sub">
      Computed using <strong style="color:var(--accent)">fuzzy trapezoidal
      membership functions</strong> over all 7 parameters.
      100 = perfect growing conditions.
    </div>
    <!-- mini param bars -->
    <div style="margin-top:24px">
    {% for item in fuzzy_list %}
    {% set col = '#10b981' if item.score >= 70 else ('#f59e0b' if item.score >= 40 else '#ef4444') %}
    <div class="prow">
      <div class="pname">{{ item.name }}</div>
      <div class="pbw"><div class="pb" style="width:{{ item.score }}%;background:{{ col }}"></div></div>
      <div class="pval">{{ item.score }}%</div>
    </div>
    {% endfor %}
    </div>
  </div>

  <!-- Main Crop Result -->
  <div class="result-main">
    <div class="crop-emoji">{{ crop_emoji }}</div>
    <div class="cname">{{ result }}</div>
    <div class="conf-txt">{{ confidence }}% confidence</div>
    <div class="bar-wrap"><div class="bar" style="width:{{ confidence }}%"></div></div>
    <div class="season-badge"
         style="background:{{ season_color }}20;border:1px solid {{ season_color }}60;color:{{ season_color }}">
      {{ season_emoji }} &nbsp;{{ season }} Season &nbsp;&middot;&nbsp; {{ season_months }}
    </div>
  </div>

  <!-- Alternatives -->
  <div class="stat-card">
    <h3>🔄 Next-Best Alternatives</h3>
    {% for alt in alternatives %}
    <div class="alt-item">
      <div style="font-size:2.5rem">{{ alt.emoji }}</div>
      <div style="flex:1">
        <div class="alt-name">{{ alt.crop }}</div>
        <div class="abar-w"><div class="abar" style="width:{{ alt.confidence }}%"></div></div>
      </div>
      <div class="alt-conf">{{ alt.confidence }}%</div>
    </div>
    {% endfor %}
    <div style="margin-top:24px;padding-top:20px;border-top:1px solid var(--border)">
      <h3 style="margin-bottom:12px">📅 Growing Season</h3>
      <div style="font-size:2.2rem;margin-bottom:8px">{{ season_emoji }}</div>
      <div style="font-weight:800;font-size:1.2rem;color:{{ season_color }}">{{ season }} Season</div>
      <div style="color:var(--muted);font-size:.9rem;margin-top:6px">{{ season_months }}</div>
    </div>
  </div>

</div><!-- /top-row -->

<!-- MID ROW: SHAP + Radar -->
<div class="mid-row">
  <div class="card" style="margin-bottom:0">
    <h3>🧠 SHAP Explainability &mdash; Why {{ result }}?</h3>
    <img class="chart-img" src="data:image/png;base64,{{ shap_chart }}" alt="SHAP Chart">
    <div class="chart-note">
      <strong style="color:var(--accent)">Green bars</strong> push the model toward recommending
      <strong>{{ result }}</strong>. <strong style="color:var(--red)">Red bars</strong> push it away.
      This is eXplainable AI (XAI) &mdash; you can see exactly why the model chose this crop.
    </div>
  </div>
  <div class="card" style="margin-bottom:0">
    <h3>🕸️ Soil Parameter Radar</h3>
    <img class="chart-img" src="data:image/png;base64,{{ fuzzy_chart }}" alt="Radar">
  </div>
</div>

{% endif %}

<footer>
  <span>Soft Computing Project</span>
  <span>&bull;</span>
  <span>Ensemble: <b>RF + GBM + KNN + NaiveBayes (Soft Voting)</b></span>
  <span>&bull;</span>
  <span>XAI: <b>SHAP</b></span>
  <span>&bull;</span>
  <span>Fuzzy: <b>Mamdani-style Trapezoidal MF</b></span>
</footer>

</div><!-- /main -->
</body>
</html>"""


# ════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])          # ← explicit POST only
def predict():
    try:
        data = {
            'N':           float(request.form['N']),
            'P':           float(request.form['P']),
            'K':           float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity':    float(request.form['humidity']),
            'ph':          float(request.form['ph']),
            'rainfall':    float(request.form['rainfall']),
        }
        df = pd.DataFrame([data])[features]

        # Ensemble prediction
        proba    = model.predict_proba(df)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3     = le.inverse_transform(top3_idx)
        top3_c   = proba[top3_idx]

        best      = top3[0]
        best_conf = round(float(top3_c[0])*100, 1)

        alts = [{'crop': top3[i],
                 'confidence': round(float(top3_c[i])*100, 1),
                 'emoji': CROP_EMOJI.get(top3[i],'🌱')} for i in range(1,3)]

        # SHAP
        cls_idx   = list(le.classes_).index(best)
        s_chart   = shap_chart(df, cls_idx)

        # Fuzzy
        fsc, hidx = fuzzy_health(**data)
        r_chart   = radar_chart(fsc)
        flist     = [{'name':k,'score':round(v*100,1)} for k,v in fsc.items()]

        # Season
        season   = SEASON_MAP.get(best,'Zaid')
        sinf     = SEASON_INFO[season]

        # Health badge
        hcol  = "#10b981" if hidx>=70 else ('#f59e0b' if hidx>=45 else '#ef4444')
        hlbl  = 'Excellent' if hidx>=70 else ('Moderate' if hidx>=45 else 'Poor')

        return render_template_string(HTML,
            result=best, confidence=best_conf,
            crop_emoji=CROP_EMOJI.get(best,'🌱'),
            alternatives=alts,
            shap_chart=s_chart, fuzzy_chart=r_chart,
            fuzzy_list=flist,
            health_idx=hidx, health_color=hcol, health_label=hlbl,
            season=season, season_emoji=sinf['emoji'],
            season_months=sinf['months'], season_color=sinf['color'],
            alerts=get_alerts(data),
            input_data=data,
        )

    except Exception as e:
        return render_template_string(HTML, error=str(e))


if __name__ == '__main__':
    print("\n✅ Starting Smart Crop Recommender...")
    print("   Open your browser at: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)