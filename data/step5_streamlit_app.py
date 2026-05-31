# ============================================================
# STEP 5: Streamlit Web App (Flood Prediction UI)
# Run with:  streamlit run step5_streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Flood Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio label { 
    background: rgba(255,255,255,0.12);
    border-radius: 8px; padding: 6px 12px;
    margin-bottom: 4px; display: block;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.25);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

/* ── Main area ── */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 60%, #29b6f6 100%);
    border-radius: 18px; padding: 2.2rem 2.5rem 2rem;
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
}
.hero::after {
    content: "🌊"; font-size: 7rem; position: absolute;
    right: 2rem; top: 50%; transform: translateY(-50%);
    opacity: 0.18; line-height: 1;
}
.hero h1 { color: #fff; font-size: 2.2rem; font-weight: 800;
            margin: 0 0 0.3rem; letter-spacing: -0.5px; }
.hero p  { color: rgba(255,255,255,0.82); font-size: 1rem; margin: 0; }

/* ── Section label ── */
.section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #1976d2;
    margin-bottom: 0.4rem; margin-top: 1.2rem;
}

/* ── Input card ── */
.input-card {
    background: #f8faff;
    border: 1.5px solid #e3eaf5;
    border-radius: 14px; padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 1rem;
}

/* ── Predict button ── */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
div[data-testid="stForm"] button {
    background: linear-gradient(90deg, #1565c0, #1976d2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 1.05rem !important; padding: 0.7rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stForm"] button:hover { opacity: 0.88 !important; }

/* ── Result cards ── */
.result-wrap {
    border-radius: 16px; padding: 2rem 1.5rem;
    text-align: center; margin-top: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.result-wrap .badge {
    font-size: 3.2rem; margin-bottom: 0.5rem;
}
.result-wrap .severity-text {
    font-size: 1.6rem; font-weight: 800; margin: 0;
}
.result-wrap .severity-sub {
    font-size: 0.9rem; opacity: 0.75; margin-top: 0.3rem;
}
.low-card    { background: linear-gradient(135deg,#e8f5e9,#f1f8e9);
               border: 2px solid #66bb6a; color: #1b5e20; }
.medium-card { background: linear-gradient(135deg,#fff8e1,#fff3e0);
               border: 2px solid #ffa726; color: #bf360c; }
.high-card   { background: linear-gradient(135deg,#ffebee,#fce4ec);
               border: 2px solid #ef5350; color: #b71c1c; }

/* ── Confidence bar ── */
.conf-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.conf-label { width: 65px; font-weight:600; font-size:0.88rem; }
.conf-bar-bg { flex:1; background:#e9ecef; border-radius:999px; height:12px; }
.conf-bar-fill { height:12px; border-radius:999px; transition: width 0.6s ease; }
.conf-pct { width:42px; text-align:right; font-size:0.88rem; font-weight:700; }

/* ── KPI cards ── */
.kpi-card {
    background: #fff; border: 1.5px solid #e3eaf5;
    border-radius: 14px; padding: 1.1rem 1rem;
    text-align: center;
}
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #1565c0; }
.kpi-label { font-size: 0.78rem; color: #607d8b; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Chart card ── */
.chart-card {
    background: #fff; border: 1.5px solid #e3eaf5;
    border-radius: 14px; padding: 1.2rem;
    margin-bottom: 1rem;
}

/* ── About card ── */
.about-card {
    background: #f8faff; border: 1.5px solid #e3eaf5;
    border-radius: 14px; padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf    = joblib.load('random_forest_model.pkl')
    feats = joblib.load('feature_cols.pkl')
    return rf, feats

@st.cache_data
def load_data():
    return pd.read_csv('data_cleaned_full.csv')

try:
    model, FEATURE_COLS = load_model()
    df_full = load_data()
    model_loaded = True
    df_full['Severity_Enc'] = df_full['Severity'].map({'Low':0,'Medium':1,'High':2})
except FileNotFoundError:
    model_loaded = False
    df_full = None

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Flood Predictor")
    st.markdown("**India Flood Severity**  \nPowered by Random Forest ML")
    st.markdown("---")
    page = st.radio("Navigate", ["🔮 Predict", "📊 Dashboard", "ℹ️ About"],
                    label_visibility="collapsed")
    st.markdown("---")
    if model_loaded:
        st.success("✅ Model ready")
        st.caption(f"Features: {len(FEATURE_COLS)}")
    else:
        st.error("❌ Run step4 first")
    st.markdown("---")
    st.caption("Built with Streamlit & scikit-learn")

month_map = {
    'January':1,'February':2,'March':3,'April':4,
    'May':5,'June':6,'July':7,'August':8,
    'September':9,'October':10,'November':11,'December':12
}

LOCATIONS = sorted([
    'Agra','Ahmedabad','Ajmer','Aligarh','Amritsar','Asansol',
    'Aurangabad','Bareilly','Bhopal','Bhubaneswar','Chennai',
    'Coimbatore','Delhi','Dhanbad','Faridabad','Ghaziabad',
    'Guwahati','Gwalior','Hyderabad','Indore','Jabalpur',
    'Jaipur','Jamshedpur','Jodhpur','Kanpur','Kochi',
    'Kolkata','Kota','Lucknow','Ludhiana','Madurai','Meerut',
    'Mumbai','Mysuru','Nagpur','Nashik','Navi Mumbai','Patna',
    'Prayagraj','Pune','Rajkot','Ranchi','Shivamogga',
    'Srinagar','Surat','Thiruvananthapuram','Vadodara',
    'Varanasi','Vijayawada','Visakhapatnam'
])

def encode_location(loc_name):
    if model_loaded and df_full is not None:
        locs = sorted(df_full['Location'].unique())
        if loc_name in locs:
            return locs.index(loc_name)
    return abs(hash(loc_name)) % 100

def conf_bar_html(label, pct, color):
    return f"""
    <div class="conf-row">
        <span class="conf-label">{label}</span>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
        <span class="conf-pct">{pct:.1f}%</span>
    </div>"""

# ═════════════════════════════════════════════════════════════
# PAGE 1: PREDICT
# ═════════════════════════════════════════════════════════════
if page == "🔮 Predict":

    st.markdown("""
    <div class="hero">
        <h1>India Flood Severity Predictor</h1>
        <p>Enter location, rainfall, and water level to instantly predict flood severity using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<p class="section-label">📍 Location & Time</p>', unsafe_allow_html=True)
            location = st.selectbox("City / Location", LOCATIONS, label_visibility="collapsed")
            month    = st.selectbox("Month of flood event", list(month_map.keys()), index=6)

        with col2:
            st.markdown('<p class="section-label">🌧️ Rainfall & Water Conditions</p>', unsafe_allow_html=True)
            rainfall    = st.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0,
                                          value=800.0, step=50.0,
                                          help="Total rainfall received in millimetres")
            water_level = st.number_input("Water Level (m)", min_value=0.0, max_value=50.0,
                                          value=5.0, step=0.5,
                                          help="River / reservoir water level in metres")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Predict Flood Severity",
                                          use_container_width=True)

    # ── Result ────────────────────────────────────────────────
    if submitted:
        if not model_loaded:
            st.error("Model not loaded — please run step4_train_model.py first.")
        else:
            loc_enc   = encode_location(location)
            month_num = month_map[month]

            # Auto-derive all remaining features
            start_month     = month_num
            start_dayofweek = -1
            end_month       = min(month_num + 1, 12)
            end_dayofyear   = int((month_num / 12) * 365)
            # Median-impute flood impact columns (model still needs them)
            flood_duration  = int(df_full['Flood_Duration_Days'].median()) if df_full is not None else 10
            affected_people = int(df_full['Affected_People'].median())     if df_full is not None else 50000
            affected_area   = float(df_full['Affected_Area_sqkm'].median())if df_full is not None else 1000.0

            input_data = pd.DataFrame([{
                'Rainfall_Quantity_mm': rainfall,
                'Month_Num'           : month_num,
                'Flood_Duration_Days' : flood_duration,
                'Water_Level_m'       : water_level,
                'Affected_People'     : affected_people,
                'Affected_Area_sqkm'  : affected_area,
                'Location_Enc'        : loc_enc,
                'Start_Month'         : start_month,
                'Start_DayOfWeek'     : start_dayofweek,
                'End_Month'           : end_month,
                'End_DayOfYear'       : end_dayofyear,
            }])[FEATURE_COLS]

            pred_enc   = model.predict(input_data)[0]
            proba      = model.predict_proba(input_data)[0]
            label_map  = {0:'Low', 1:'Medium', 2:'High'}
            pred_label = label_map[pred_enc]

            badge  = {'Low':'🟢','Medium':'🟠','High':'🔴'}[pred_label]
            css_cl = {'Low':'low-card','Medium':'medium-card','High':'high-card'}[pred_label]
            advice = {
                'Low'   : 'No immediate action required. Monitor updates.',
                'Medium': 'Prepare evacuation plans. Stay alert.',
                'High'  : 'Immediate evacuation recommended!'
            }[pred_label]

            res_col, conf_col = st.columns([1, 1], gap="large")

            with res_col:
                st.markdown(f"""
                <div class="result-wrap {css_cl}">
                    <div class="badge">{badge}</div>
                    <p class="severity-text">Severity: {pred_label}</p>
                    <p class="severity-sub">{advice}</p>
                </div>
                """, unsafe_allow_html=True)

            with conf_col:
                st.markdown("#### Confidence Breakdown")
                colors = {'Low':'#4CAF50','Medium':'#FF9800','High':'#F44336'}
                bars_html = ""
                for lbl, clr in colors.items():
                    idx = ['Low','Medium','High'].index(lbl)
                    bars_html += conf_bar_html(lbl, proba[idx]*100, clr)
                st.markdown(bars_html, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Input Summary**")
                st.markdown(f"📍 **{location}** &nbsp;·&nbsp; 📅 **{month}**")
                st.markdown(f"🌧️ Rainfall: **{rainfall:.0f} mm** &nbsp;·&nbsp; 💧 Water Level: **{water_level:.1f} m**")

# ═════════════════════════════════════════════════════════════
# PAGE 2: DASHBOARD
# ═════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Flood Data Dashboard</h1>
        <p>Explore patterns and insights from 10,000 historical flood records across India.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("Run step4_train_model.py first to load model & data.")
    else:
        # ── KPI row ──────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            ("10,000",  "Total Records"),
            (str(df_full['Location'].nunique()), "Cities Covered"),
            (f"{df_full['Rainfall_Quantity_mm'].mean():.0f} mm", "Avg Rainfall"),
            (f"{df_full['Water_Level_m'].mean():.1f} m",  "Avg Water Level"),
        ]
        for col, (val, lbl) in zip([k1,k2,k3,k4], kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Severity donut + Rainfall boxplot ──────────
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            st.markdown("##### Severity Distribution")
            counts = df_full['Severity'].value_counts().reindex(['Low','Medium','High'])
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors  = ['#4CAF50','#FF9800','#F44336']
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, colors=colors,
                autopct='%1.1f%%', startangle=140,
                wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
            )
            for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
            ax.set_title("Flood Severity Split", fontsize=12, fontweight='bold', pad=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with c_right:
            st.markdown("##### Rainfall by Severity")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            palette = {'Low':'#4CAF50','Medium':'#FF9800','High':'#F44336'}
            sns.boxplot(data=df_full, x='Severity', y='Rainfall_Quantity_mm',
                        palette=palette, order=['Low','Medium','High'],
                        ax=ax, linewidth=1.5, fliersize=3)
            ax.set_xlabel("Severity", fontsize=10)
            ax.set_ylabel("Rainfall (mm)", fontsize=10)
            ax.set_title("Rainfall Distribution per Severity", fontsize=12, fontweight='bold')
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        # ── Row 2: Monthly trend ──────────────────────────────
        st.markdown("##### Monthly Flood Event Frequency")
        month_order = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December']
        monthly = df_full['Month'].value_counts().reindex(month_order).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 3))
        bar_colors = ['#ef5350' if v == monthly.max()
                      else '#42a5f5' for v in monthly.values]
        bars = ax.bar(range(12), monthly.values, color=bar_colors,
                      edgecolor='white', linewidth=1.2, width=0.65)
        ax.set_xticks(range(12))
        ax.set_xticklabels([m[:3] for m in month_order], fontsize=10)
        ax.set_ylabel("Events", fontsize=10)
        ax.set_title("Flood Events by Month  (red = peak)", fontsize=12, fontweight='bold')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── Row 3: Feature importance + Water level violin ────
        c3, c4 = st.columns(2, gap="large")

        with c3:
            st.markdown("##### Feature Importances")
            fi = pd.Series(model.feature_importances_,
                           index=FEATURE_COLS).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_fi = ['#1565c0' if v == fi.max() else '#90caf9' for v in fi.values]
            fi.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
            ax.set_xlabel("Importance", fontsize=10)
            ax.set_title("Random Forest Feature Importances", fontsize=12, fontweight='bold')
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with c4:
            st.markdown("##### Water Level vs Severity")
            fig, ax = plt.subplots(figsize=(5, 4))
            palette = {'Low':'#4CAF50','Medium':'#FF9800','High':'#F44336'}
            sns.violinplot(data=df_full, x='Severity', y='Water_Level_m',
                           palette=palette, order=['Low','Medium','High'],
                           ax=ax, linewidth=1.2, inner='quartile')
            ax.set_xlabel("Severity", fontsize=10)
            ax.set_ylabel("Water Level (m)", fontsize=10)
            ax.set_title("Water Level Distribution", fontsize=12, fontweight='bold')
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

# ═════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ═════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="hero">
        <h1>About This Project</h1>
        <p>India Flood Prediction System — built for AIML Mini Project</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")

    with a1:
        # ── Dataset card ──────────────────────────────────────
        with st.container(border=True):
            st.markdown("#### 🗄️ Dataset")
            c_l, c_r = st.columns(2)
            with c_l:
                st.metric("Records",  "10,000")
                st.metric("Cities",   "50")
            with c_r:
                st.metric("Features", "10 columns")
                st.metric("Target",   "3 classes")
            st.caption("Low  ·  Medium  ·  High severity")

        st.markdown("")

        # ── Model card ────────────────────────────────────────
        with st.container(border=True):
            st.markdown("#### 🔬 Model Details")
            st.markdown("""
| Parameter | Value |
|:---|:---|
| Algorithm | Random Forest |
| Estimators | 200 trees |
| Feature Selection | sqrt |
| Class Weights | Balanced |
| Validation | 5-Fold Stratified CV |
""")

    with a2:
        # ── Features card ─────────────────────────────────────
        with st.container(border=True):
            st.markdown("#### ⚙️ Features Used")
            feats_info = {
                "🌧️ Rainfall (mm)"       : "Primary flood trigger",
                "💧 Water Level (m)"     : "River / reservoir depth",
                "📅 Month"               : "Seasonality signal",
                "📍 Location"            : "City encoded numerically",
                "📆 Date features"       : "Auto-derived from month",
                "⏱️ Duration"            : "Median-imputed from data",
            }
            for feat, desc in feats_info.items():
                st.markdown(f"**{feat}** — {desc}")

        st.markdown("")

        # ── Project files card ────────────────────────────────
        with st.container(border=True):
            st.markdown("#### 📂 Project Structure")
            st.code("""flood_prediction/
├── step1_data_loading.py
├── step2_preprocessing.py
├── step3_visualisation.py
├── step4_train_model.py
├── step5_streamlit_app.py  ← here
├── random_forest_model.pkl
├── feature_cols.pkl
├── data_cleaned_full.csv
└── data_model_ready.csv""", language="text")
