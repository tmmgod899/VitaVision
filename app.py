import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json
import base64
from datetime import date
from openai import OpenAI
from supabase import create_client

# =========================================
# Page configuration
# =========================================
st.set_page_config(
    page_title="VitaVision Health Analyzer",
    page_icon="IconVitaVision.png",
    layout="wide"
)

# =========================================
# Load ML Model + OpenAI
# =========================================
MODEL_PATH = "models/vitavision_hybrid_model_v1.pkl"

@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

ml_model = load_ml_model()

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None

OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini") if client else "gpt-4o-mini"

# =========================================
# Supabase Auth + Database
# =========================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception:
    supabase = None


if "language" not in st.session_state:
    st.session_state["language"] = "English"

if "disclaimer_agreed" not in st.session_state:
    st.session_state["disclaimer_agreed"] = False

if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

if "auth_role" not in st.session_state:
    st.session_state["auth_role"] = "guest"

if "auth_email" not in st.session_state:
    st.session_state["auth_email"] = None

# =========================================
# Disclaimer dialog
# =========================================
@st.dialog("\u00A0", dismissible=False)
def show_disclaimer():
    lang_c1, lang_c2, lang_c3 = st.columns([1, 2, 1])
    with lang_c2:
        disclaimer_lang = st.selectbox(
            "Language / اللغة",
            ["English", "العربية"],
            index=0 if st.session_state["language"] == "English" else 1,
            key="disclaimer_language"
        )

    is_ar = disclaimer_lang == "العربية"

    if not is_ar:
        title = "⚠️Medical Disclaimer⚠️"
        text = (
            "VitaVision provides health-related insights based on laboratory values and predefined medical reference ranges. "
            "It is intended for educational and awareness purposes only, and should not be considered a substitute for medical diagnosis, "
            "consultation, or treatment.\n\n"
            "Although VitaVision aims to provide useful and accurate interpretations, it does not take into account the user's medical history, "
            "health conditions, medications, or other clinical factors that may affect the results.\n\n"
            "By using VitaVision, you acknowledge the following:\n"
            "• The results are generated automatically and may not fully reflect your health condition.\n"
            "• VitaVision does not replace consultation with a licensed doctor or healthcare professional.\n"
            "• Any medical decisions should only be made after consulting a qualified specialist.\n\n"
            "You also agree to use VitaVision responsibly and understand that the developers are not responsible for any misuse, "
            "misinterpretation, or decisions made based on the provided information.\n\n"
            "If you have symptoms or abnormal results, please seek professional medical advice immediately."
        )
        button_text = "I Agree and Continue"
    else:
        title = "⚠️تنبيه طبي⚠️"
        text = (
            "يقدم نظام VitaVision معلومات وتحليلات صحية مبنية على نتائج الفحوصات المخبرية ونطاقات مرجعية طبية محدده "
            "وهو مخصص لأغراض تعليمية وتوعوية فقط ولا يُعتبر بديلاً عن التشخيص الطبي أو الاستشارة أو العلاج.\n\n"
            "رغم أن نظام VitaVision يسعى لتقديم تفسيرات دقيقة ومفيدة، إلا أنه لا يأخذ بعين الاعتبار التاريخ الطبي "
            "للمستخدم أو الحالات الصحية أو الأدوية أو العوامل السريرية الأخرى التي قد تؤثر على النتائج.\n\n"
            "باستخدامك لنظام VitaVision، فإنك تقر بما يلي:\n"
            "• النتائج يتم توليدها بشكل آلي وقد لا تعكس حالتك الصحية بشكل كامل.\n"
            "• لا يغني نظام VitaVision عن استشارة طبيب أو مختص صحي مرخص.\n"
            "• أي قرارات طبية يجب أن تتم فقط بعد الرجوع إلى مختص مؤهل.\n\n"
            "كما توافق على استخدام نظام VitaVision بمسؤولية، وتدرك أن مطوري التطبيق غير مسؤولين عن أي استخدام "
            "خاطئ أو تفسير غير دقيق أو قرارات يتم اتخاذها بناءً على هذه المعلومات.\n\n"
            "في حال وجود أي أعراض أو نتائج غير طبيعية، يرجى مراجعة مختص صحي بشكل فوري."
        )
        button_text = "أوافق وأتابع"

    dir_val = "rtl" if is_ar else "ltr"
    align_val = "right" if is_ar else "left"

    formatted_text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")

    st.html(f"""
<div style="
    border: 1px solid rgba(0,191,255,0.4);
    border-radius: 18px;
    padding: 28px 30px;
    background: linear-gradient(145deg, rgba(0,15,30,0.98), rgba(0,25,45,0.95));
    box-shadow: 0 8px 40px rgba(0,191,255,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
">
    <div style="
        text-align: center;
        color: #00BFFF;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 22px;
        letter-spacing: 0.3px;
        font-family: 'Segoe UI', sans-serif;
    ">
        {title}
    </div>
    <div style="
        width: 60px;
        height: 2px;
        margin: 0 auto 22px;
        border-radius: 2px;
    "></div>
    <div style="
        color: #D8E8F0;
        font-size: 14.5px;
        line-height: 1.9;
        direction: {dir_val};
        text-align: {align_val};
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        background: rgba(0,191,255,0.03);
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid rgba(0,191,255,0.1);
    ">
        {formatted_text}
    </div>
</div>
""")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if st.button(button_text, use_container_width=True, type="primary"):
        st.session_state["language"] = disclaimer_lang
        st.session_state["main_language"] = disclaimer_lang
        st.session_state["disclaimer_agreed"] = True
        st.rerun()


# =========================================
# Google Fonts
# =========================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# =========================================
# Global UI Styles
# =========================================
st.markdown("""
<style>

/* ── Base & Fonts ─────────────────────────── */
:root {
    --blue:        #00BFFF;
    --blue-dim:    rgba(0,191,255,0.18);
    --blue-border: rgba(0,191,255,0.30);
    --bg-card:     rgba(255,255,255,0.032);
    --bg-card-hover: rgba(0,191,255,0.06);
    --text-main:   #F0F4F8;
    --text-muted:  #8A9BAD;
    --text-sub:    #B8C8D8;
    --red:    #FF4B4B;
    --green:  #1DB954;
    --orange: #FFA500;
    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --shadow-blue: 0 4px 24px rgba(0,191,255,0.10);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif !important;
    background: #060d14 !important;
    color: var(--text-main) !important;
    width: 100% !important;
    min-height: 100% !important;
}
* {
    box-sizing: border-box !important;
}

.block-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 1.2rem 1.5rem 3rem !important;
}

/* ── Remove Streamlit chrome ─────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Dialog cleanup ──────────────────────── */
div[role="dialog"] header { display: none !important; }
div[role="dialog"] > div { padding-top: 0 !important; }
div[role="dialog"] { margin-top: -20px !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Navigation Tabs ─────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 6px;

    background: rgba(0,15,30,0.75);
    border: 1px solid rgba(0,191,255,0.18);

    border-radius: 18px;
    padding: 10px;

    width: fit-content;

    margin: 20px auto 30px auto;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.25),
        0 0 20px rgba(0,191,255,0.05);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: var(--text-muted) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 9px 26px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--blue) !important;
    background: rgba(0,191,255,0.06) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    background: rgba(0,191,255,0.12) !important;
    box-shadow: 0 0 0 1px rgba(0,191,255,0.25) !important;
}

/* ── Buttons ─────────────────────────────── */
div.stButton > button {
    border-radius: var(--radius-md) !important;
    height: 46px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: 1px solid var(--blue-border) !important;
    color: var(--text-main) !important;
    background: rgba(0,191,255,0.06) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}

div.stButton > button:hover {
    border-color: var(--blue) !important;
    background: rgba(0,191,255,0.16) !important;
    box-shadow: 0 0 16px rgba(0,191,255,0.20) !important;
    transform: translateY(-1px) !important;
}

div.stButton > button[kind="primary"],
div.stButton > button.primary {
    background: linear-gradient(135deg, #163845, #0e6081) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0,191,255,0.30) !important;
}

div.stDownloadButton > button {
    border-radius: var(--radius-md) !important;
    height: 46px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    background: linear-gradient(135deg, rgba(0,128,179,0.25), rgba(0,191,255,0.15)) !important;
    border: 1px solid var(--blue-border) !important;
    color: var(--blue) !important;
    transition: all 0.2s ease !important;
}

div.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(0,128,179,0.40), rgba(0,191,255,0.30)) !important;
    box-shadow: 0 0 20px rgba(0,191,255,0.25) !important;
}

/* ── Inputs ──────────────────────────────── */
input, textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-main) !important;
    font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif !important;
    transition: border-color 0.2s !important;
}

input:focus, textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(0,191,255,0.15) !important;
    outline: none !important;
}

[data-testid="stNumberInput"] button {
    border-color: var(--blue-border) !important;
    color: var(--blue) !important;
}

/* ── Selectbox ───────────────────────────── */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.10) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color 0.2s !important;
}

div[data-baseweb="select"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(0,191,255,0.15) !important;
}

/* ── Radio ───────────────────────────────── */
[data-testid="stRadio"] input[type="radio"] { accent-color: var(--blue) !important; }
[data-testid="stRadio"] label { gap: 10px !important; }

/* ── DataFrames / Tables ─────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,191,255,0.18) !important;
}

/* ── Divider ─────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 32px 0 !important; }

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(0,191,255,0.30); border-radius: 3px; }

/* ── Cards ───────────────────────────────── */
.vv-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.038), rgba(255,255,255,0.018));
    border: 1px solid var(--blue-border);
    border-radius: var(--radius-lg);
    padding: 26px 30px;
    margin-bottom: 20px;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    position: relative;
    overflow: hidden;
}

.vv-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,191,255,0.4), transparent);
}

.vv-card:hover {
    border-color: rgba(0,191,255,0.55);
    box-shadow: 0 6px 28px rgba(0,191,255,0.10);
    transform: translateY(-1px);
}

.vv-card-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--blue);
    margin-bottom: 12px;
    letter-spacing: 0.2px;
    font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
}

.vv-card-text {
    font-size: 15px;
    color: var(--text-sub);
    line-height: 1.75;
    font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
}

.vv-card-list {
    margin-top: 14px;
    padding-left: 20px;
    list-style: none;
    padding-left: 0;
}

.vv-card-list li {
    position: relative;
    padding-left: 20px;
    margin-bottom: 8px;
    color: #B8C8D8;
    font-size: 14.5px;
    line-height: 1.6;
}

.vv-card-list li::before {
    content: '•';
    position: absolute;
    left: 0;
    color: var(--blue);
    font-weight: 700;
    font-size: 16px;
}

/* RTL list arrows */
[dir="rtl"] .vv-card-list li { padding-left: 0; padding-right: 20px; }
[dir="rtl"] .vv-card-list li::before { left: auto; right: 0; }

/* ── Result card ─────────────────────────── */
.result-card {
    border-radius: var(--radius-lg);
    padding: 22px 26px;
    margin-bottom: 16px;
    background: linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}

.result-card:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.20);
    transform: translateY(-1px);
}

/* ── Section title ───────────────────────── */
.vv-section-title {
    font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
    font-weight: 800;
    color: #F0F4F8;
    margin-top: 34px;
    margin-bottom: 16px;
    letter-spacing: -0.3px;
    position: relative;
    display: flex;
    align-items: center;
    gap: 10px;
}

.vv-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,191,255,0.25), transparent);
}

/* ── Misc helpers ────────────────────────── */
.small-muted { color: var(--text-muted); font-size: 13px; }
.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* ── Stats mini-card ─────────────────────── */
.stat-mini {
    min-width: 0 !important;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    text-align: center;
    transition: border-color 0.2s;
}

.stat-mini:hover { border-color: var(--blue-border); }
.stat-number { font-size: 28px; font-weight: 800; line-height: 1.1; }
.stat-label  { font-size: 12px; color: var(--text-muted); margin-top: 4px; letter-spacing: 0.4px; text-transform: uppercase; }

/* ── Disclaimer banner ───────────────────── */
.disclaimer-banner {
    background: linear-gradient(135deg, rgba(255,193,7,0.08), rgba(255,152,0,0.05));
    border: 1px solid rgba(255,193,7,0.30);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-top: 24px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.disclaimer-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.disclaimer-title { font-size: 14px; font-weight: 700; color: #FFD54F; margin-bottom: 4px; }
.disclaimer-text  { font-size: 13.5px; color: #D4C5A0; line-height: 1.6; }



/* ── Input method cards ───────────────────── */
.input-method-wrap {
    margin-top: -6px;
    margin-bottom: 22px;
}
[data-testid="stRadio"] > div[role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 14px !important;
    width: 100% !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label {
    min-height: 76px !important;
    padding: 16px 18px !important;
    border-radius: 18px !important;
    border: 1px solid rgba(0,191,255,0.22) !important;
    background: linear-gradient(145deg, rgba(255,255,255,0.038), rgba(255,255,255,0.018)) !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.16) !important;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
    cursor: pointer !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(0,191,255,0.65) !important;
    background: linear-gradient(145deg, rgba(0,191,255,0.11), rgba(255,255,255,0.025)) !important;
    box-shadow: 0 8px 28px rgba(0,191,255,0.14) !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    border-color: rgba(0,191,255,0.95) !important;
    background: linear-gradient(135deg, rgba(0,191,255,0.18), rgba(0,95,135,0.18)) !important;
    box-shadow: 0 0 0 1px rgba(0,191,255,0.28), 0 8px 30px rgba(0,191,255,0.16) !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label p {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: #DDEAF3 !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) p {
    color: #00BFFF !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label:first-child p::before {
    content: '⌨️';
    display: block;
    font-size: 22px;
    margin-bottom: 4px;
}
[data-testid="stRadio"] > div[role="radiogroup"] label:nth-child(2) p::before {
    content: '📄';
    display: block;
    font-size: 22px;
    margin-bottom: 4px;
}
[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* ── CSV preview expander ─────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,191,255,0.20) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.025) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #00BFFF !important;
    font-weight: 800 !important;
}



/* ── VitaVision landing redesign ───────────── */
.vv-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(0,191,255,0.18);
    border-radius: 28px;
    padding: 38px 34px 32px;
    margin: 10px 0 28px;
    background:
        radial-gradient(circle at 18% 20%, rgba(0,191,255,0.18), transparent 28%),
        radial-gradient(circle at 88% 10%, rgba(29,185,84,0.11), transparent 30%),
        linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
    box-shadow: 0 18px 60px rgba(0,0,0,0.25), 0 0 42px rgba(0,191,255,0.08);
}

.vv-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 42px 42px;
    opacity: 0.18;
    pointer-events: none;
}

.vv-hero-inner {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 540px;
    gap: 26px;
    align-items: center;
}

.vv-hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    width: fit-content;
    padding: 7px 13px;
    border-radius: 999px;
    color: #00BFFF;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.4px;
    background: rgba(0,191,255,0.09);
    border: 1px solid rgba(0,191,255,0.24);
    margin-bottom: 14px;
}

.vv-hero-title {
    font-size: clamp(38px, 6vw, 64px);
    line-height: 0.96;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0 0 14px;
    color: #fff;
}

.vv-hero-title span {
    color: #00BFFF;
    text-shadow: 0 0 18px rgba(0,191,255,0.35);
}

.vv-hero-subtitle {
    max-width: 650px;
    color: #AFC4D5;
    font-size: 16px;
    line-height: 1.75;
    margin-bottom: 20px;
}

.vv-hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
}

.vv-hero-badge {
    padding: 9px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    color: #D7E8F2;
    font-size: 12.5px;
    font-weight: 700;
}

.vv-hero-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    max-width: 560px;
    margin-top: 20px;
}

.vv-hero-stat {
    padding: 12px 13px;
    border-radius: 16px;
    background: rgba(1,19,34,0.68);
    border: 1px solid rgba(0,191,255,0.18);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.045), 0 10px 24px rgba(0,0,0,0.14);
}

.vv-hero-stat strong {
    display: block;
    color: #FFFFFF;
    font-size: 21px;
    line-height: 1;
    margin-bottom: 6px;
}

.vv-hero-stat span {
    display: block;
    color: #8FA8BA;
    font-size: 11.5px;
    line-height: 1.35;
}

.vv-hero-panel {
    border-radius: 24px;
    padding: 24px;
    background: rgba(0,15,30,0.58);
    border: 1px solid rgba(0,191,255,0.20);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

.vv-mini-progress-title {
    font-size: 15px;
    font-weight: 900;
    color: #F0F4F8;
    margin-bottom: 8px;
}

.vv-snake-flow {
    position: relative;
    display: grid;
    direction: ltr;
    grid-template-columns: minmax(220px, 1fr) 24px minmax(220px, 1fr);
    grid-auto-rows: 75px;
    row-gap: 0px;
    min-height: 0;
    margin-top: 10px;
    padding: 6px 0;
}

.vv-snake-flow::before {
    content: "";
    display: none;
}

.vv-snake-links {
    display: none;
}

.vv-curve-link {
    position: absolute;
    z-index: 1;
    left: calc(50% - 36px);
    width: 72px;
    height: 85px;
    pointer-events: none;
    background-repeat: no-repeat;
    background-size: 100% 100%;
    filter: drop-shadow(0 0 6px rgba(0,191,255,0.48));
    opacity: 0.95;
}

.vv-curve-link::before,
.vv-curve-link::after {
    content: none;
    display: none;
}

.vv-curve-link.c12 { top: 50px; }
.vv-curve-link.c23 { top: 130px; }
.vv-curve-link.c34 { top: 210px; }

.vv-snake-flow.ltr .vv-curve-link.c12,
.vv-snake-flow.ltr .vv-curve-link.c34,
.vv-snake-flow.rtl .vv-curve-link.c23 {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 72 127' preserveAspectRatio='none'%3E%3Cpath d='M1 6 C72 6 1 121 71 121' fill='none' stroke='%2300BFFF' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
}

.vv-snake-flow.ltr .vv-curve-link.c23,
.vv-snake-flow.rtl .vv-curve-link.c12,
.vv-snake-flow.rtl .vv-curve-link.c34 {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 72 127' preserveAspectRatio='none'%3E%3Cpath d='M71 6 C1 6 72 121 1 121' fill='none' stroke='%2300BFFF' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
}

.vv-snake-step {
    position: relative;
    z-index: 2;
    width: auto;
    height: 95px;
    max-width: 220px;
    min-height: 95px;
    padding: 12px 16px;
    box-sizing: border-box;
    padding: 8px 12px;
    border-radius: 16px;
    background: #011322;
    border: 1px solid rgba(0,191,255,0.30);
    box-shadow: 0 8px 20px rgba(0,0,0,0.20);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.vv-snake-flow.ltr .vv-snake-step {
    direction: ltr;
    text-align: left;
}

.vv-snake-flow.rtl .vv-snake-step {
    direction: rtl;
    text-align: right;
}

.vv-snake-step::before {
    content: "";
    display: none;
}

.vv-snake-step::after {
    content: "";
    display: none;
}

.vv-snake-step.s1,
.vv-snake-step.s3 {
    grid-column: 1;
}

.vv-snake-step.s2,
.vv-snake-step.s4 {
    grid-column: 3;
}

.vv-snake-flow.rtl .vv-snake-step.s1,
.vv-snake-flow.rtl .vv-snake-step.s3 {
    grid-column: 3;
}

.vv-snake-flow.rtl .vv-snake-step.s2,
.vv-snake-flow.rtl .vv-snake-step.s4 {
    grid-column: 1;
}

.vv-snake-step.s1 { grid-row: 1; }
.vv-snake-step.s2 { grid-row: 2; }
.vv-snake-step.s3 { grid-row: 3; }
.vv-snake-step.s4 { grid-row: 4; }

.vv-snake-step.s1::before,
.vv-snake-step.s3::before {
    left: 100%;
}

.vv-snake-step.s2::before,
.vv-snake-step.s4::before {
    right: 100%;
}

.vv-snake-step.s1::after,
.vv-snake-step.s3::after {
    right: -18px;
}

.vv-snake-step.s2::after,
.vv-snake-step.s4::after {
    left: -18px;
}

.vv-snake-flow.rtl .vv-snake-step.s1::before,
.vv-snake-flow.rtl .vv-snake-step.s3::before {
    left: auto;
    right: 100%;
}

.vv-snake-flow.rtl .vv-snake-step.s2::before,
.vv-snake-flow.rtl .vv-snake-step.s4::before {
    right: auto;
    left: 100%;
}

.vv-snake-flow.rtl .vv-snake-step.s1::after,
.vv-snake-flow.rtl .vv-snake-step.s3::after {
    right: auto;
    left: -18px;
}

.vv-snake-flow.rtl .vv-snake-step.s2::after,
.vv-snake-flow.rtl .vv-snake-step.s4::after {
    left: auto;
    right: -18px;
}

.vv-snake-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
}

.vv-snake-num {
    flex: 0 0 auto;
    width: 28px;
    height: 28px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    color: #00111E;
    font-size: 12px;
    font-weight: 900;
    background: linear-gradient(135deg, #00BFFF, #47E6FF);
    box-shadow: 0 0 16px rgba(0,191,255,0.35);
}

.vv-step-title {
    color: #EAF5FB;
    font-size: 13.5px;
    font-weight: 900;
}

.vv-step-text {
    color: #8FA8BA;
    font-size: 11.2px;
    line-height: 1.32;
}

.vv-workflow {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 18px 0 30px;
}

.vv-workflow-card {
    position: relative;
    min-height: 112px;
    padding: 18px 16px;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.018));
    border: 1px solid rgba(0,191,255,0.18);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.vv-workflow-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0,191,255,0.52);
    box-shadow: 0 10px 28px rgba(0,191,255,0.10);
}

.vv-workflow-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.vv-workflow-title {
    font-size: 14px;
    font-weight: 900;
    color: #F0F4F8;
    margin-bottom: 5px;
}

.vv-workflow-text {
    font-size: 12.5px;
    color: #8EA4B5;
    line-height: 1.55;
}

.vv-start-card {
    border-radius: 24px;
    padding: 22px 22px 18px;
    margin: 4px 0 24px;
    background: linear-gradient(145deg, rgba(0,191,255,0.075), rgba(255,255,255,0.018));
    border: 1px solid rgba(0,191,255,0.24);
}

.vv-start-card-title {
    font-size: 18px;
    font-weight: 900;
    color: #F0F4F8;
    margin-bottom: 6px;
}

.vv-start-card-text {
    font-size: 13.5px;
    color: #8EA4B5;
    line-height: 1.65;
}

@media (max-width: 850px) {
    .vv-hero { padding: 28px 20px; border-radius: 22px; }
    .vv-hero-inner { grid-template-columns: 1fr; }
    .vv-hero-panel { padding: 18px; }
    .vv-workflow { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 520px) {
    .vv-hero-title { letter-spacing: -1px; }
    .vv-hero-badges { gap: 8px; }
    .vv-hero-badge { width: 100%; text-align: center; }
    .vv-hero-stats { grid-template-columns: 1fr; }
    .vv-snake-flow { grid-template-columns: 1fr; gap: 10px; }
    .vv-snake-flow::before,
    .vv-curve-link,
    .vv-snake-step::before,
    .vv-snake-step::after { display: none; }
    .vv-snake-links,
    .vv-snake-svg { display: none; }
    .vv-snake-step,
    .vv-snake-step.s1,
    .vv-snake-step.s2,
    .vv-snake-step.s3,
    .vv-snake-step.s4 { grid-column: 1; width: 100%; }
    .vv-workflow { grid-template-columns: 1fr; }
}


/* ── Multi-page flow cards ─────────────────── */
.vv-choice-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    margin: 18px 0 26px;
}

.vv-choice-card {
    min-height: 210px;
    padding: 26px 24px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.018));
    border: 1px solid rgba(0,191,255,0.22);
    box-shadow: 0 12px 34px rgba(0,0,0,0.18);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.vv-choice-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,191,255,0.65);
    box-shadow: 0 18px 46px rgba(0,191,255,0.12);
}

.vv-choice-icon {
    font-size: 38px;
    margin-bottom: 14px;
}

.vv-choice-title {
    font-size: 21px;
    font-weight: 900;
    color: #F0F4F8;
    margin-bottom: 9px;
}

.vv-choice-text {
    font-size: 14px;
    color: #8EA4B5;
    line-height: 1.7;
}

.vv-page-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin: 18px 0 12px;
    padding: 13px 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(0,191,255,0.16);
}

.vv-page-pill {
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(0,191,255,0.08);
    border: 1px solid rgba(0,191,255,0.22);
    color: #00BFFF;
    font-size: 12px;
    font-weight: 800;
}

.vv-progress-flow {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 14px 0 24px;
}

.vv-progress-step {
    padding: 10px 12px;
    border-radius: 14px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    color: #7A9BB5;
    font-size: 12px;
    font-weight: 800;
    text-align: center;
}

.vv-progress-step.active {
    background: rgba(0,191,255,0.10);
    border-color: rgba(0,191,255,0.42);
    color: #00BFFF;
}

@media (max-width: 700px) {
    .vv-choice-grid { grid-template-columns: 1fr; }
    .vv-page-topbar { align-items: stretch; flex-direction: column; }
    .vv-progress-flow { grid-template-columns: 1fr; }
}



/* ── Clean input section spacing ───────────── */
.vv-hero + .vv-progress-flow {
    margin-top: 0 !important;
}
.vv-progress-flow {
    margin-bottom: 18px !important;
}
[data-testid="stFileUploader"] {
    border-radius: 18px !important;
}

/* ── Smart AI upload review ───────────────── */
.vv-smart-box {
    border-radius: 22px;
    padding: 22px;
    margin: 16px 0 20px;
    background:
        radial-gradient(circle at top left, rgba(0,191,255,0.12), transparent 35%),
        linear-gradient(145deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
    border: 1px solid rgba(0,191,255,0.25);
}

.vv-smart-title {
    font-size: 20px;
    font-weight: 900;
    color: #F0F4F8;
    margin-bottom: 8px;
}

.vv-smart-text {
    font-size: 14px;
    color: #8EA4B5;
    line-height: 1.7;
}

.vv-smart-note {
    padding: 13px 15px;
    border-radius: 14px;
    background: rgba(255,193,7,0.07);
    border: 1px solid rgba(255,193,7,0.22);
    color: #D6C68D;
    font-size: 13px;
    line-height: 1.65;
    margin: 12px 0 18px;
}


/* ── Auth bar / admin ─────────────────── */
.vv-auth-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;border-radius:16px;margin:8px 0 16px;background:rgba(255,255,255,0.028);border:1px solid rgba(0,191,255,0.16);}
.vv-auth-pill{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border-radius:999px;color:#00BFFF;background:rgba(0,191,255,0.08);border:1px solid rgba(0,191,255,0.22);font-size:12px;font-weight:800;}
.vv-auth-sub{color:#8EA4B5;font-size:12.5px;margin-top:4px;}
.vv-admin-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0;}
.vv-admin-card{padding:16px;border-radius:16px;background:rgba(255,255,255,0.032);border:1px solid rgba(0,191,255,0.16);}
.vv-admin-num{font-size:28px;font-weight:900;color:#00BFFF;}
.vv-admin-label{font-size:12px;color:#8EA4B5;text-transform:uppercase;letter-spacing:0.4px;}

/* ── Responsive ──────────────────────────── */
@media (max-width: 768px) {
    .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    .stat-mini {
        min-width: 0 !important;
        padding: 10px 6px !important;
    }

    .stat-label {
        font-size: 10px !important;
        white-space: normal !important;
        word-break: break-word !important;
    }

    [data-testid="stRadio"] > div[role="radiogroup"] {
        grid-template-columns: 1fr !important;
        gap: 10px !important;
    }

    [data-testid="stRadio"] > div[role="radiogroup"] label {
        min-height: 64px !important;
        padding: 13px 14px !important;
    }

}

/* ── iPhone layout hardening ──────────────── */
@media (max-width: 520px) {
    .block-container {
        padding-left: 0.55rem !important;
        padding-right: 0.55rem !important;
        padding-bottom: 2rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        width: 100% !important;
        max-width: 100% !important;
        padding: 7px !important;
        gap: 4px !important;
        margin: 14px auto 18px auto !important;
    }

    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        padding: 8px 6px !important;
        font-size: 12px !important;
        text-align: center !important;
    }

    .vv-auth-bar {
        flex-direction: column;
        align-items: stretch;
        text-align: center;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        height: auto !important;
        min-height: 44px !important;
        padding: 10px 8px !important;
        white-space: normal !important;
        line-height: 1.25 !important;
    }

    div.stButton > button p,
    div.stDownloadButton > button p {
        white-space: normal !important;
        line-height: 1.25 !important;
        font-size: 12.5px !important;
    }

    input,
    textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        font-size: 16px !important;
    }

    .vv-hero {
        padding: 20px 12px 18px;
        border-radius: 20px;
        margin: 8px 0 18px;
    }

    .vv-hero-inner {
        grid-template-columns: 1fr !important;
        gap: 18px;
    }

    .vv-hero-kicker {
        font-size: 10.5px;
        padding: 6px 10px;
        margin-bottom: 10px;
    }

    .vv-hero-title {
        font-size: clamp(36px, 12.5vw, 48px);
        line-height: 1;
        letter-spacing: -1px;
        margin-bottom: 10px;
    }

    .vv-hero-subtitle {
        font-size: 13.5px;
        line-height: 1.62;
        margin-bottom: 14px;
    }

    .vv-hero-badges {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }

    .vv-hero-badge {
        width: auto;
        min-width: 0;
        padding: 8px 7px;
        font-size: 10.6px;
        line-height: 1.25;
        text-align: center;
        white-space: normal;
    }

    .vv-hero-stats {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 7px;
        margin-top: 14px;
    }

    .vv-hero-stat {
        min-width: 0;
        padding: 9px 6px;
        border-radius: 12px;
        text-align: center;
    }

    .vv-hero-stat strong {
        font-size: 18px;
        margin-bottom: 4px;
    }

    .vv-hero-stat span {
        font-size: 9.4px;
        line-height: 1.25;
    }

    .vv-hero-panel {
        padding: 14px 11px;
        border-radius: 18px;
        overflow: hidden;
    }

    .vv-mini-progress-title {
        font-size: 13.5px;
        margin-bottom: 10px;
        text-align: center;
    }

    .vv-snake-flow {
        grid-template-columns: minmax(0, 1fr) 22px minmax(0, 1fr) !important;
        grid-auto-rows: 104px;
        row-gap: 8px;
        column-gap: 0;
        gap: 8px 0 !important;
        margin-top: 8px;
        padding: 4px 0;
        min-width: 0;
    }

    .vv-curve-link {
        display: block !important;
        left: calc(50% - 24px);
        width: 48px;
        height: 112px;
        filter: drop-shadow(0 0 5px rgba(0,191,255,0.42));
    }

    .vv-curve-link.c12 { top: 52px; }
    .vv-curve-link.c23 { top: 164px; }
    .vv-curve-link.c34 { top: 276px; }

    .vv-snake-flow.ltr .vv-curve-link.c12,
    .vv-snake-flow.ltr .vv-curve-link.c34,
    .vv-snake-flow.rtl .vv-curve-link.c23 {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 112' preserveAspectRatio='none'%3E%3Cpath d='M1 0 C48 20 1 92 47 112' fill='none' stroke='%2300BFFF' stroke-width='2.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    }

    .vv-snake-flow.ltr .vv-curve-link.c23,
    .vv-snake-flow.rtl .vv-curve-link.c12,
    .vv-snake-flow.rtl .vv-curve-link.c34 {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 112' preserveAspectRatio='none'%3E%3Cpath d='M47 0 C0 20 47 92 1 112' fill='none' stroke='%2300BFFF' stroke-width='2.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    }

    .vv-snake-step,
    .vv-snake-step.s1,
    .vv-snake-step.s2,
    .vv-snake-step.s3,
    .vv-snake-step.s4 {
        width: 100%;
        max-width: none;
        height: 104px;
        min-height: 104px;
        padding: 9px 8px;
        border-radius: 13px;
        overflow: hidden;
    }

    .vv-snake-step.s1,
    .vv-snake-step.s3 {
        grid-column: 1 !important;
    }

    .vv-snake-step.s2,
    .vv-snake-step.s4 {
        grid-column: 3 !important;
    }

    .vv-snake-flow.rtl .vv-snake-step.s1,
    .vv-snake-flow.rtl .vv-snake-step.s3 {
        grid-column: 3 !important;
    }

    .vv-snake-flow.rtl .vv-snake-step.s2,
    .vv-snake-flow.rtl .vv-snake-step.s4 {
        grid-column: 1 !important;
    }

    .vv-snake-head {
        gap: 5px;
        margin-bottom: 4px;
        align-items: center;
    }

    .vv-snake-num {
        width: 23px;
        height: 23px;
        border-radius: 8px;
        font-size: 10px;
    }

    .vv-step-title {
        font-size: 11.2px;
        line-height: 1.15;
    }

    .vv-step-text {
        font-size: 9.2px;
        line-height: 1.28;
    }

    .vv-workflow,
    .vv-choice-grid,
    .vv-progress-flow {
        grid-template-columns: 1fr !important;
    }

    .vv-choice-card {
        min-height: auto;
        padding: 20px 18px;
    }

    .vv-page-topbar {
        flex-direction: column;
        align-items: stretch;
    }
}

@media (max-width: 380px) {
    .vv-hero {
        padding: 18px 10px 16px;
    }

    .vv-hero-panel {
        padding: 12px 9px;
    }

    .vv-snake-flow {
        grid-template-columns: minmax(0, 1fr) 18px minmax(0, 1fr) !important;
        grid-auto-rows: 108px;
    }

    .vv-curve-link {
        left: calc(50% - 21px);
        width: 42px;
        height: 116px;
    }

    .vv-curve-link.c12 { top: 54px; }
    .vv-curve-link.c23 { top: 170px; }
    .vv-curve-link.c34 { top: 286px; }

    .vv-snake-step,
    .vv-snake-step.s1,
    .vv-snake-step.s2,
    .vv-snake-step.s3,
    .vv-snake-step.s4 {
        height: 108px;
        min-height: 108px;
        padding: 8px 7px;
    }

    .vv-step-title {
        font-size: 10.6px;
    }

    .vv-step-text {
        font-size: 8.8px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================
# Language selector
# =========================================
lang_left, lang_center, lang_right = st.columns([2, 1, 2])

old_language = st.session_state.get("language", "English")

with lang_center:
    language = st.selectbox(
        "",
        ["English", "العربية"],
        index=0 if st.session_state["language"] == "English" else 1,
        label_visibility="collapsed",
        key="main_language"
    )

if language != old_language:
    st.session_state["language"] = language
    st.session_state["language_changed"] = True
else:
    st.session_state["language"] = language

is_arabic = language == "العربية"

# Direction styles
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    direction: {"rtl" if is_arabic else "ltr"};
    text-align: {"right" if is_arabic else "left"};
}}

.vv-card-text, .vv-card-list, .small-muted,
.disclaimer-text, .result-explanation,
li, p, label {{
    direction: {"rtl" if is_arabic else "ltr"};
    text-align: {"right" if is_arabic else "left"};
    font-family: {"'Cairo', sans-serif" if is_arabic else "'Plus Jakarta Sans', sans-serif"} !important;
}}

.vv-card-list {{
    padding-right: {"20px" if is_arabic else "0"} !important;
    padding-left: {"0" if is_arabic else "20px"} !important;
}}

.vv-card-list li {{
    padding-left: {"0" if is_arabic else "20px"} !important;
    padding-right: {"20px" if is_arabic else "0"} !important;
    text-align: {"right" if is_arabic else "left"} !important;
}}

.vv-card-list li::before {{
    left: {"auto" if is_arabic else "0"} !important;
    right: {"0" if is_arabic else "auto"} !important;
}}

[data-testid="stRadio"] label > div:last-child {{
    margin-right: {"8px" if is_arabic else "0"} !important;
    margin-left: {"0" if is_arabic else "8px"} !important;
}}

.vv-section-title::after {{
    background: linear-gradient({"270deg" if is_arabic else "90deg"}, rgba(0,191,255,0.25), transparent) !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================================
# Helpers
# =========================================
def tr(en, ar):
    return en if language == "English" else ar

def section_title(text_value, size=26, icon=""):
    icon_html = f'<span style="font-size:{size-4}px">{icon}</span>' if icon else ""
    st.html(f"""
    <div class="vv-section-title" style="font-size:{size}px;">
        {icon_html}{text_value}
    </div>
    """)


# =========================================
# Auth helpers
# =========================================
def get_current_user_id():
    user = st.session_state.get("auth_user")
    return user.get("id") if user else None

def load_user_profile(user_id):
    if supabase is None or not user_id:
        return None
    try:
        res = supabase.table("profiles").select("*").eq("id", user_id).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None

def ensure_profile(user_id, email, full_name=""):
    if supabase is None or not user_id:
        return
    try:
        existing = supabase.table("profiles").select("id").eq("id", user_id).limit(1).execute()
        if not existing.data:
            supabase.table("profiles").insert({
                "id": user_id,
                "email": email,
                "full_name": full_name,
                "role": "user"
            }).execute()
    except Exception:
        pass

def auth_sign_in(email, password):
    if supabase is None:
        return False, tr("Supabase is not configured.", "لم يتم إعداد Supabase.")
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        user = res.user
        if not user:
            return False, tr("Login failed.", "فشل تسجيل الدخول.")
        profile = load_user_profile(user.id)
        role = profile.get("role", "user") if profile else "user"
        st.session_state["auth_user"] = {"id": user.id, "email": user.email}
        st.session_state["auth_email"] = user.email
        st.session_state["auth_role"] = role
        return True, tr("Logged in successfully.", "تم تسجيل الدخول بنجاح.")
    except Exception as e:
        return False, tr(f"Login failed: {e}", f"فشل تسجيل الدخول: {e}")

def auth_register(full_name, email, password, gender=None, birth_date=None):
    if supabase is None:
        return False, tr("Supabase is not configured.", "لم يتم إعداد Supabase.")
    try:
        res = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name, "gender": gender, "birth_date": birth_date}}
        })
        user = res.user
        if not user:
            return False, tr("Registration failed.", "فشل إنشاء الحساب.")
        ensure_profile(user.id, user.email, full_name)
        st.session_state["auth_user"] = {"id": user.id, "email": user.email}
        st.session_state["auth_email"] = user.email
        st.session_state["auth_role"] = "user"
        return True, tr("Account created successfully.", "تم إنشاء الحساب بنجاح.")
    except Exception as e:
        return False, tr(f"Registration failed: {e}", f"فشل إنشاء الحساب: {e}")

def auth_logout():
    try:
        if supabase is not None:
            supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state["auth_user"] = None
    st.session_state["auth_email"] = None
    st.session_state["auth_role"] = "guest"
    st.session_state["app_page"] = "choose_input"
    st.rerun()

def save_analysis_log(input_method, results_df):
    if supabase is None or results_df is None or results_df.empty:
        return
    user_id = get_current_user_id()
    if not user_id:
        return
    try:
        supabase.table("analysis_logs").insert({
            "user_id": user_id,
            "mode": "registered",
            "input_method": input_method,
            "results": results_df.to_dict(orient="records")
        }).execute()
    except Exception:
        pass

def render_auth_bar():
    role = st.session_state.get("auth_role", "guest")

    left_space, center_col, right_space = st.columns([3,2,3])

    with center_col:
        if role == "guest":
            if st.button(
                tr("Login / Sign Up", "تسجيل الدخول / تسجيل جديد"),
                use_container_width=True,
                key="top_auth_entry"
            ):
                st.session_state["app_page"] = "login_page"
                st.rerun()

        else:
            if st.button(
                tr("Logout", "تسجيل الخروج"),
                use_container_width=True,
                key="top_auth_logout"
            ):
                auth_logout()
                st.rerun()

def render_login_page():
    section_title(tr("Login", "تسجيل الدخول"), 24, "👤")
    email = st.text_input(tr("Email", "البريد الإلكتروني"), key="login_email")
    password = st.text_input(tr("Password", "كلمة المرور"), type="password", key="login_password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button(tr("Login", "دخول"), use_container_width=True, type="primary"):
            ok, msg = auth_sign_in(email, password)
            if ok:
                st.success(msg)
                st.session_state["app_page"] = "choose_input"
                st.rerun()
            else:
                st.error(msg)
    with c2:
        if st.button(tr("Continue as Guest", "المتابعة كضيف"), use_container_width=True):
            st.session_state["app_page"] = "choose_input"
            st.rerun()

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:18px; color:#8EA4B5; font-size:14px;">
            {tr("Don't have an account?", "ليس لديك حساب؟")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    signup_col1, signup_col2, signup_col3 = st.columns([3.2, 3.6, 3.2])
    with signup_col2:
        if st.button(tr("Sign Up", "تسجيل جديد"), use_container_width=True, key="login_to_register"):
            st.session_state["app_page"] = "register_page"
            st.rerun()

def render_register_page():
    section_title(tr("Create Account", "إنشاء حساب"), 24, "📝")
    full_name = st.text_input(tr("Full Name", "الاسم الكامل"), key="register_name")
    gender_col, birth_col = st.columns(2)
    with gender_col:
        gender = st.selectbox(tr("Gender", 'الجنس'), [tr("Male", 'ذكر'), tr("Female", 'أنثى'), tr("Prefer not to say", 'أفضل عدم الإجابة')], key="register_gender")
    with birth_col:
        birth_date_value = st.date_input(tr("Date of Birth", 'تاريخ الميلاد'), value=None, min_value=date(1900, 1, 1), max_value=date.today(), key="register_birth_date")
    birth_date = birth_date_value.isoformat() if birth_date_value else None
    email = st.text_input(tr("Email", "البريد الإلكتروني"), key="register_email")
    password = st.text_input(tr("Password", "كلمة المرور"), type="password", key="register_password")
    st.markdown(
        """
        <style>
        div[data-testid="stPopover"] button {
            background: transparent !important;
            border: 0 !important;
            color: #00BFFF !important;
            text-decoration: underline !important;
            padding: 0 !important;
            min-height: auto !important;
            box-shadow: none !important;
            font-weight: 800 !important;
        }
        div[data-testid="stPopover"] button:hover { color: #4DDCFF !important; }
        [data-testid="stElementToolbar"] { display: none !important; }
        div[data-testid="stPopover"] [data-testid="stElementToolbar"],
        div[data-testid="stPopover"] [data-testid="stElementToolbarButton"],
        div[data-testid="stPopover"] button[title*="Copy"],
        div[data-testid="stPopover"] button[aria-label*="Copy"] { display: none !important; }
        .vv-terms-tail { padding-top: 7px; color: #FFFFFF; font-weight: 700; white-space: nowrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    terms_area = st.container()
    with terms_area:
        if is_arabic:
            spacer_col, terms_col = st.columns([6.0, 4.0])
            with terms_col:
                accepted_terms = st.checkbox('أوافق على شروط الاستخدام وسياسة الخصوصية', key="register_terms_accepted")
                terms_popup = st.popover('شروط الاستخدام وسياسة الخصوصية')
        else:
            terms_col, spacer_col = st.columns([4.0, 6.0])
            with terms_col:
                accepted_terms = st.checkbox("I agree to the Terms of Use and Privacy Policy", key="register_terms_accepted")
                terms_popup = st.popover("View Terms of Use and Privacy Policy")
        with terms_popup:
            st.markdown(tr("#### Terms of Use and Privacy Policy", '#### شروط الاستخدام وسياسة الخصوصية'))
            st.markdown(tr(
                "By using VitaVision, you agree to the following:\n\n1. VitaVision provides educational and awareness-based health information only and does not provide medical diagnosis.\n2. Results and insights depend on the data entered by the user and may not fully reflect the user's health condition.\n3. You should consult a doctor or qualified healthcare professional before making any medical or treatment decisions.\n4. You are responsible for the accuracy of the information you enter.\n5. Account data and analysis history may be stored to improve the user experience within the app.\n6. Your personal data will not be shared with external parties without your permission, unless required by law or system operation.\n7. VitaVision does not replace medical tests, diagnosis, or professional healthcare follow-up.",
                'باستخدامك لتطبيق VitaVision، فإنك توافق على ما يلي:\n\n1. التطبيق يقدم معلومات صحية توعوية وتعليمية فقط، ولا يعتبر تشخيصًا طبيًا.\n2. النتائج والتحليلات تعتمد على البيانات التي يدخلها المستخدم، وقد لا تعكس الحالة الصحية الكاملة.\n3. يجب استشارة طبيب أو مختص صحي قبل اتخاذ أي قرار طبي أو علاجي.\n4. المستخدم مسؤول عن صحة البيانات التي يدخلها في التطبيق.\n5. قد يتم حفظ بيانات الحساب وسجل التحليلات لتحسين تجربة المستخدم داخل التطبيق.\n6. لن يتم مشاركة بياناتك الشخصية مع أطراف خارجية بدون إذنك، إلا إذا تطلب ذلك النظام أو القانون.\n7. استخدام التطبيق لا يغني عن الفحوصات الطبية أو المتابعة مع المختصين.'
            ))
    c1, c2 = st.columns(2)
    with c1:
        if st.button(tr("Create Account", "إنشاء الحساب"), use_container_width=True, type="primary"):
            if not accepted_terms:
                st.warning(tr(
                    "Please read and accept the terms before creating an account.",
                    "يجب قراءة الشروط والموافقة عليها قبل إنشاء الحساب."
                ))
            else:
                ok, msg = auth_register(full_name, email, password, gender, birth_date)
                if ok:
                    st.success(msg)
                    st.session_state["app_page"] = "choose_input"
                    st.rerun()
                else:
                    st.error(msg)
    with c2:
        if st.button(tr("Back", "رجوع"), use_container_width=True):
            st.session_state["app_page"] = "choose_input"
            st.rerun()

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:18px; color:#8EA4B5; font-size:14px;">
            {tr("Already have an account?", "لديك حساب بالفعل؟")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    login_col1, login_col2, login_col3 = st.columns([3.2, 3.6, 3.2])
    with login_col2:
        if st.button(tr("Login", "تسجيل الدخول"), use_container_width=True, key="register_to_login"):
            st.session_state["app_page"] = "login_page"
            st.rerun()

def render_admin_page():
    render_header()
    section_title(tr("Admin Dashboard", "لوحة تحكم الأدمن"), 24, "🛡️")
    if st.session_state.get("auth_role") != "admin":
        st.error(tr("You do not have admin access.", "لا تملك صلاحية الأدمن."))
        if st.button(tr("Back", "رجوع"), use_container_width=True):
            st.session_state["app_page"] = "choose_input"
            st.rerun()
        return

    users_count = 0
    logs_df = pd.DataFrame()
    try:
        users = supabase.table("profiles").select("*").execute()
        users_count = len(users.data or [])
    except Exception:
        pass
    try:
        logs = supabase.table("analysis_logs").select("*").order("created_at", desc=True).limit(20).execute()
        logs_df = pd.DataFrame(logs.data or [])
    except Exception:
        pass

    st.html(f"""
<div class="vv-admin-grid">
    <div class="vv-admin-card"><div class="vv-admin-num">{users_count}</div><div class="vv-admin-label">{tr("Users", "المستخدمين")}</div></div>
    <div class="vv-admin-card"><div class="vv-admin-num">{len(logs_df)}</div><div class="vv-admin-label">{tr("Recent Logs", "آخر التحليلات")}</div></div>
</div>
""")
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info(tr("No logs available or access is restricted.", "لا توجد سجلات أو الصلاحية محدودة."))
    if st.button(tr("Back to App", "الرجوع للتطبيق"), use_container_width=True):
        st.session_state["app_page"] = "choose_input"
        st.rerun()


# =========================================
# Navigation tabs
# =========================================
if st.session_state.get("app_page", "choose_input") not in ["login_page", "register_page", "admin_page"]:
    render_auth_bar()

home_tab, about_tab, contact_tab = st.tabs([
    tr("Home", " الرئيسية"),
    tr("About", "عن المشروع"),
    tr("Contact", "تواصل معنا")
])

# =========================================
# Header
# =========================================
def render_header():
    subtitle = tr(
    "Smart Insight for Vitamin Health",
    "Smart Insight for Vitamin Health"
)
    dir_val = "rtl" if is_arabic else "ltr"
    align = "right" if is_arabic else "left"

    st.html(f"""

<div class="vv-hero" style="direction:{dir_val}; text-align:{align}; margin-bottom:22px;">
    <div class="vv-hero-inner">
        <div>
            <div class="vv-hero-kicker">
                {tr("AI Health Companion", "رفيق صحي بالذكاء الاصطناعي")}
            </div>
            <h1 class="vv-hero-title">
                <span>Vita</span>Vision
            </h1>
            <div class="vv-hero-subtitle">
                {subtitle}
            </div>
            <div class="vv-hero-badges">
                <div class="vv-hero-badge">⚡ {tr("Instant Analysis", "تحليل فوري")}</div>
                <div class="vv-hero-badge">🧠 {tr("AI Explanation", "شرح ذكي")}</div>
                <div class="vv-hero-badge">🔒 {tr("Private by Design", "الخصوصية أولاً")}</div>
                <div class="vv-hero-badge">🌐 {tr("Arabic / English", "عربي / إنجليزي")}</div>
            </div>
            <div class="vv-hero-stats">
                <div class="vv-hero-stat">
                    <strong>12+</strong>
                    <span>{tr("Supported nutrients", "عنصر مدعوم")}</span>
                </div>
                <div class="vv-hero-stat">
                    <strong>3+</strong>
                    <span>{tr("Input methods", "طرق إدخال")}</span>
                </div>
                <div class="vv-hero-stat">
                    <strong>AI</strong>
                    <span>{tr("Smart explanations", "شرح ذكي")}</span>
                </div>
            </div>
        </div>

        <div class="vv-hero-panel">
            <div class="vv-mini-progress-title">
                {tr("How your analysis works", 'كيف يتم تحليل نتائجك')}
            </div>

            <div class="vv-snake-flow {'rtl' if is_arabic else 'ltr'}" aria-label="{tr('Analysis workflow', 'مسار تحليل النتائج')}">
                <div class="vv-curve-link c12" aria-hidden="true"></div>
                <div class="vv-curve-link c23" aria-hidden="true"></div>
                <div class="vv-curve-link c34" aria-hidden="true"></div>

                <div class="vv-snake-step s1">
                    <div class="vv-snake-head"><span class="vv-snake-num">1</span><span class="vv-step-title">{tr("Input", 'الإدخال')}</span></div>
                    <div class="vv-step-text">{tr("Enter values manually or upload CSV.", 'أدخل القيم يدويًا أو ارفع ملف CSV.')}</div>
                </div>

                <div class="vv-snake-step s2">
                    <div class="vv-snake-head"><span class="vv-snake-num">2</span><span class="vv-step-title">{tr("Classification", 'التصنيف')}</span></div>
                    <div class="vv-step-text">{tr("Reference ranges and machine learning techniques classify each result.", 'النطاقات المرجعية وتقنيات التعلم الآلي تصنف كل نتيجة.')}</div>
                </div>

                <div class="vv-snake-step s3">
                    <div class="vv-snake-head"><span class="vv-snake-num">3</span><span class="vv-step-title">{tr("AI Companion", 'الرفيق الذكي')}</span></div>
                    <div class="vv-step-text">{tr("Generate explanations only when you need them.", 'ولّد الشرح عند الحاجة فقط.')}</div>
                </div>

                <div class="vv-snake-step s4">
                    <div class="vv-snake-head"><span class="vv-snake-num">4</span><span class="vv-step-title">{tr("Report", 'التقرير')}</span></div>
                    <div class="vv-step-text">{tr("Review charts and downloadable results.", 'راجع الرسوم والنتائج القابلة للتحميل.')}</div>
                </div>
            </div>
        </div>
    </div>
</div>
""")

def render_workflow_strip():
    dir_val = "rtl" if is_arabic else "ltr"
    align = "right" if is_arabic else "left"
    st.html(f"""
<div class="vv-workflow" style="direction:{dir_val}; text-align:{align};">
    <div class="vv-workflow-card">
        <div class="vv-workflow-icon">📥</div>
        <div class="vv-workflow-title">{tr("Upload or Enter", "ارفع أو أدخل")}</div>
        <div class="vv-workflow-text">{tr("Start with manual values or a CSV file.", "ابدأ بالقيم اليدوية أو ملف CSV.")}</div>
    </div>
    <div class="vv-workflow-card">
        <div class="vv-workflow-icon">🔬</div>
        <div class="vv-workflow-title">{tr("Rule-Based Result", "نتيجة الرول بيسد")}</div>
        <div class="vv-workflow-text">{tr("Compare values with clinical reference ranges.", "مقارنة القيم بالنطاقات المرجعية.")}</div>
    </div>
    <div class="vv-workflow-card">
        <div class="vv-workflow-icon">🤖</div>
        <div class="vv-workflow-title">{tr("Model Support", "دعم المودل")}</div>
        <div class="vv-workflow-text">{tr("Show ML prediction beside the rule result.", "عرض توقع المودل بجانب النتيجة.")}</div>
    </div>
    <div class="vv-workflow-card">
        <div class="vv-workflow-icon">💬</div>
        <div class="vv-workflow-title">{tr("Friendly AI Summary", "ملخص ذكي ودّي")}</div>
        <div class="vv-workflow-text">{tr("Explain the patient picture in simple language.", "شرح الحالة بلغة بسيطة للمريض.")}</div>
    </div>
</div>
""")

# =========================================
# Reference ranges (unchanged)
# =========================================
REFERENCE_RANGES = {
    "Zinc":      {"low": 66,   "high": 106,  "unit": "µg/dL",  "max_reasonable": 300},
    "Vitamin_E": {"low": 500,  "high": 2000, "unit": "µg/dL",  "max_reasonable": 5000},
    "Vitamin_A": {"low": 28,   "high": 86,   "unit": "µg/dL",  "max_reasonable": 250},
    "Vitamin_D": {"low": 20,   "high": 50,   "unit": "ng/mL",  "max_reasonable": 200},
    "Vitamin_C": {"low": 0.4,  "high": 2.0,  "unit": "mg/dL",  "max_reasonable": 5},
    "Magnesium": {"low": 1.7,  "high": 2.2,  "unit": "mg/dL",  "max_reasonable": 5},
    "Folate":    {"low": 3,    "high": 20,   "unit": "ng/mL",  "max_reasonable": 60},
    "Vitamin_K": {"low": 0.10, "high": 2.20, "unit": "ng/mL",  "max_reasonable": 8},
    "B12":       {"low": 200,  "high": 900,  "unit": "pg/mL",  "max_reasonable": 3000},
    "B6":        {"low": 20,   "high": 100,  "unit": "nmol/L", "max_reasonable": 400},
    "Calcium":   {"low": 8.6,  "high": 10.2, "unit": "mg/dL",  "max_reasonable": 16},
}

# =========================================
# Nutrient display names
# =========================================
def nutrient_display_name(nutrient):
    names = {
        "Zinc":      tr("Zinc",       "الزنك"),
        "Vitamin_E": tr("Vitamin E",  "فيتامين E"),
        "Vitamin_A": tr("Vitamin A",  "فيتامين A"),
        "Vitamin_D": tr("Vitamin D",  "فيتامين D"),
        "Vitamin_C": tr("Vitamin C",  "فيتامين C"),
        "Magnesium": tr("Magnesium",  "المغنيسيوم"),
        "Folate":    tr("Folate",     "الفولات"),
        "Vitamin_K": tr("Vitamin K",  "فيتامين K"),
        "B12":       tr("Vitamin B12","فيتامين B12"),
        "B6":        tr("Vitamin B6", "فيتامين B6"),
        "Calcium":   tr("Calcium",    "الكالسيوم"),
        "Ferritin":  tr("Ferritin",   "الفيريتين"),
    }
    return names.get(nutrient, nutrient)

# =========================================
# Ferritin range & helpers (unchanged logic)
# =========================================
def get_ferritin_range(gender):
    if gender in [1, "1", "Male", "male", "M", "m", "ذكر"]:
        return {"low": 30,  "high": 400, "unit": "ng/mL", "max_reasonable": 1200}
    return     {"low": 13,  "high": 150, "unit": "ng/mL", "max_reasonable": 800}

def normalize_gender(gender):
    if gender in [1, "1", "Male", "male", "M", "m", "ذكر"]:   return "Male"
    if gender in [2, "2", "Female", "female", "F", "f", "أنثى"]: return "Female"
    return "Male"

def gender_to_num(gender):
    if gender in ["Male", "male", "M", "m", "ذكر", 1, "1"]:
        return 1
    if gender in ["Female", "female", "F", "f", "أنثى", 2, "2"]:
        return 2
    return 1

def get_range(nutrient, gender):
    nutrient = str(nutrient).strip()
    if nutrient == "Ferritin":
        return get_ferritin_range(gender)
    return REFERENCE_RANGES.get(nutrient)

def validate_value(value, range_info):
    if value is None:
        return False, tr("Missing value.", "القيمة مفقودة.")
    if value <= 0:
        return False, tr("Value cannot be negative.", "القيمة لا يمكن أن تكون سالبة.")
    if value > range_info["max_reasonable"]:
        return False, tr(
            "Please check the input value. It looks unusually high.",
            "تأكد من المدخلات، القيمة تبدو مرتفعة بشكل غير منطقي."
        )
    return True, ""

def classify_value(value, low, high):
    if value < low:  return "Deficient"
    if value <= high: return "Normal"
    return "Excessive"

def status_text(status):
    mapping = {
        "Deficient": tr("Deficient",     "ناقص"),
        "Normal":    tr("Normal",         "طبيعي"),
        "Excessive": tr("Excessive",      "مرتفع"),
        "Invalid":   tr("Invalid Input",  "مدخل غير منطقي"),
        "Unknown":   tr("Unknown",        "غير معروف"),
        "Error":     tr("Error",          "خطأ"),
    }
    return mapping.get(status, status)

def status_color(status):
    colors = {
        "Deficient": "#FF4B4B",
        "Normal":    "#1DB954",
        "Excessive": "#FFA500",
        "Invalid":   "#00BFFF",
        "Unknown":   "#888888",
        "Error":     "#888888",
    }
    return colors.get(status, "#888888")

def status_icon(status):
    icons = {
        "Deficient": "",
        "Normal":    "",
        "Excessive": "",
        "Invalid":   "",
        "Unknown":   "—",
        "Error":     "",
    }
    return icons.get(status, "—")

def get_explanation(nutrient, value, unit, status, low, high):
    name = nutrient_display_name(nutrient)
    if status == "Deficient":
        return tr(
            f"Your {name} level is below the normal range ({low}–{high} {unit}).",
            f"مستوى {name} أقل من النطاق الطبيعي ({low}–{high} {unit})."
        )
    if status == "Normal":
        return tr(
            f"Your {name} level is within the normal range ({low}–{high} {unit}).",
            f"مستوى {name} ضمن النطاق الطبيعي ({low}–{high} {unit})."
        )
    return tr(
        f"Your {name} level is above the normal range ({low}–{high} {unit}).",
        f"مستوى {name} أعلى من النطاق الطبيعي ({low}–{high} {unit})."
    )

def get_possible_causes(nutrient, status):
    causes_en = {
        "Deficient": {
            "Vitamin_D": ["Low sunlight exposure", "Low dietary intake"],
            "Vitamin_C": ["Low fruit intake", "Poor nutrition"],
            "Vitamin_A": ["Low intake", "Absorption issues"],
            "Vitamin_E": ["Poor fat absorption"],
            "Vitamin_K": ["Low leafy greens intake"],
            "B12": ["Low animal products", "Absorption problems"],
            "B6": ["Poor diet"],
            "Folate": ["Low vegetables intake"],
            "Ferritin": ["Low iron intake", "Blood loss"],
            "Zinc": ["Poor nutrition"],
            "Magnesium": ["Low intake", "Digestive loss"],
            "Calcium": ["Low intake", "Vitamin D deficiency"],
        },
        "Normal": {"default": ["Balanced nutrition"]},
        "Excessive": {
            "Vitamin_D": ["High supplement intake"],
            "Vitamin_C": ["Excess supplements"],
            "Vitamin_A": ["Over supplementation"],
            "Vitamin_E": ["Supplement overuse"],
            "Vitamin_K": ["High intake"],
            "B12": ["Supplement use"],
            "B6": ["Over supplementation"],
            "Folate": ["Excess supplements"],
            "Ferritin": ["Inflammation", "Iron overload"],
            "Zinc": ["High supplements"],
            "Magnesium": ["Supplement overuse"],
            "Calcium": ["High intake"],
        },
    }
    causes_ar = {
        "Deficient": {
            "Vitamin_D": ["قلة التعرض للشمس", "انخفاض المدخول الغذائي"],
            "Vitamin_C": ["قلة تناول الفواكه", "ضعف التغذية"],
            "Vitamin_A": ["انخفاض المدخول", "مشاكل امتصاص"],
            "Vitamin_E": ["ضعف امتصاص الدهون"],
            "Vitamin_K": ["قلة تناول الخضار الورقية"],
            "B12": ["قلة تناول المنتجات الحيوانية", "مشاكل امتصاص"],
            "B6": ["ضعف النظام الغذائي"],
            "Folate": ["قلة تناول الخضار"],
            "Ferritin": ["قلة تناول الحديد", "فقدان الدم"],
            "Zinc": ["ضعف التغذية"],
            "Magnesium": ["قلة المدخول", "فقدان عبر الجهاز الهضمي"],
            "Calcium": ["قلة المدخول", "نقص فيتامين D"],
        },
        "Normal": {"default": ["تغذية متوازنة"]},
        "Excessive": {
            "Vitamin_D": ["زيادة استخدام المكملات"],
            "Vitamin_C": ["زيادة المكملات"],
            "Vitamin_A": ["زيادة استخدام المكملات"],
            "Vitamin_E": ["الإفراط في المكملات"],
            "Vitamin_K": ["ارتفاع المدخول"],
            "B12": ["استخدام مكملات"],
            "B6": ["الإفراط في المكملات"],
            "Folate": ["زيادة المكملات"],
            "Ferritin": ["التهاب", "زيادة الحديد"],
            "Zinc": ["زيادة المكملات"],
            "Magnesium": ["الإفراط في المكملات"],
            "Calcium": ["ارتفاع المدخول"],
        },
    }
    causes = causes_en if language == "English" else causes_ar
    if status == "Normal":
        return causes["Normal"]["default"]
    return causes.get(status, {}).get(
        nutrient,
        [tr("Diet or absorption related", "مرتبط بالغذاء أو الامتصاص")]
    )

def get_recommendations(status):
    if language == "English":
        recs = {
            "Deficient": ["Improve dietary intake.", "Consult a doctor before supplements."],
            "Normal":    ["Maintain healthy lifestyle."],
            "Excessive": ["Avoid unnecessary supplements.", "Consult a healthcare professional."],
            "Invalid":   ["Check the entered value.", "Make sure the unit matches the nutrient."],
        }
    else:
        recs = {
            "Deficient": ["حسّن المدخول الغذائي.", "استشر الطبيب قبل استخدام المكملات."],
            "Normal":    ["حافظ على نمط حياة صحي."],
            "Excessive": ["تجنب المكملات غير الضرورية.", "استشر مختصًا صحيًا."],
            "Invalid":   ["تأكد من القيمة المدخلة.", "تأكد أن الوحدة مناسبة للعنصر."],
        }
    return recs.get(status, [tr("Consult a healthcare professional.", "استشر مختصًا صحيًا.")])

def get_ml_prediction(age, gender, nutrient, value):
    if ml_model is None:
        return "Model not available"
    try:
        sample = pd.DataFrame({
            "Age": [25 if age is None or pd.isna(age) else age],
            "Gender": [gender_to_num(gender)],
            "Nutrient": [nutrient],
            "Value": [value]
        })
        return ml_model.predict(sample)[0]
    except Exception:
        return "ML Error"

def openai_ai_explanation(temp_row):
    if client is None:
        return tr(
            "AI explanation is not available because the API key is missing.",
            "شرح الذكاء الاصطناعي غير متاح لأن مفتاح API غير موجود."
        )

    cache_key = (
        language,
        str(temp_row.get("Nutrient")),
        str(temp_row.get("Value")),
        str(temp_row.get("Unit")),
        str(temp_row.get("Status")),
        str(temp_row.get("ML_Prediction")),
    )

    if "ai_cache" in st.session_state and cache_key in st.session_state["ai_cache"]:
        return st.session_state["ai_cache"][cache_key]

    prompt = f"""
You are VitaVision AI Explanation Agent.

Explain this vitamin/mineral lab result in simple, warm, patient-friendly language.
Do not diagnose diseases.
Do not prescribe medicine.
Do not claim certainty.
Explain what the result may mean, why it might happen, and one or two practical lifestyle/nutrition notes.
Always say this is educational support and not a medical diagnosis.

Data:
Nutrient: {temp_row.get("Nutrient")}
Value: {temp_row.get("Value")} {temp_row.get("Unit")}
Normal range: {temp_row.get("Low")} - {temp_row.get("High")} {temp_row.get("Unit")}
Rule-based status: {temp_row.get("Status")}
ML prediction: {temp_row.get("ML_Prediction")}
Language: {"Arabic" if language == "العربية" else "English"}

Write a short explanation in 4 lines maximum.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=220
        )
        result = response.output_text
        st.session_state.setdefault("ai_cache", {})[cache_key] = result
        return result
    except Exception as e:
        return tr(
            f"AI explanation unavailable: {e}",
            f"شرح الذكاء الاصطناعي غير متاح: {e}"
        )

def build_patient_signature(df):
    if df is None or df.empty:
        return None
    cols = [c for c in ["Age", "Gender", "Nutrient", "Value", "Unit", "Low", "High", "Status", "ML_Prediction"] if c in df.columns]
    return df[cols].astype(str).to_csv(index=False)

def openai_full_patient_analysis(results_df):
    if client is None:
        return tr(
            "Full AI analysis is not available because the API key is missing.",
            "التحليل الكامل بالذكاء الاصطناعي غير متاح لأن مفتاح API غير موجود."
        )

    if results_df is None or results_df.empty:
        return tr(
            "No patient results available for full AI analysis.",
            "لا توجد نتائج للمريض حتى يتم تحليلها بالكامل بالذكاء الاصطناعي."
        )

    valid_df = results_df.copy()
    summary_counts = valid_df["Status"].value_counts().to_dict() if "Status" in valid_df.columns else {}
    patient_age = valid_df["Age"].dropna().iloc[0] if "Age" in valid_df.columns and not valid_df["Age"].dropna().empty else "Not provided"
    patient_gender = valid_df["Gender"].dropna().iloc[0] if "Gender" in valid_df.columns and not valid_df["Gender"].dropna().empty else "Not provided"

    rows_text = []
    for _, r in valid_df.iterrows():
        rows_text.append(
            f"- {r.get('Nutrient')}: {r.get('Value')} {r.get('Unit')} | "
            f"normal range {r.get('Low')} - {r.get('High')} {r.get('Unit')} | "
            f"status {r.get('Status')} | ML prediction {r.get('ML_Prediction')} | "
            f"rule explanation: {r.get('Explanation')} | "
            f"possible causes: {r.get('Possible Causes')} | "
            f"recommendations: {r.get('Recommendations')}"
        )

    prompt = f"""
You are VitaVision AI Health Companion.

Talk to the patient in a warm, friendly, supportive, and easy-to-understand way.
Do not just list lab results. Help the patient understand:
- what their overall situation looks like,
- why some results might be abnormal,
- what these results may mean in daily life,
- what practical lifestyle or nutrition steps they can start with,
- and which results deserve more attention.

Safety rules:
- Do not diagnose diseases.
- Do not prescribe medicine.
- Do not claim certainty.
- Do not scare the patient or exaggerate risk.
- Do not say the results are definitely caused by one reason.
- Always explain that this is educational support, not a medical diagnosis.
- Recommend consulting a healthcare professional, especially for abnormal or concerning results.
- If a result is excessive/high, mention caution with supplements and checking with a professional.

Patient information:
Age: {patient_age}
Gender: {patient_gender}
Result counts: {summary_counts}

All analyzed results:
{chr(10).join(rows_text)}

Language: {"Arabic" if language == "العربية" else "English"}

Use this exact structure:

1. Your Overall Picture
Start by making the patient feel understood. Mention what looks good and what needs attention.

2. What This May Mean For You
Explain how abnormal results may affect daily life in simple terms, such as energy, tiredness, focus, mood, immunity, bones, or general wellness when relevant. Use cautious language like "may" or "could".

3. Why This Might Be Happening
Give possible lifestyle or nutrition reasons based on the results, such as low sun exposure, low intake, absorption-related issues, supplement overuse, or diet pattern. Do not make a diagnosis.

4. What You Can Start Improving
Give practical, realistic, non-medication advice. Include food/lifestyle examples when useful. Keep it actionable and simple.

5. Results That Need Attention
Clearly point out the most important abnormal results and explain why they should not be ignored. Use calm wording.

6. Friendly Final Note
End with reassurance and a clear medical disclaimer.

Keep it concise but useful. Avoid robotic bullet-only output. Use short paragraphs and friendly wording.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=750
        )
        return response.output_text
    except Exception as e:
        return tr(
            f"Full AI analysis unavailable: {e}",
            f"التحليل الكامل بالذكاء الاصطناعي غير متاح: {e}"
        )

def render_full_patient_ai_analysis(results_df):
    dir_val = "rtl" if is_arabic else "ltr"
    align = "right" if is_arabic else "left"
    signature = build_patient_signature(results_df)

    st.html(f"""
<div class="vv-card" style="direction:{dir_val}; text-align:{align}; border-color:rgba(0,191,255,0.35);">
    <div class="vv-card-title">{tr("AI Health Companion Summary", "ملخص الرفيق الصحي بالذكاء الاصطناعي")}</div>
    <div class="vv-card-text">
        {tr(
            "After reviewing the results, generate one friendly AI summary that explains the whole patient picture and practical next steps.",
            "بعد مراجعة النتائج، ولّد ملخصًا ودّيًا بالذكاء الاصطناعي يشرح الصورة الكاملة للمريض والخطوات العملية للتحسين."
        )}
    </div>
</div>
""")

    if st.button(tr("Generate AI Health Companion Summary", "توليد ملخص الرفيق الصحي"), use_container_width=True, type="primary"):
        with st.spinner(tr("Generating full AI patient analysis...", "جاري توليد التحليل الكامل للحالة...")):
            st.session_state["patient_ai_analysis"] = openai_full_patient_analysis(results_df)
            st.session_state["patient_ai_signature"] = signature

    if st.session_state.get("patient_ai_analysis") and st.session_state.get("patient_ai_signature") == signature:
        formatted = str(st.session_state["patient_ai_analysis"]).replace("\n", "<br>")
        st.html(f"""
<div style="margin:16px 0 24px; padding:20px 22px; border-radius:16px;
            background:linear-gradient(145deg, rgba(0,191,255,0.08), rgba(255,255,255,0.025));
            border:1px solid rgba(0,191,255,0.28); color:#D8E8F0;
            font-size:14.5px; line-height:1.9; text-align:{align}; direction:{dir_val};">
    <div style="font-size:18px; font-weight:800; color:#00BFFF; margin-bottom:12px;">
        {tr("Complete AI Health Companion Analysis", "التحليل الكامل من الرفيق الصحي")}
    </div>
    {formatted}
</div>
""")

def clear_generated_ai_explanations():
    for key in list(st.session_state.keys()):
        if str(key).startswith("ai_text_"):
            del st.session_state[key]
    st.session_state["patient_ai_analysis"] = None
    st.session_state["patient_ai_signature"] = None


def normalize_extracted_nutrient_name(name):
    raw = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "zinc": "Zinc",
        "zn": "Zinc",
        "vitamin_e": "Vitamin_E",
        "vit_e": "Vitamin_E",
        "alpha_tocopherol": "Vitamin_E",
        "vitamin_a": "Vitamin_A",
        "vit_a": "Vitamin_A",
        "retinol": "Vitamin_A",
        "vitamin_d": "Vitamin_D",
        "vit_d": "Vitamin_D",
        "25_oh_vitamin_d": "Vitamin_D",
        "25oh_vitamin_d": "Vitamin_D",
        "25_hydroxyvitamin_d": "Vitamin_D",
        "vitamin_c": "Vitamin_C",
        "vit_c": "Vitamin_C",
        "ascorbic_acid": "Vitamin_C",
        "magnesium": "Magnesium",
        "mg": "Magnesium",
        "folate": "Folate",
        "folic_acid": "Folate",
        "ferritin": "Ferritin",
        "vitamin_k": "Vitamin_K",
        "vit_k": "Vitamin_K",
        "vitamin_b12": "B12",
        "b12": "B12",
        "vit_b12": "B12",
        "vitamin_b6": "B6",
        "b6": "B6",
        "plp": "B6",
        "calcium": "Calcium",
        "ca": "Calcium",
    }
    if raw in mapping:
        return mapping[raw]
    for n in list(REFERENCE_RANGES.keys()) + ["Ferritin"]:
        if raw == n.lower():
            return n
    return str(name).strip()

def parse_ai_extraction_json(text_value):
    cleaned = str(text_value).strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("AI did not return valid JSON.")
        data = json.loads(match.group(0))

    rows = data.get("results", [])
    if not isinstance(rows, list):
        rows = []

    extracted = []
    default_age = data.get("age", None)
    default_gender = data.get("gender", 1)

    for item in rows:
        nutrient = normalize_extracted_nutrient_name(item.get("Nutrient") or item.get("nutrient") or "")
        value = item.get("Value", item.get("value", None))
        unit = item.get("Unit", item.get("unit", ""))
        age = item.get("Age", item.get("age", default_age))
        gender = item.get("Gender", item.get("gender", default_gender))

        if nutrient not in list(REFERENCE_RANGES.keys()) + ["Ferritin"]:
            continue

        try:
            value = float(str(value).replace(",", "").strip())
        except Exception:
            continue

        extracted.append({
            "Age": age if age not in ["", None] else None,
            "Gender": gender if gender not in ["", None] else 1,
            "Nutrient": nutrient,
            "Value": value,
            "Detected_Unit": unit,
        })

    return pd.DataFrame(extracted)

def openai_extract_lab_values_from_file(file_name, file_bytes, mime_type):
    if client is None:
        return pd.DataFrame(), tr(
            "Smart AI Upload is not available because the OpenAI API key is missing.",
            "الرفع الذكي غير متاح لأن مفتاح OpenAI غير موجود."
        )

    supported = ", ".join(list(REFERENCE_RANGES.keys()) + ["Ferritin"])
    file_b64 = base64.b64encode(file_bytes).decode("utf-8")

    prompt = f"""
You are VitaVision Lab Value Extraction Agent.

Your task:
Extract only these supported nutrients from the uploaded lab file:
{supported}

Return JSON only. No markdown.

Required JSON format:
{{
  "age": null,
  "gender": 1,
  "results": [
    {{"Nutrient": "Vitamin_D", "Value": 18.5, "Unit": "ng/mL", "Age": null, "Gender": 1}}
  ]
}}

Rules:
- Extract numeric lab values only.
- Do not invent missing values.
- If age or gender is not present, use null for age and 1 for gender.
- Gender values: 1 for Male, 2 for Female.
- Use VitaVision nutrient names exactly:
Zinc, Vitamin_E, Vitamin_A, Vitamin_D, Vitamin_C, Magnesium, Folate, Ferritin, Vitamin_K, B12, B6, Calcium
- If the document uses names like Vitamin B12, 25-OH Vitamin D, Retinol, Alpha-tocopherol, PLP, map them correctly.
- Keep the detected unit in Unit.
- If no supported nutrients are found, return {{"age": null, "gender": 1, "results": []}}.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_file",
                            "filename": file_name,
                            "file_data": f"data:{mime_type};base64,{file_b64}",
                        },
                    ],
                }
            ],
            max_output_tokens=1200
        )
        df = parse_ai_extraction_json(response.output_text)
        return df, ""
    except Exception as e:
        return pd.DataFrame(), tr(
            f"Could not extract lab values from the file: {e}",
            f"تعذر استخراج القيم من الملف: {e}"
        )


# =========================================
# Analyze row (rule + ML, AI on demand)
# =========================================
def analyze_row(row):
    nutrient = str(row.get("Nutrient", "")).strip()
    value    = float(row.get("Value", 0))
    gender   = normalize_gender(row.get("Gender", "Male"))
    age      = row.get("Age", None)
    range_info = get_range(nutrient, gender)

    if range_info is None:
        return {
            "Age": age, "Gender": gender, "Nutrient": nutrient, "Value": value,
            "Unit": "Unknown", "Low": None, "High": None, "Status": "Unknown",
            "ML_Prediction": "Not applied",
            "Explanation": tr("Unknown nutrient.", "عنصر غير معروف"),
            "Possible Causes": "", "Recommendations": "",
        }

    valid, msg = validate_value(value, range_info)

    if not valid:
        return {
            "Age": age, "Gender": gender, "Nutrient": nutrient, "Value": value,
            "Unit": range_info["unit"], "Low": range_info["low"], "High": range_info["high"],
            "Status": "Invalid", "ML_Prediction": "Not applied", "Explanation": msg,
            "Possible Causes": tr(
                "Possible wrong unit or typing error.",
                "قد تكون الوحدة غير صحيحة أو يوجد خطأ في الإدخال."
            ),
            "Recommendations": "; ".join(get_recommendations("Invalid")),
        }

    low, high, unit = range_info["low"], range_info["high"], range_info["unit"]
    status = classify_value(value, low, high)
    ml_pred = get_ml_prediction(age, gender, nutrient, value)

    return {
        "Age": age, "Gender": gender, "Nutrient": nutrient, "Value": value,
        "Unit": unit, "Low": low, "High": high, "Status": status, "ML_Prediction": ml_pred,
        "Explanation": get_explanation(nutrient, value, unit, status, low, high),
        "Possible Causes": "; ".join(get_possible_causes(nutrient, status)),
        "Recommendations": "; ".join(get_recommendations(status)),
    }

# =========================================
# Reference range chart
# =========================================
def create_reference_chart(row):
    if row.get("Low") is None or row.get("High") is None:
        return None
    value, low, high = row["Value"], row["Low"], row["High"]
    nutrient, unit  = row["Nutrient"], row["Unit"]
    if pd.isna(low) or pd.isna(high):
        return None

    max_axis = max(value, high) * 1.35
    fig = go.Figure()

    fig.add_trace(go.Bar(x=[low],          y=[nutrient], orientation="h",
                         marker_color="rgba(255,75,75,0.75)",  name=tr("Below normal", "أقل من الطبيعي"), base=0))
    fig.add_trace(go.Bar(x=[high - low],   y=[nutrient], orientation="h",
                         marker_color="rgba(29,185,84,0.75)",  name=tr("Normal range", "النطاق الطبيعي"), base=low))
    fig.add_trace(go.Bar(x=[max_axis - high], y=[nutrient], orientation="h",
                         marker_color="rgba(255,165,0,0.60)", name=tr("Above normal", "أعلى من الطبيعي"), base=high))
    fig.add_trace(go.Scatter(
        x=[value], y=[nutrient], mode="markers+text",
        text=[f"  {value} {unit}"], textposition="middle right",
        textfont=dict(color="white", size=12, family="Plus Jakarta Sans, Cairo"),
        marker=dict(size=16, color="white", line=dict(color="#00BFFF", width=2.5),
                    symbol="diamond"),
        name=tr("Your value", "قيمتك"),
    ))

    fig.update_layout(
    title=dict(
        text=tr(
            f"{nutrient_display_name(nutrient)}: Value vs Normal Range",
            f"{nutrient_display_name(nutrient)}: القيمة مقارنة بالنطاق الطبيعي"
        ),
        font=dict(size=14, color="#B8C8D8", family="Plus Jakarta Sans, Cairo"),
    ),
    height=230 if is_arabic else 220,
    barmode="overlay",
    xaxis=dict(
        range=[0, max_axis],
        title=unit,
        gridcolor="rgba(255,255,255,0.05)",
        tickfont=dict(color="#7A9BB5", size=11)
    ),
    yaxis=dict(
        title="",
        tickfont=dict(color="#7A9BB5")
    ),
    margin=dict(l=10, r=10, t=40, b=20),
    legend=dict(
        orientation="h",
        y=-0.5,
        font=dict(color="#9AAAB8", size=11),
        bgcolor="rgba(0,0,0,0)"
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, Cairo"),

    
        dragmode=False,
        hovermode=False
    )

    
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig 

# =========================================
# Result card (enhanced)
# =========================================
def render_result_card(row, card_key="0"):
    status   = row["Status"]
    color    = status_color(status)
    icon     = status_icon(status)
    nutrient = nutrient_display_name(row["Nutrient"])
    value    = row["Value"]
    unit     = row.get("Unit", "")
    dir_val  = "rtl" if is_arabic else "ltr"
    align    = "right" if is_arabic else "left"

    causes_html = "".join(
        f'<li>{c.strip()}</li>'
        for c in str(row.get("Possible Causes", "")).split(";") if c.strip()
    )
    recs_html = "".join(
        f'<li>{r.strip()}</li>'
        for r in str(row.get("Recommendations", "")).split(";") if r.strip()
    )

    explanation = row.get("Explanation", "")
    ml_pred = row.get("ML_Prediction", "N/A")

    padding_side = "padding-right: 20px" if is_arabic else "padding-left: 20px"
    arrow_side   = "right: 0" if is_arabic else "left: 0"

    st.html(f"""
<div class="result-card" style="border: 1px solid {color}30; border-left: 4px solid {color}; direction:{dir_val};">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
        <div>
            <div style="font-size:19px; font-weight:800; color:#F0F4F8;
                        font-family:'Plus Jakarta Sans','Cairo',sans-serif; margin-bottom:3px;">
                {nutrient}
            </div>
            <div style="font-size:12px; color:#7A9BB5; letter-spacing:0.4px; text-transform:uppercase;">
                {tr("Lab Result", "نتيجة التحليل")}
            </div>
        </div>
        <span class="badge" style="color:{color}; background:{'rgba'+color[3:]+'22)' if color.startswith('#') else color+'22'};
                                   border: 1px solid {color}55; font-size:13px;">
            {icon} {status_text(status)}
        </span>
    </div>

    <div style="font-size:34px; font-weight:800; color:{color};
                font-family:'Plus Jakarta Sans','Cairo',sans-serif; margin-bottom:10px; line-height:1;">
        {value}
        <span style="font-size:16px; color:#7A9BB5; font-weight:500; margin-left:6px;">{unit}</span>
    </div>

    <div style="font-size:13px; color:#7A9BB5; margin-bottom:14px;">
        <b>ML Prediction:</b> {ml_pred}
    </div>

    <div style="font-size:14px; color:#B8C8D8; line-height:1.7; margin-bottom:14px;
                background:rgba(255,255,255,0.03); border-radius:8px; padding:12px 14px;
                border-left:2px solid {color}55; text-align:{align}; direction:{dir_val};">
        {explanation}
    </div>

    <div style="
display:flex;
justify-content:center;
align-items:center;
margin-top:20px;
">
        <div style="background:rgba(255,255,255,0.025); border-radius:10px; padding:12px 14px;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:11px; font-weight:700; color:#7A9BB5; letter-spacing:0.5px;
                        text-transform:uppercase; margin-bottom:8px;">
                🔍 {tr("Possible Causes", "الأسباب المحتملة")}
            </div>
            <ul style="margin:0; padding:0; list-style:none; direction:{dir_val}; text-align:{align};">
                {causes_html.replace('<li>', f'<li style="font-size:13px; color:#B8C8D8; margin-bottom:5px; {padding_side}; position:relative;"><span style="position:absolute; {arrow_side}; color:{color};">•</span>')}
            </ul>
        </div>
        <div style="background:rgba(255,255,255,0.025); border-radius:10px; padding:12px 14px;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:11px; font-weight:700; color:#7A9BB5; letter-spacing:0.5px;
                        text-transform:uppercase; margin-bottom:8px;">
                💡 {tr("Recommendations", "التوصيات")}
            </div>
            <ul style="margin:0; padding:0; list-style:none; direction:{dir_val}; text-align:{align};">
                {recs_html.replace('<li>', f'<li style="font-size:13px; color:#B8C8D8; margin-bottom:5px; {padding_side}; position:relative;"><span style="position:absolute; {arrow_side}; color:{color};">•</span>')}
            </ul>
        </div>
    </div>

    <div style="margin-top:14px; padding:10px 14px; border-radius:8px;
                background:rgba(255,193,7,0.06); border:1px solid rgba(255,193,7,0.20);
                display:flex; align-items:flex-start; gap:8px; direction:{dir_val};">
        <span style="font-size:14px; flex-shrink:0;"></span>
        <span style="font-size:13px; color:#C8B87A; line-height:1.5; text-align:{align};">
            {tr(
                "Medical note: This result is not a medical diagnosis. Please consult a healthcare professional.",
                "ملاحظة طبية: هذه النتيجة ليست تشخيصًا طبيًا. يرجى استشارة مختص صحي."
            )}
        </span>
    </div>
</div>
<div style="height:8px;"></div>
""")

    ai_button_key = f"ai_explain_{card_key}"
    ai_text_key = f"ai_text_{card_key}"

    if st.button(tr("Generate AI Explanation", "توليد شرح الذكاء الاصطناعي"), key=ai_button_key, use_container_width=True):
        temp_row = {
            "Nutrient": row.get("Nutrient"),
            "Value": row.get("Value"),
            "Unit": row.get("Unit"),
            "Low": row.get("Low"),
            "High": row.get("High"),
            "Status": row.get("Status"),
            "ML_Prediction": row.get("ML_Prediction"),
        }
        with st.spinner(tr("Generating AI explanation...", "جاري توليد شرح الذكاء الاصطناعي...")):
            st.session_state[ai_text_key] = openai_ai_explanation(temp_row)

    if ai_text_key in st.session_state:
        ai_text = str(st.session_state[ai_text_key]).replace("\n", "<br>")
        st.html(f"""
<div style="margin:-4px 0 18px; padding:14px 16px; border-radius:10px;
            background:rgba(0,191,255,0.06); border:1px solid rgba(0,191,255,0.22);
            color:#B8C8D8; font-size:13.5px; line-height:1.7;
            direction:{dir_val}; text-align:{align};">
    <b style="color:#00BFFF;">{tr("AI Explanation", "شرح الذكاء الاصطناعي")}:</b><br>
    {ai_text}
</div>
""")

# =========================================
# Styled results table
# =========================================
def render_results_table(df):
    dir_val = "rtl" if is_arabic else "ltr"
    align   = "right" if is_arabic else "left"

    headers = [
        tr("Nutrient","العنصر"),
        tr("Value","القيمة"),
        tr("Unit","الوحدة"),
        tr("Rule-Based","Rule-Based"),
        tr("ML Model","ML Model"),
        tr("Status","الحالة"),
        tr("Low","الأدنى"),
        tr("High","الأقصى"),
        tr("Age","العمر"),
        tr("Gender","الجنس"),
    ]

    header_cells = "".join(
        f'<th style="padding:11px 16px; font-size:12px; font-weight:700; color:#7A9BB5; '
        f'letter-spacing:0.5px; text-transform:uppercase; background:rgba(0,191,255,0.05); '
        f'white-space:nowrap; text-align:{align};">{h}</th>'
        for h in headers
    )

    rows_html = ""
    for _, row in df.iterrows():
        status  = row.get("Status", "")
        color   = status_color(status)
        icon    = status_icon(status)
        gender_display = tr("Male","ذكر") if str(row.get("Gender","")).lower() in ["male","1","m"] else tr("Female","أنثى")

        cells = [
            f'<td style="font-weight:700; color:#F0F4F8;">{nutrient_display_name(row.get("Nutrient",""))}</td>',
            f'<td style="font-weight:700; color:{color}; font-size:16px;">{row.get("Value","")}</td>',
            f'<td style="color:#7A9BB5;">{row.get("Unit","")}</td>',
            f'<td style="color:#00BFFF; font-weight:700;">{row.get("Status","")}</td>',
            f'<td style="color:#FFA500; font-weight:700;">{row.get("ML_Prediction","N/A")}</td>',
            f'<td><span style="display:inline-flex; align-items:center; gap:5px; padding:4px 12px; '
            f'border-radius:999px; font-size:12px; font-weight:700; color:{color}; '
            f'background:{color}18; border:1px solid {color}44;">'
            f'{icon} {status_text(status)}</span></td>',
            f'<td style="color:#9AAAB8;">{row.get("Low","")}</td>',
            f'<td style="color:#9AAAB8;">{row.get("High","")}</td>',
            f'<td style="color:#9AAAB8;">{row.get("Age","")}</td>',
            f'<td style="color:#9AAAB8;">{gender_display}</td>',
        ]

        row_cells = "".join(
            f'<td style="padding:12px 16px; font-size:14px; text-align:{align}; '
            f'border-bottom:1px solid rgba(255,255,255,0.04);">{cell[cell.find(">")+1:]}'
            for cell in cells
        )
        rows_html += f'<tr style="transition:background 0.15s;" onmouseover="this.style.background=\'rgba(0,191,255,0.04)\'" onmouseout="this.style.background=\'transparent\'">{row_cells}</tr>'

    st.html(f"""
<div style="overflow-x:auto; border-radius:16px; border:1px solid rgba(0,191,255,0.18);
            box-shadow:0 4px 24px rgba(0,0,0,0.20); direction:{dir_val};">
    <table style="width:100%; border-collapse:collapse; font-family:'Plus Jakarta Sans','Cairo',sans-serif;">
        <thead>
            <tr>{header_cells}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>
""")

# =========================================
# Summary stats bar
# =========================================
def render_summary_stats(df):
    total      = len(df)
    normal     = len(df[df["Status"] == "Normal"])
    deficient  = len(df[df["Status"] == "Deficient"])
    excessive  = len(df[df["Status"] == "Excessive"])

    st.html(f"""
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(115px, 1fr)); gap:12px; margin:18px 0; width:100%;">
    <div class="stat-mini">
        <div class="stat-number" style="color:#F0F4F8;">{total}</div>
        <div class="stat-label">{tr("Total", "الإجمالي")}</div>
    </div>
    <div class="stat-mini">
        <div class="stat-number" style="color:#1DB954;">{normal}</div>
        <div class="stat-label">{tr("Normal", "طبيعي")}</div>
    </div>
    <div class="stat-mini">
        <div class="stat-number" style="color:#FF4B4B;">{deficient}</div>
        <div class="stat-label">{tr("Deficient", "ناقص")}</div>
    </div>
    <div class="stat-mini">
        <div class="stat-number" style="color:#FFA500;">{excessive}</div>
        <div class="stat-label">{tr("Excessive", "مرتفع")}</div>
    </div>
</div>
""")

# =========================================
# Session state
# =========================================
if "manual_items" not in st.session_state:
    st.session_state["manual_items"] = []

if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame()

if "csv_input_df" not in st.session_state:
    st.session_state["csv_input_df"] = None

if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}

if "patient_ai_analysis" not in st.session_state:
    st.session_state["patient_ai_analysis"] = None

if "patient_ai_signature" not in st.session_state:
    st.session_state["patient_ai_signature"] = None

if "app_page" not in st.session_state:
    st.session_state["app_page"] = "choose_input"

if "selected_input_method" not in st.session_state:
    st.session_state["selected_input_method"] = None

if "smart_uploaded_file_name" not in st.session_state:
    st.session_state["smart_uploaded_file_name"] = None

if "smart_file_bytes" not in st.session_state:
    st.session_state["smart_file_bytes"] = None

if "smart_file_mime" not in st.session_state:
    st.session_state["smart_file_mime"] = None

if "smart_extracted_df" not in st.session_state:
    st.session_state["smart_extracted_df"] = pd.DataFrame()

if st.session_state.get("language_changed", False):
    if st.session_state["csv_input_df"] is not None:
        analyzed = [analyze_row(row) for _, row in st.session_state["csv_input_df"].iterrows()]
        st.session_state["results_df"] = pd.DataFrame(analyzed)
    elif len(st.session_state["manual_items"]) > 0:
        input_df = pd.DataFrame(st.session_state["manual_items"])
        analyzed = [analyze_row(row) for _, row in input_df.iterrows()]
        st.session_state["results_df"] = pd.DataFrame(analyzed)
    clear_generated_ai_explanations()
    st.session_state["language_changed"] = False

# =========================================
# HOME TAB
# =========================================
with home_tab:
    current_page = st.session_state.get("app_page", "choose_input")
    dir_val = "rtl" if is_arabic else "ltr"
    align = "right" if is_arabic else "left"

    def render_progress(active):
        steps = [
            ("choose_input", tr("1. Choose Method", "١. اختر الطريقة")),
            ("input_page", tr("2. Enter Data", "٢. أدخل البيانات")),
            ("results_page", tr("3. Results", "٣. النتائج")),
        ]
        html = "".join(
            f'<div class="vv-progress-step {"active" if key == active else ""}">{label}</div>'
            for key, label in steps
        )
        st.html(f'<div class="vv-progress-flow" style="direction:{dir_val};">{html}</div>')

    def go_to_choose():
        st.session_state["app_page"] = "choose_input"
        st.session_state["selected_input_method"] = None
        st.rerun()

    def go_to_input(method):
        st.session_state["selected_input_method"] = method
        st.session_state["app_page"] = "input_page"
        st.rerun()

    def go_to_results():
        st.session_state["app_page"] = "results_page"
        st.rerun()

    # ── Auth pages ───────────────────────────
    if current_page == "login_page":
        render_login_page()

    elif current_page == "register_page":
        render_register_page()

    elif current_page == "admin_page":
        render_admin_page()

    # ── Page 1: Choose input method ───────────
    elif current_page == "choose_input":
        render_header()

        st.html(f"""
<div class="vv-start-card" style="direction:{dir_val}; text-align:{align};">
    <div class="vv-start-card-title">{tr("How would you like to start?", "كيف تبغى تبدأ؟")}</div>
    <div class="vv-start-card-text">
        {tr("Choose the best way to enter the patient's lab values. You can enter values manually or upload a CSV file.",
            "اختر الطريقة المناسبة لإدخال نتائج المريض. تقدر تدخل القيم يدويًا أو ترفع ملف CSV.")}
    </div>
</div>

<div class="vv-choice-grid" style="direction:{dir_val}; text-align:{align};">
    <div class="vv-choice-card">
        <div class="vv-choice-icon">⌨️</div>
        <div class="vv-choice-title">{tr("Manual Input", "إدخال يدوي")}</div>
        <div class="vv-choice-text">
            {tr("Best for testing one patient by adding nutrient values one by one.",
                "مناسب لتحليل مريض واحد بإضافة قيم العناصر وحدة وحدة.")}
        </div>
    </div>
    <div class="vv-choice-card">
        <div class="vv-choice-icon">📄</div>
        <div class="vv-choice-title">{tr("Upload CSV", "رفع CSV")}</div>
        <div class="vv-choice-text">
            {tr("Best for analyzing multiple lab values from a prepared CSV file.",
                "مناسب لتحليل عدة قيم من ملف CSV جاهز.")}
        </div>
    </div>
    <div class="vv-choice-card">
        <div class="vv-choice-icon">🤖</div>
        <div class="vv-choice-title">{tr("Smart AI Upload", "رفع ذكي بالذكاء الاصطناعي")}</div>
        <div class="vv-choice-text">
            {tr("Upload PDF, image, Word, Excel, or text and let AI extract supported lab values.",
                "ارفع PDF أو صورة أو Word أو Excel أو نص، والذكاء يستخرج القيم المدعومة.")}
        </div>
    </div>
</div>
""")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(f"⌨️  {tr('Start Manual Input', 'ابدأ الإدخال اليدوي')}", use_container_width=True, type="primary"):
                go_to_input("manual")
        with c2:
            if st.button(f"📄  {tr('Upload CSV File', 'ارفع ملف CSV')}", use_container_width=True, type="primary"):
                go_to_input("csv")
        with c3:
            if st.button(f"🤖  {tr('Smart AI Upload', 'رفع ذكي')}", use_container_width=True, type="primary"):
                go_to_input("smart")

    # ── Page 2: Input page ────────────────────
    elif current_page == "input_page":
        method = st.session_state.get("selected_input_method", "manual")
        render_header()
        render_progress("input_page")

        if st.button(f"← {tr('Back to Method Selection', 'رجوع لاختيار الطريقة')}", use_container_width=True):
            go_to_choose()


        if method == "manual":
            section_title(tr("Enter Lab Values", "أدخل قيم التحليل"), 22, "")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                age = st.number_input(tr("Age", "العمر"), min_value=0, max_value=120, value=25, step=1)
            with col2:
                gender_text = st.selectbox(tr("Gender", "الجنس"), [tr("Male", "ذكر"), tr("Female", "أنثى")])
            with col3:
                nutrient = st.selectbox(tr("Nutrient", "العنصر"), list(REFERENCE_RANGES.keys()) + ["Ferritin"])
            with col4:
                value = st.number_input(tr("Value", "القيمة"), min_value=0.0, value=0.0, step=0.1)

            add_col, analyze_col, clear_col = st.columns(3)
            with add_col:
                if st.button(f"➕  {tr('Add Nutrient', 'إضافة العنصر')}", use_container_width=True):
                    gender_value = 1 if gender_text in ["Male", "ذكر"] else 2
                    st.session_state["manual_items"].append({
                        "Age": age, "Gender": gender_value,
                        "Nutrient": nutrient, "Value": value,
                    })
            with analyze_col:
                analyze_manual_clicked = st.button(f"🔬  {tr('Analyze & Show Results', 'تحليل وعرض النتائج')}", use_container_width=True, type="primary")
            with clear_col:
                if st.button(f"🗑️  {tr('Clear All', 'مسح الكل')}", use_container_width=True):
                    st.session_state["manual_items"] = []
                    st.session_state["csv_input_df"] = None
                    st.session_state["results_df"] = pd.DataFrame()
                    clear_generated_ai_explanations()
                    st.rerun()

            section_title(tr("Added Nutrients", "العناصر المضافة"), 20, "")

            if len(st.session_state["manual_items"]) == 0:
                st.html(f"""
<div style="text-align:center; padding:28px; border:1px dashed rgba(0,191,255,0.20);
            border-radius:14px; color:#5A7A8A; font-size:14px;">
    {tr("No nutrients added yet. Use the form above to add lab values.",
        "لم تتم إضافة أي عنصر حتى الآن. استخدم النموذج أعلاه لإضافة قيم التحاليل.")}
</div>
""")
            else:
                header_cells = "".join(
                    f'<th style="padding:10px 14px; font-size:12px; font-weight:700; color:#7A9BB5; '
                    f'text-transform:uppercase; letter-spacing:0.4px; text-align:{align}; '
                    f'background:rgba(0,191,255,0.04);">{h}</th>'
                    for h in [tr("Age","العمر"), tr("Gender","الجنس"),
                              tr("Nutrient","العنصر"), tr("Value","القيمة")]
                )
                rows_html = ""
                for item in st.session_state["manual_items"]:
                    shown_gender = tr("Male","ذكر") if item["Gender"] == 1 else tr("Female","أنثى")
                    rows_html += f"""
<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
    <td style="padding:10px 14px; font-size:14px; color:#B8C8D8; text-align:{align};">{item['Age']}</td>
    <td style="padding:10px 14px; font-size:14px; color:#B8C8D8; text-align:{align};">{shown_gender}</td>
    <td style="padding:10px 14px; font-size:14px; font-weight:600; color:#F0F4F8; text-align:{align};">{nutrient_display_name(item['Nutrient'])}</td>
    <td style="padding:10px 14px; font-size:14px; font-weight:700; color:#00BFFF; text-align:{align};">{item['Value']}</td>
</tr>"""

                st.html(f"""
<div style="overflow-x:auto; border-radius:14px; border:1px solid rgba(255,255,255,0.07); direction:{dir_val};">
    <table style="width:100%; border-collapse:collapse; font-family:'Plus Jakarta Sans','Cairo',sans-serif;">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>""")

            if analyze_manual_clicked:
                if len(st.session_state["manual_items"]) == 0:
                    st.warning(tr("Please add at least one nutrient.", "يرجى إضافة عنصر واحد على الأقل."))
                else:
                    with st.spinner(tr("Analyzing...", "جاري التحليل...")):
                        input_df = pd.DataFrame(st.session_state["manual_items"])
                        analyzed = [analyze_row(r) for _, r in input_df.iterrows()]
                        st.session_state["results_df"] = pd.DataFrame(analyzed)
                        st.session_state["csv_input_df"] = None
                        save_analysis_log("manual", st.session_state["results_df"])
                        clear_generated_ai_explanations()
                        st.session_state["app_page"] = "results_page"
                        st.rerun()


        elif method == "csv":
            section_title(tr("Upload CSV File", "رفع ملف CSV"), 22, "")

            st.html(f"""
<div style="background:rgba(0,191,255,0.04); border:1px dashed rgba(0,191,255,0.25);
            border-radius:14px; padding:16px 20px; margin-bottom:16px; font-size:13.5px; color:#7A9BB5;
            direction:{dir_val}; text-align:{align};">
    <strong style="color:#00BFFF;">  {tr("Required columns:", "الأعمدة المطلوبة:")}</strong>
    &nbsp;Nutrient, Value &nbsp;|&nbsp;
    <strong style="color:#9AAAB8;">{tr("Optional:", "اختياري:")}</strong> Age, Gender
</div>
""")

            uploaded_file = st.file_uploader(tr("Upload CSV file", "ارفع ملف CSV"), type=["csv"])

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)

                with st.expander(tr("Show Preview", "عرض المعاينة"), expanded=False):
                    st.dataframe(df, use_container_width=True)

                required_columns = ["Nutrient", "Value"]
                missing_columns  = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(tr(
                        f"Missing required columns: {missing_columns}",
                        f"الأعمدة الناقصة: {missing_columns}"
                    ))
                    st.info(tr(
                        "Required columns: Nutrient, Value. Optional: Age, Gender.",
                        "الأعمدة المطلوبة: Nutrient و Value. الاختيارية: Age و Gender."
                    ))
                else:
                    if "Age"    not in df.columns: df["Age"]    = None
                    if "Gender" not in df.columns: df["Gender"] = 1

                    if st.button(f"🔬  {tr('Analyze CSV & Show Results', 'تحليل الملف وعرض النتائج')}", use_container_width=True, type="primary"):
                        analyzed = []
                        with st.spinner(tr("Analyzing CSV...", "جاري تحليل الملف...")):
                            for _, row in df.iterrows():
                                try:
                                    analyzed.append(analyze_row(row))
                                except Exception as e:
                                    analyzed.append({
                                        "Age": row.get("Age", None),
                                        "Gender": row.get("Gender", None),
                                        "Nutrient": row.get("Nutrient", "Unknown"),
                                        "Value": row.get("Value", None),
                                        "Unit": "Unknown", "Low": None, "High": None,
                                        "Status": "Error",
                                        "ML_Prediction": "Not applied",
                                        "Explanation": tr(f"Could not analyze: {e}", f"تعذر التحليل: {e}"),
                                        "Possible Causes": tr("Not available", "غير متوفر"),
                                        "Recommendations": tr("Check data format.", "تأكد من صيغة البيانات."),
                                    })
                        st.session_state["results_df"]  = pd.DataFrame(analyzed)
                        st.session_state["csv_input_df"] = df.copy()
                        save_analysis_log("csv", st.session_state["results_df"])
                        clear_generated_ai_explanations()
                        st.session_state["app_page"] = "results_page"
                        st.rerun()


        elif method == "smart":
            section_title(tr("Smart AI Upload", "رفع ذكي بالذكاء الاصطناعي"), 22, "")

            st.html(f"""
<div class="vv-smart-box" style="direction:{dir_val}; text-align:{align};">
    <div class="vv-smart-title">{tr("Upload any lab file", "ارفع أي ملف تحليل")}</div>
    <div class="vv-smart-text">
        {tr("AI will read the file, search for VitaVision-supported nutrients, and extract values for your review before analysis.",
            "الذكاء الاصطناعي سيقرأ الملف ويبحث عن العناصر المدعومة في VitaVision، ثم يعرض القيم للمراجعة قبل التحليل.")}
    </div>
</div>
<div class="vv-smart-note" style="direction:{dir_val}; text-align:{align};">
    {tr("Important: AI extraction can make mistakes. Always review the extracted values before confirming the analysis.",
        "مهم: استخراج الذكاء الاصطناعي قد يخطئ. راجع القيم المستخرجة دائمًا قبل تأكيد التحليل.")}
</div>
""")

            smart_file = st.file_uploader(
                tr("Upload PDF, image, Word, Excel, CSV, or text file", "ارفع PDF أو صورة أو Word أو Excel أو CSV أو ملف نصي"),
                type=["pdf", "png", "jpg", "jpeg", "webp", "docx", "xlsx", "xls", "csv", "txt"]
            )

            if smart_file is not None:
                file_bytes = smart_file.getvalue()
                mime_type = smart_file.type or "application/octet-stream"
                st.session_state["smart_uploaded_file_name"] = smart_file.name
                st.session_state["smart_file_bytes"] = file_bytes
                st.session_state["smart_file_mime"] = mime_type

                st.success(tr(f"File uploaded: {smart_file.name}", f"تم رفع الملف: {smart_file.name}"))

                if st.button(f"🤖  {tr('Extract Lab Values with AI', 'استخراج القيم بالذكاء الاصطناعي')}", use_container_width=True, type="primary"):
                    with st.spinner(tr("AI is reading the file and extracting lab values...", "الذكاء الاصطناعي يقرأ الملف ويستخرج القيم...")):
                        extracted_df, err = openai_extract_lab_values_from_file(
                            smart_file.name,
                            file_bytes,
                            mime_type
                        )
                        if err:
                            st.error(err)
                        else:
                            st.session_state["smart_extracted_df"] = extracted_df
                            if extracted_df.empty:
                                st.warning(tr(
                                    "No supported VitaVision nutrients were found in this file.",
                                    "لم يتم العثور على عناصر مدعومة من VitaVision في هذا الملف."
                                ))
                            else:
                                st.success(tr(
                                    "Lab values extracted. Please review them before analysis.",
                                    "تم استخراج القيم. يرجى مراجعتها قبل التحليل."
                                ))

            if not st.session_state["smart_extracted_df"].empty:
                section_title(tr("Review Extracted Values", "مراجعة القيم المستخرجة"), 20, "")

                reviewed_df = st.data_editor(
                    st.session_state["smart_extracted_df"],
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "Nutrient": st.column_config.SelectboxColumn(
                            tr("Nutrient", "العنصر"),
                            options=list(REFERENCE_RANGES.keys()) + ["Ferritin"],
                            required=True,
                        ),
                        "Value": st.column_config.NumberColumn(tr("Value", "القيمة"), required=True),
                        "Age": st.column_config.NumberColumn(tr("Age", "العمر")),
                        "Gender": st.column_config.SelectboxColumn(
                            tr("Gender", "الجنس"),
                            options=[1, 2, "Male", "Female", "ذكر", "أنثى"],
                        ),
                    },
                    key="smart_review_editor"
                )

                c_confirm, c_clear = st.columns(2)
                with c_confirm:
                    if st.button(f"✅  {tr('Confirm & Analyze', 'تأكيد وتحليل')}", use_container_width=True, type="primary"):
                        analyzed = []
                        with st.spinner(tr("Analyzing confirmed values...", "جاري تحليل القيم المؤكدة...")):
                            for _, row in reviewed_df.iterrows():
                                try:
                                    analyzed.append(analyze_row(row))
                                except Exception as e:
                                    analyzed.append({
                                        "Age": row.get("Age", None),
                                        "Gender": row.get("Gender", None),
                                        "Nutrient": row.get("Nutrient", "Unknown"),
                                        "Value": row.get("Value", None),
                                        "Unit": "Unknown", "Low": None, "High": None,
                                        "Status": "Error",
                                        "ML_Prediction": "Not applied",
                                        "Explanation": tr(f"Could not analyze: {e}", f"تعذر التحليل: {e}"),
                                        "Possible Causes": tr("Not available", "غير متوفر"),
                                        "Recommendations": tr("Check extracted data.", "تأكد من البيانات المستخرجة."),
                                    })
                            st.session_state["results_df"] = pd.DataFrame(analyzed)
                            st.session_state["csv_input_df"] = reviewed_df.copy()
                            save_analysis_log("smart_ai_upload", st.session_state["results_df"])
                            clear_generated_ai_explanations()
                            st.session_state["app_page"] = "results_page"
                            st.rerun()
                with c_clear:
                    if st.button(f"🗑️  {tr('Clear Extracted Values', 'مسح القيم المستخرجة')}", use_container_width=True):
                        st.session_state["smart_extracted_df"] = pd.DataFrame()
                        st.rerun()
        if method == "smart":
            pass

    # ── Page 3: Results page ──────────────────
    elif current_page == "results_page":
        render_progress("results_page")
        results_df = st.session_state["results_df"]

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button(f"← {tr('Back to Input', 'رجوع للإدخال')}", use_container_width=True):
                st.session_state["app_page"] = "input_page"
                st.rerun()
        with nav_col2:
            if st.button(f"🏠 {tr('Start New Analysis', 'تحليل جديد')}", use_container_width=True):
                st.session_state["manual_items"] = []
                st.session_state["csv_input_df"] = None
                st.session_state["results_df"] = pd.DataFrame()
                clear_generated_ai_explanations()
                go_to_choose()

        if results_df.empty:
            st.warning(tr("No results yet. Please enter data first.", "لا توجد نتائج حتى الآن. يرجى إدخال البيانات أولًا."))
        else:
            section_title(tr("Results Summary", "ملخص النتائج"), 26, "")
            render_summary_stats(results_df)

            section_title(tr("Results Table", "جدول النتائج"), 22, "")
            render_results_table(results_df)

            section_title(tr("Smart Result Cards", "بطاقات النتائج الذكية"), 22, "")
            for i, row in results_df.reset_index(drop=True).iterrows():
                render_result_card(row, card_key=str(i))

            render_full_patient_ai_analysis(results_df)

            valid_results = results_df[
                results_df["Status"].isin(["Deficient", "Normal", "Excessive"])
            ].copy()

            if not valid_results.empty:
                section_title(tr("Status Distribution", "توزيع الحالات"), 20)

                if is_arabic:
                    valid_results["Status_Display"] = valid_results["Status"].map({
                        "Deficient": "ناقص",
                        "Normal": "طبيعي",
                        "Excessive": "مرتفع"
                    })
                    label_col = "Status_Display"
                    color_map = {"ناقص": "#FF4B4B", "طبيعي": "#1DB954", "مرتفع": "#FFA500"}
                else:
                    label_col = "Status"
                    color_map = {"Deficient": "#FF4B4B", "Normal": "#1DB954", "Excessive": "#FFA500"}

                fig_summary = px.pie(
                    valid_results.groupby(label_col).size().reset_index(name="Count"),
                    names=label_col,
                    values="Count",
                    color=label_col,
                    color_discrete_map=color_map,
                    hole=0.55,
                    title=tr("Status Distribution", "توزيع الحالات"),
                )
                fig_summary.update_traces(
                    textfont=dict(color="white", size=13, family="Plus Jakarta Sans, Cairo"),
                    marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
                )
                fig_summary.update_layout(
                    height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9AAAB8", family="Plus Jakarta Sans, Cairo"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#B8C8D8")),
                    title=dict(font=dict(color="#B8C8D8", size=15)),
                )
                st.plotly_chart(fig_summary, use_container_width=True, config={"displayModeBar": False})

                section_title(tr("Reference Range Visualization", "مقارنة القيمة بالنطاق الطبيعي"), 20)

                for i, row in valid_results.reset_index(drop=True).iterrows():
                    fig_ref = create_reference_chart(row)
                    if fig_ref is not None:
                        st.plotly_chart(
                            fig_ref,
                            use_container_width=True,
                            config={
                                "displayModeBar": False,
                                "scrollZoom": False,
                                "doubleClick": False,
                                "staticPlot": True
                            },
                            key=f"ref_chart_{i}"
                        )

            invalid_results = results_df[results_df["Status"] == "Invalid"].copy()
            if not invalid_results.empty:
                section_title(tr("Input Warnings", "تنبيهات الإدخال"), 22, "⚠️")
                for _, row in invalid_results.iterrows():
                    st.error(tr(
                        f"{row['Nutrient']} value ({row['Value']}) looks unrealistic. Please check the input.",
                        f"قيمة {row['Nutrient']} ({row['Value']}) تبدو غير منطقية. يرجى التأكد من المدخلات."
                    ))

            section_title(tr("Download Results", "تحميل النتائج"), 22, "")
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label=f"  {tr('Download Results CSV', 'تحميل النتائج CSV')}",
                data=csv_data,
                file_name="vitavision_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

# =========================================
# ABOUT TAB
# =========================================
with about_tab:
    dir_val = "rtl" if is_arabic else "ltr"
    align   = "right" if is_arabic else "left"

    cards = [
        {
            "icon": "",
            "title": tr("About VitaVision", "عن VitaVision"),
            "text": tr(
                "VitaVision is an intelligent decision-support system designed to simplify the interpretation of vitamin and mineral laboratory results. It transforms raw lab values into meaningful health insights, helping users quickly understand whether their levels fall within a healthy range.",
                "VitaVision هو نظام ذكي لدعم اتخاذ القرار مصمم لتبسيط تفسير نتائج تحاليل الفيتامينات والمعادن، حيث يحول القيم المخبرية إلى معلومات صحية مفهومة تساعد المستخدم على معرفة حالته بسهولة."
            ),
            "items": [
                tr("Provides instant classification of lab results", "تصنيف فوري لنتائج التحاليل"),
                tr("Uses scientifically defined reference ranges", "يعتمد على نطاقات مرجعية علمية"),
                tr("Offers simplified explanations for better understanding", "يقدم شرح مبسط لفهم أفضل"),
                tr("Designed as an educational and supportive tool", "مصمم كأداة تعليمية داعمة"),
            ],
        },
        {
            "icon": "",
            "title": tr("Project Scope", "نطاق المشروع"),
            "text": tr(
                "The VitaVision system focuses on analyzing vitamin and mineral lab results by classifying them into Deficient, Normal, or Excessive categories. It enhances user understanding by providing contextual explanations, possible causes, and actionable recommendations.",
                "يركز نظام VitaVision على تحليل نتائج الفيتامينات والمعادن وتصنيفها إلى ناقص أو طبيعي أو مرتفع، مع تقديم تفسير واضح وأسباب محتملة وتوصيات عملية تساعد المستخدم."
            ),
            "items": [
                tr("Supports multiple nutrients and lab indicators", "يدعم عدة عناصر غذائية وتحاليل"),
                tr("Provides visual comparison with reference ranges", "يعرض مقارنة بصرية مع النطاق الطبيعي"),
                tr("Includes both manual input and CSV upload", "يدعم الإدخال اليدوي ورفع CSV"),
                tr("Helps users make informed health decisions", "يساعد المستخدم على اتخاذ قرارات صحية واعية"),
            ],
        },
        {
            "icon": "",
            "title": tr("Alignment with Saudi Vision 2030", "التوافق مع رؤية السعودية 2030"),
            "text": tr(
                "VitaVision supports Saudi Vision 2030 by promoting digital transformation in the healthcare sector. It enhances health awareness, empowers individuals to better understand their medical data, and contributes to improving the overall quality of life through smart health solutions.",
                "يدعم مشروع VitaVision رؤية السعودية 2030 من خلال تعزيز التحول الرقمي في القطاع الصحي، ورفع الوعي الصحي، وتمكين الأفراد من فهم بياناتهم الطبية بشكل أفضل، والمساهمة في تحسين جودة الحياة عبر حلول صحية ذكية."
            ),
            "items": [
                tr("Supports digital health transformation", "يدعم التحول الرقمي الصحي"),
                tr("Enhances health awareness in society", "يعزز الوعي الصحي في المجتمع"),
                tr("Empowers individuals with health insights", "يمكن الأفراد من فهم حالتهم الصحية"),
                tr("Contributes to improving quality of life", "يساهم في تحسين جودة الحياة"),
            ],
        },
        {
            "icon": "",
            "title": tr("How It Works", "كيف يعمل النظام"),
            "text": tr(
                "The system analyzes each nutrient value by comparing it against medically defined reference ranges. It then classifies the result and enhances it using intelligent logic to generate explanations, possible causes, and recommendations.",
                "يقوم النظام بتحليل كل قيمة غذائية بمقارنتها مع النطاقات المرجعية الطبية، ثم يصنف النتيجة ويضيف شرحًا ذكيًا مع الأسباب المحتملة والتوصيات."
            ),
            "items": [
                tr("Input → Processing → Classification", "إدخال ← معالجة ← تصنيف"),
                tr("Rule-based + Intelligent logic", "يعتمد على قواعد + منطق ذكي"),
                tr("Generates explanations and recommendations", "يولد شرح وتوصيات"),
            ],
        },
    ]

    for card in cards:
        items_html = "".join(f"<li>{it}</li>" for it in card["items"])
        st.html(f"""
<div class="vv-card" style="direction:{dir_val}; text-align:{align};">
    <div class="vv-card-title">
        <span style="margin-{'left' if not is_arabic else 'right'}:8px;">{card['icon']}</span>{card['title']}
    </div>
    <div class="vv-card-text">{card['text']}</div>
    <ul class="vv-card-list">{items_html}</ul>
</div>
""")

# =========================================
# CONTACT TAB
# =========================================
with contact_tab:
    dir_val = "rtl" if is_arabic else "ltr"
    align   = "right" if is_arabic else "left"

    st.html(f"""
<div class="vv-card" style="direction:{dir_val}; text-align:{align}; overflow:hidden;">
    <div class="vv-card-title">{tr("Contact Us", "تواصل معنا")}</div>

    <div class="vv-card-text" style="margin-bottom:22px;">
        {tr(
            "For any inquiries or feedback, feel free to reach out through the following channel:",
            "لأي استفسار أو ملاحظات، تواصل معنا عبر القناة التالية:"
        )}
    </div>

    <div style="
        display:flex;
        justify-content:center;
        width:100%;
    ">
        <a href="mailto:info@vitavision.com"
           style="
               text-decoration:none;
               width:100%;
               max-width:420px;
               min-width:0;
           ">
            <div style="
                background:rgba(0,191,255,0.05);
                border:1px solid rgba(0,191,255,0.18);
                border-radius:14px;
                padding:18px 16px;
                transition:all 0.2s;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                text-align:center;
                gap:8px;
                width:100%;
                min-width:0;
                overflow:hidden;
            ">
                <span style="font-size:24px;">📧</span>

                <div style="
                    font-size:11px;
                    color:#7A9BB5;
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                ">EMAIL</div>

                <div style="
                    font-size:14px;
                    font-weight:700;
                    color:#B8C8D8;
                    max-width:100%;
                    overflow-wrap:anywhere;
                    word-break:break-word;
                    line-height:1.4;
                ">
                    info@vitavision.com
                </div>
            </div>
        </a>
    </div>
</div>
""")

# =========================================
# Footer
# =========================================
st.html("""
<div style="
    text-align: center;
    font-size: 13px;
    color: #3A5060;
    margin-top: 48px;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
">
    <span style="color:#005A80;">Vita</span><span style="color:#007A9E;">Vision</span>
    &nbsp;·&nbsp; © 2026 All rights reserved
</div>
""")

# =========================================
# Show disclaimer if not agreed
# =========================================
if not st.session_state["disclaimer_agreed"]:
    show_disclaimer()

