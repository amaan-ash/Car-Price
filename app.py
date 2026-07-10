"""
AutoWorth AI - Used Car Price Prediction
==========================================
Premium SaaS-grade Streamlit dashboard for used car price prediction.

Design language inspired by Notion, Stripe Dashboard, Linear, Vercel and
modern automotive dashboards — glassmorphism, soft shadows, gradients and
purposeful motion.

Author   : Amaan Ali Shaikh
PRN      : 125BTCM2005
Subject  : Machine Learning
Dataset  : CarDekho Used Car Dataset

IMPORTANT
---------
This file ONLY contains presentation / UI logic. The Machine Learning
model, encoders and metadata are loaded exactly as trained and are never
modified, retrained or altered in any way. All prediction logic mirrors
the original backend contract:

    features (in order) = [km_driven, fuel, seller_type,
                            transmission, owner, brand, car_age]

Navigation
----------
v2.1.0 replaces the native st.sidebar with a fully custom, fixed-position
navigation panel rendered in the main DOM tree (st.container(key=...)).
This removes Streamlit's automatic sidebar collapse/hide behavior — the
panel only opens/closes when the user presses the toggle button, and its
state is tracked in st.session_state so it survives every rerun.

No extra third-party packages are required for this UI.
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AutoWorth AI | Intelligent Car Valuation",
    page_icon="🚗",
    layout="wide",
)

APP_VERSION = "v2.1.0"

# Nav items: (page key, icon, display label)
NAV_ITEMS = [
    ("Home", "🏠", "Home"),
    ("Predict Price", "🔮", "Predict Price"),
    ("Analytics", "📊", "Analytics"),
    ("About", "ℹ️", "About"),
]
NAV_EXPANDED_WIDTH = 272
NAV_COLLAPSED_WIDTH = 84

# ============================================================================
# CONSTANTS & PATHS  (unchanged — backend contract preserved)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "model", "car_price_model.pkl"),
    os.path.join(BASE_DIR, "car_price_model.pkl"),
]
ENCODER_CANDIDATES = [
    os.path.join(BASE_DIR, "model", "encoders.pkl"),
    os.path.join(BASE_DIR, "encoders.pkl"),
]
METADATA_CANDIDATES = [
    os.path.join(BASE_DIR, "model", "metadata.pkl"),
    os.path.join(BASE_DIR, "metadata.pkl"),
]
DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "dataset", "CAR DETAILS FROM CAR DEKHO.csv"),
    os.path.join(BASE_DIR, "CAR DETAILS FROM CAR DEKHO.csv"),
]

# Exact feature order the model was trained on — DO NOT REORDER
FEATURE_ORDER = ["km_driven", "fuel", "seller_type", "transmission", "owner", "brand", "car_age"]


# ============================================================================
# CUSTOM CSS — GLASSMORPHIC PREMIUM SAAS THEME
# ============================================================================
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-0: #05070d;
            --bg-1: #0a0e19;
            --bg-2: #0f1424;
            --surface: rgba(255,255,255,0.045);
            --surface-hover: rgba(255,255,255,0.075);
            --border: rgba(255,255,255,0.09);
            --border-hover: rgba(139,92,246,0.55);
            --text-1: #f4f5fb;
            --text-2: #b3b8cc;
            --text-3: #7d829a;
            --violet: #7c6bf2;
            --purple: #a855f7;
            --pink: #ec4899;
            --emerald: #10b981;
            --amber: #f59e0b;
            --grad-brand: linear-gradient(135deg, #6366f1 0%, #8b5cf6 45%, #ec4899 100%);
            --grad-emerald: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --radius-lg: 24px;
            --radius-md: 18px;
            --radius-sm: 12px;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif !important; }
        code, .mono { font-family: 'JetBrains Mono', monospace; }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stDecoration"] { display: none; }

        .stApp {
            background:
                radial-gradient(circle at 15% -10%, rgba(124,107,242,0.16) 0%, transparent 45%),
                radial-gradient(circle at 90% 10%, rgba(236,72,153,0.10) 0%, transparent 40%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-0) 100%);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        /* ---------------------------------------------------------------
           ANIMATIONS
        --------------------------------------------------------------- */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes floatGlow {
            0%, 100% { transform: translateY(0px); }
            50%      { transform: translateY(-10px); }
        }
        @keyframes shimmer {
            0%   { background-position: -400px 0; }
            100% { background-position: 400px 0; }
        }
        @keyframes pulseDot {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.55); }
            50%      { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
        }
        .fade-in { animation: fadeInUp 0.6s ease both; }
        .fade-in-1 { animation: fadeInUp 0.6s ease 0.08s both; }
        .fade-in-2 { animation: fadeInUp 0.6s ease 0.16s both; }
        .fade-in-3 { animation: fadeInUp 0.6s ease 0.24s both; }

        /* ---------------------------------------------------------------
           SIDEBAR
        --------------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1020 0%, #070a14 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * { color: var(--text-1); }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

        .brand-logo-wrap {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.4rem 0.2rem 1rem 0.2rem;
        }
        .brand-icon-badge {
            width: 42px; height: 42px;
            border-radius: 13px;
            background: var(--grad-brand);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.35rem;
            box-shadow: 0 8px 20px rgba(124,107,242,0.4);
            animation: floatGlow 4s ease-in-out infinite;
        }
        .brand-name {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 1.18rem;
            background: linear-gradient(135deg, #cbd5ff, #f5b8e6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
        }
        .brand-sub {
            font-size: 0.72rem;
            color: var(--text-3);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 600;
        }

        .sidebar-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.14), transparent);
            margin: 0.9rem 0;
            border: none;
        }

        .sidebar-info-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
        }
        .sidebar-info-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-3);
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .sidebar-info-value {
            font-size: 0.86rem;
            color: var(--text-1);
            font-weight: 600;
        }
        .live-dot {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--emerald);
            margin-right: 6px;
            animation: pulseDot 1.8s infinite;
        }
        .github-btn {
            display: flex; align-items: center; justify-content: center; gap: 0.5rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.6rem 1rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-1);
            transition: all 0.2s ease;
            cursor: pointer;
            text-decoration: none;
        }
        .github-btn:hover {
            background: rgba(255,255,255,0.11);
            border-color: var(--border-hover);
        }
        .sidebar-footer {
            font-size: 0.72rem;
            color: var(--text-3);
            text-align: center;
            margin-top: 0.6rem;
        }

        /* option-menu overrides injected via kwargs — extra polish here */
        div[data-testid="stSidebarUserContent"] nav { border-radius: var(--radius-md); }

        /* ---------------------------------------------------------------
           HERO
        --------------------------------------------------------------- */
        .hero {
            position: relative;
            background: var(--grad-brand);
            border-radius: 32px;
            padding: 3.4rem 3.2rem;
            margin-bottom: 1.8rem;
            overflow: hidden;
            box-shadow: 0 30px 70px rgba(99,102,241,0.32);
        }
        .hero::before {
            content: "";
            position: absolute; top: -60%; right: -8%;
            width: 560px; height: 560px;
            background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, transparent 70%);
        }
        .hero::after {
            content: "";
            position: absolute; bottom: -40%; left: -6%;
            width: 420px; height: 420px;
            background: radial-gradient(circle, rgba(255,255,255,0.09) 0%, transparent 70%);
        }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 0.4rem;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 0.42rem 1.05rem;
            border-radius: 999px;
            font-size: 0.8rem; font-weight: 700; color: #fff;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(255,255,255,0.32);
        }
        .hero-title {
            font-size: 3.15rem; font-weight: 900; color: #fff;
            margin: 0 0 0.7rem 0; line-height: 1.12;
            position: relative; z-index: 1;
        }
        .hero-subtitle {
            font-size: 1.16rem; color: rgba(255,255,255,0.94);
            max-width: 660px; line-height: 1.65; font-weight: 400;
            position: relative; z-index: 1;
        }
        .hero-cta-row { margin-top: 1.6rem; display: flex; gap: 0.8rem; position: relative; z-index: 1; }

        /* ---------------------------------------------------------------
           CARDS
        --------------------------------------------------------------- */
        .glass-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.6rem 1.7rem;
            backdrop-filter: blur(14px);
            transition: all 0.28s cubic-bezier(.2,.8,.2,1);
            height: 100%;
        }
        .glass-card:hover {
            border-color: var(--border-hover);
            transform: translateY(-5px);
            box-shadow: 0 16px 34px rgba(139,92,246,0.18);
            background: var(--surface-hover);
        }
        .feature-icon-badge {
            width: 46px; height: 46px;
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            background: linear-gradient(135deg, rgba(124,107,242,0.25), rgba(236,72,153,0.2));
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 0.8rem;
        }
        .feature-title { font-size: 1.03rem; font-weight: 700; color: var(--text-1); margin-bottom: 0.4rem; }
        .feature-desc { font-size: 0.88rem; color: var(--text-2); line-height: 1.55; }

        /* ---------------------------------------------------------------
           METRIC / KPI CARDS
        --------------------------------------------------------------- */
        .kpi-card {
            background: linear-gradient(160deg, rgba(124,107,242,0.16), rgba(236,72,153,0.09));
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.35rem 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::after {
            content: "";
            position: absolute; top: -30%; right: -20%;
            width: 120px; height: 120px; border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.08), transparent 70%);
        }
        .kpi-icon { font-size: 1.3rem; margin-bottom: 0.5rem; opacity: 0.9; }
        .kpi-label {
            font-size: 0.76rem; color: var(--text-2); font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 1.85rem; font-weight: 800; color: #fff;
            margin-top: 0.15rem; font-family: 'Poppins', sans-serif;
        }
        .kpi-delta { font-size: 0.76rem; color: var(--emerald); font-weight: 600; margin-top: 0.2rem; }

        /* ---------------------------------------------------------------
           SECTION HEADERS
        --------------------------------------------------------------- */
        .section-heading {
            font-size: 1.55rem; font-weight: 800; color: var(--text-1);
            margin: 2.4rem 0 0.35rem 0;
            display: flex; align-items: center; gap: 0.55rem;
        }
        .section-sub { color: var(--text-2); font-size: 0.95rem; margin-bottom: 1.3rem; }
        .eyebrow {
            font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--purple); margin-bottom: 0.3rem;
        }

        /* ---------------------------------------------------------------
           WORKFLOW / TIMELINE
        --------------------------------------------------------------- */
        .flow-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.3rem 1.1rem;
            text-align: center;
            position: relative;
            transition: all 0.25s ease;
        }
        .flow-card:hover { border-color: var(--border-hover); transform: translateY(-4px); }
        .flow-num {
            width: 36px; height: 36px; border-radius: 50%;
            background: var(--grad-brand); color: white;
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; margin: 0 auto 0.7rem auto; font-size: 0.9rem;
            box-shadow: 0 6px 16px rgba(124,107,242,0.4);
        }

        .timeline-item {
            display: flex; gap: 1rem; margin-bottom: 1.4rem; position: relative;
        }
        .timeline-dot {
            min-width: 14px; height: 14px; border-radius: 50%;
            background: var(--grad-brand); margin-top: 0.3rem;
            box-shadow: 0 0 0 4px rgba(124,107,242,0.16);
        }
        .timeline-line {
            position: absolute; left: 6px; top: 20px; bottom: -22px; width: 2px;
            background: linear-gradient(180deg, rgba(124,107,242,0.5), transparent);
        }
        .timeline-title { font-weight: 700; color: var(--text-1); font-size: 0.98rem; }
        .timeline-desc { color: var(--text-2); font-size: 0.86rem; margin-top: 0.15rem; line-height: 1.5; }

        /* ---------------------------------------------------------------
           PREDICTION PAGE
        --------------------------------------------------------------- */
        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.9rem 2.1rem;
            height: 100%;
        }
        .panel-title {
            font-size: 1.15rem; font-weight: 800; color: var(--text-1);
            display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;
        }
        .panel-sub { color: var(--text-3); font-size: 0.85rem; margin-bottom: 1.3rem; }

        div[data-testid="stForm"] { border: none; padding: 0; background: transparent; }

        .result-hero {
            background: var(--grad-emerald);
            border-radius: var(--radius-lg);
            padding: 2.2rem 2rem;
            text-align: center;
            box-shadow: 0 22px 55px rgba(16,185,129,0.32);
            position: relative;
            overflow: hidden;
        }
        .result-hero::before {
            content: "";
            position: absolute; top: -50%; right: -10%;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
        }
        .result-label {
            color: rgba(255,255,255,0.88); font-size: 0.85rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.08em; position: relative; z-index: 1;
        }
        .result-value {
            color: #fff; font-size: 2.9rem; font-weight: 900; margin: 0.25rem 0;
            font-family: 'Poppins', sans-serif; position: relative; z-index: 1;
        }
        .result-caption { color: rgba(255,255,255,0.85); font-size: 0.86rem; position: relative; z-index: 1; }

        .mini-stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .mini-stat-label { font-size: 0.7rem; color: var(--text-3); text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em; }
        .mini-stat-value { font-size: 1.05rem; color: var(--text-1); font-weight: 800; margin-top: 0.15rem; }

        .confidence-bar-track {
            width: 100%; height: 8px; border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden; margin-top: 0.5rem;
        }
        .confidence-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, #10b981, #34d399);
        }

        .status-pill {
            display: inline-flex; align-items: center; gap: 0.35rem;
            padding: 0.3rem 0.85rem; border-radius: 999px;
            font-size: 0.76rem; font-weight: 700;
            background: rgba(16,185,129,0.16); color: #34d399;
            border: 1px solid rgba(52,211,153,0.35);
        }

        /* ---------------------------------------------------------------
           BUTTONS / INPUTS
        --------------------------------------------------------------- */
        .stButton > button {
            background: var(--grad-brand);
            color: white; border: none; border-radius: 14px;
            padding: 0.9rem 1.6rem; font-weight: 700; font-size: 1.03rem;
            width: 100%;
            box-shadow: 0 12px 30px rgba(139,92,246,0.35);
            transition: all 0.22s cubic-bezier(.2,.8,.2,1);
        }
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 18px 40px rgba(139,92,246,0.5);
        }
        .stButton > button:active { transform: translateY(-1px) scale(0.99); }

        div[data-baseweb="select"] > div, .stNumberInput input {
            background: rgba(255,255,255,0.045) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
        }

        [data-testid="stMetricValue"] { font-family: 'Poppins', sans-serif; }

        /* ---------------------------------------------------------------
           SDG / TAGS
        --------------------------------------------------------------- */
        .tag-chip {
            display: inline-block;
            background: rgba(124,107,242,0.14);
            border: 1px solid rgba(124,107,242,0.35);
            color: #c9c1fb;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin: 0.2rem 0.3rem 0.2rem 0;
        }

        /* ---------------------------------------------------------------
           FOOTER
        --------------------------------------------------------------- */
        .footer-box {
            text-align: center; padding: 2.2rem 0 1rem 0;
            color: var(--text-3); font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.07);
            margin-top: 3rem;
        }
        .footer-box b { color: var(--text-2); }

        /* ---------------------------------------------------------------
           CUSTOM FIXED NAVIGATION PANEL (replaces st.sidebar entirely)
        --------------------------------------------------------------- */
        [class*="st-key-app-nav-panel"] {
            position: fixed !important;
            top: 0; left: 0; height: 100vh; z-index: 999;
            background: linear-gradient(180deg, rgba(14,18,32,0.88) 0%, rgba(7,9,18,0.94) 100%);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255,255,255,0.08);
            box-shadow: 8px 0 40px rgba(0,0,0,0.35);
            padding: 1.3rem 0.9rem 1rem 0.9rem;
            overflow-y: auto; overflow-x: hidden;
            transition: width 0.28s cubic-bezier(.2,.8,.2,1);
        }
        [class*="st-key-app-nav-panel"]::-webkit-scrollbar { width: 4px; }
        [class*="st-key-app-nav-panel"]::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }

        /* Nav buttons: override the global brand-gradient button style
           so the panel reads as a nav list, not a wall of CTA buttons */
        [class*="st-key-app-nav-panel"] .stButton > button {
            width: 100%;
            text-align: left;
            justify-content: flex-start;
            font-size: 0.92rem;
            font-weight: 600;
            padding: 0.65rem 0.85rem;
            border-radius: 12px;
            margin: 3px 0;
            box-shadow: none;
            transition: all 0.2s ease;
        }
        [class*="st-key-app-nav-panel"] .stButton > button[kind="secondary"] {
            background: transparent;
            color: var(--text-2);
            border: 1px solid transparent;
        }
        [class*="st-key-app-nav-panel"] .stButton > button[kind="secondary"]:hover {
            background: rgba(255,255,255,0.06);
            color: var(--text-1);
            border-color: var(--border);
            transform: none;
        }
        [class*="st-key-app-nav-panel"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #a855f7);
            color: #fff;
            border: none;
            box-shadow: 0 8px 20px rgba(124,107,242,0.35);
        }
        [class*="st-key-app-nav-panel"] .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
        }
        [class*="st-key-app-nav-panel"] div[data-testid="stHorizontalBlock"] .stButton > button {
            padding: 0.5rem 0.7rem;
        }

        .navp-logo-row {
            display: flex; align-items: center; gap: 0.65rem;
            padding: 0.2rem 0.1rem 1rem 0.1rem;
        }
        .navp-logo-badge {
            min-width: 40px; height: 40px; border-radius: 12px;
            background: var(--grad-brand);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.25rem; flex-shrink: 0;
            box-shadow: 0 8px 20px rgba(124,107,242,0.4);
            animation: floatGlow 4s ease-in-out infinite;
        }
        .navp-brand-name {
            font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.08rem;
            background: linear-gradient(135deg, #cbd5ff, #f5b8e6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            line-height: 1.1; white-space: nowrap;
        }
        .navp-brand-sub {
            font-size: 0.68rem; color: var(--text-3); letter-spacing: 0.04em;
            text-transform: uppercase; font-weight: 600; white-space: nowrap;
        }
        .navp-section-label {
            font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--text-3); font-weight: 700; margin: 0.9rem 0.3rem 0.35rem 0.3rem;
        }
        .navp-info-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius-sm); padding: 0.7rem 0.85rem; margin-bottom: 0.5rem;
        }
        .navp-info-label {
            font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--text-3); font-weight: 700; margin-bottom: 0.1rem;
        }
        .navp-info-value { font-size: 0.82rem; color: var(--text-1); font-weight: 600; }
        .navp-link-btn {
            display: flex; align-items: center; gap: 0.55rem;
            background: rgba(255,255,255,0.05); border: 1px solid var(--border);
            border-radius: 10px; padding: 0.55rem 0.8rem; margin-bottom: 0.45rem;
            font-size: 0.8rem; font-weight: 600; color: var(--text-1);
            text-decoration: none; transition: all 0.2s ease;
        }
        .navp-link-btn:hover { background: rgba(255,255,255,0.09); border-color: var(--border-hover); }
        .navp-footer {
            font-size: 0.68rem; color: var(--text-3); text-align: center;
            margin-top: 0.8rem; line-height: 1.5;
        }
        .navp-icon-only { text-align: center; font-size: 1.3rem; margin: 0.5rem 0; }

        /* Responsive fallback: no JS viewport hook in Streamlit, so collapse
           to icon-only rail automatically on narrow / tablet viewports */
        @media (max-width: 900px) {
            [class*="st-key-app-nav-panel"] { width: 84px !important; }
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# ARTIFACT LOADING  (unchanged backend contract)
# ============================================================================
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, encoders and metadata exactly as trained. Never retrains."""
    try:
        model_path = _first_existing(MODEL_CANDIDATES)
        enc_path = _first_existing(ENCODER_CANDIDATES)
        meta_path = _first_existing(METADATA_CANDIDATES)

        if not (model_path and enc_path and meta_path):
            missing = []
            if not model_path:
                missing.append("car_price_model.pkl")
            if not enc_path:
                missing.append("encoders.pkl")
            if not meta_path:
                missing.append("metadata.pkl")
            return None, None, None, f"Missing artifact(s): {', '.join(missing)}"

        model = joblib.load(model_path)
        encoders = joblib.load(enc_path)
        metadata = joblib.load(meta_path)
        return model, encoders, metadata, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the CarDekho dataset (read-only) used purely for analytics display."""
    path = _first_existing(DATASET_CANDIDATES)
    if not path:
        return None, "Dataset file not found."
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, str(e)


def safe_encode(encoder, value):
    """Encode a categorical value using the pre-fitted LabelEncoder, unchanged logic."""
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        classes = list(encoder.classes_)
        return int(encoder.transform([classes[0]])[0])


def predict_price(model, encoders, brand, fuel, seller_type, transmission, owner, car_age, km_driven):
    """Build the feature vector in the exact training order and predict. Unchanged logic."""
    row = {
        "km_driven": km_driven,
        "fuel": safe_encode(encoders["fuel"], fuel),
        "seller_type": safe_encode(encoders["seller_type"], seller_type),
        "transmission": safe_encode(encoders["transmission"], transmission),
        "owner": safe_encode(encoders["owner"], owner),
        "brand": safe_encode(encoders["brand"], brand),
        "car_age": car_age,
    }
    X = pd.DataFrame([[row[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    prediction = model.predict(X)[0]
    return max(float(prediction), 0.0)


def estimate_confidence(model, X_row):
    """
    Derive a stable, illustrative confidence score from the Random Forest's
    tree-to-tree agreement (lower spread across trees = higher confidence).
    Purely a display aid — does not alter the point prediction in any way.
    """
    try:
        tree_preds = np.array([t.predict(X_row)[0] for t in model.estimators_])
        mean_pred = tree_preds.mean()
        spread = tree_preds.std()
        if mean_pred <= 0:
            return 90.0
        rel_spread = spread / mean_pred
        confidence = max(60.0, min(99.0, 100.0 - rel_spread * 180))
        return round(confidence, 1)
    except Exception:
        return 92.0


def format_inr(amount):
    """Format a number as an Indian Rupee currency string with lakh grouping."""
    amount = round(amount)
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"₹{','.join(parts)},{last3}"


# ============================================================================
# CUSTOM FIXED NAVIGATION PANEL — replaces st.sidebar entirely
#
# Why not st.sidebar: Streamlit's native sidebar auto-collapses on narrow
# viewports and can be dragged/hidden by the user via its own toggle, which
# is exactly the "disappears unexpectedly" behavior we're removing. Instead
# this panel lives in the MAIN document tree (st.container(key=...)) and is
# pinned with `position: fixed` in CSS, so its open/closed state is fully
# owned by st.session_state and changes only when the toggle button below
# is pressed — never automatically.
# ============================================================================
def render_nav() -> str:
    """Render the fixed nav panel and return the currently active page."""
    if "nav_collapsed" not in st.session_state:
        st.session_state.nav_collapsed = False
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"
    if "nav_override" in st.session_state:
        st.session_state.active_page = st.session_state.pop("nav_override")

    collapsed = st.session_state.nav_collapsed
    panel_width = NAV_COLLAPSED_WIDTH if collapsed else NAV_EXPANDED_WIDTH

    # Push main content clear of the fixed panel + inject the live width
    # (the CSS transition on both rules animates the resize smoothly).
    st.markdown(f"""
    <style>
        [class*="st-key-app-nav-panel"] {{ width: {panel_width}px !important; }}
        .block-container {{ margin-left: {panel_width + 28}px !important; transition: margin-left 0.28s cubic-bezier(.2,.8,.2,1); }}
        @media (max-width: 900px) {{ .block-container {{ margin-left: {NAV_COLLAPSED_WIDTH + 28}px !important; }} }}
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="app-nav-panel"):
        # ---- Logo + collapse toggle ----
        if collapsed:
            st.markdown('<div class="navp-icon-only">🚗</div>', unsafe_allow_html=True)
            if st.button("»", key="nav_toggle", help="Expand navigation"):
                st.session_state.nav_collapsed = False
                st.rerun()
        else:
            st.markdown("""
            <div class="navp-logo-row">
                <div class="navp-logo-badge">🚗</div>
                <div>
                    <div class="navp-brand-name">AutoWorth AI</div>
                    <div class="navp-brand-sub">Car Valuation Engine</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("«  Collapse", key="nav_toggle", help="Collapse navigation"):
                st.session_state.nav_collapsed = True
                st.rerun()

        if not collapsed:
            st.markdown('<div class="navp-section-label">Navigate</div>', unsafe_allow_html=True)

        # ---- Nav links ----
        for page_key, icon, label in NAV_ITEMS:
            is_active = st.session_state.active_page == page_key
            btn_label = icon if collapsed else f"{icon}  {label}"
            if st.button(
                btn_label,
                key=f"nav_btn_{page_key}",
                type="primary" if is_active else "secondary",
                help=label if collapsed else None,
                use_container_width=True,
            ):
                st.session_state.active_page = page_key
                st.rerun()

        # ---- Status / meta ----
        if collapsed:
            st.markdown('<div class="navp-icon-only" title="Model Online">🟢</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="navp-section-label">System</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="navp-info-card">
                <div class="navp-info-label">Model Status</div>
                <div class="navp-info-value"><span class="live-dot"></span>Online &amp; Ready</div>
            </div>
            <div class="navp-info-card">
                <div class="navp-info-label">Algorithm</div>
                <div class="navp-info-value">Random Forest Regressor</div>
            </div>
            <div class="navp-info-card">
                <div class="navp-info-label">App Version</div>
                <div class="navp-info-value">{APP_VERSION}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="navp-section-label">Links</div>', unsafe_allow_html=True)
            st.markdown("""
            <a href="#" class="navp-link-btn" target="_blank"><span>🔗</span><span>GitHub</span></a>
            <a href="#" class="navp-link-btn" target="_blank"><span>💼</span><span>LinkedIn</span></a>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="navp-footer">
                Developed by <b style="color:#e8ebf5;">Amaan Ali Shaikh</b><br>
                PRN&nbsp;125BTCM2005 &nbsp;·&nbsp; © 2026
            </div>
            """, unsafe_allow_html=True)

    return st.session_state.active_page


# ============================================================================
# REUSABLE UI COMPONENTS
# ============================================================================
def kpi_card(icon, label, value, delta=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def feature_card(icon, title, desc):
    st.markdown(f"""
    <div class="glass-card fade-in">
        <div class="feature-icon-badge">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def flow_step(num, icon, title, desc):
    st.markdown(f"""
    <div class="flow-card">
        <div class="flow-num">{num}</div>
        <div style="font-size:1.5rem;">{icon}</div>
        <div class="feature-title" style="margin-top:0.35rem; font-size:0.95rem;">{title}</div>
        <div class="feature-desc" style="font-size:0.8rem;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="footer-box">
        Built with ❤️ using <b>Streamlit</b> &amp; <b>Scikit-learn</b> &nbsp;·&nbsp;
        AutoWorth AI © 2026 &nbsp;·&nbsp; Developed by <b>Amaan Ali Shaikh</b>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: HOME
# ============================================================================
def page_home(metadata, df):
    st.markdown("""
    <div class="hero fade-in">
        <div class="hero-badge">⚡ AI-Powered Valuation Engine</div>
        <div class="hero-title">Know Your Car's True<br>Worth, Instantly.</div>
        <div class="hero-subtitle">
            AutoWorth AI blends a production-grade Random Forest model with real-world
            CarDekho market data to deliver instant, trustworthy resale price estimates —
            built like a product, not a prototype.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Statistics cards ----
    st.markdown('<div class="eyebrow">📈 Live Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Dataset at a Glance</div>', unsafe_allow_html=True)

    total_records = len(df) if df is not None else "—"
    total_brands = len(metadata.get("brands", []))
    total_fuel = len(metadata.get("fuel", []))
    avg_price = format_inr(df["selling_price"].mean()) if (df is not None and "selling_price" in df.columns) else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("🚘", "Total Listings", total_records)
    with c2:
        kpi_card("🏷️", "Car Brands", total_brands)
    with c3:
        kpi_card("⛽", "Fuel Types", total_fuel)
    with c4:
        kpi_card("💰", "Avg. Selling Price", avg_price)

    # ---- Dataset / Algorithm / Prediction cards ----
    st.markdown('<div class="section-heading">🧩 How It All Comes Together</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("🗃️", "Dataset Card", "Real CarDekho used-car listings covering brand, fuel, ownership, mileage and more.")
    with c2:
        feature_card("🌲", "Algorithm Card", "A 500-tree Random Forest Regressor trained to capture non-linear pricing patterns.")
    with c3:
        feature_card("🎯", "Prediction Card", "Instant, explainable price estimates delivered through a clean, guided form.")

    # ---- ML Workflow ----
    st.markdown('<div class="section-heading">⚙️ Machine Learning Workflow</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("1", "📥", "Data Collection", "CarDekho used car listings"),
        ("2", "🧹", "Preprocessing", "Cleaning & label encoding"),
        ("3", "🌲", "Model Training", "Random Forest Regressor"),
        ("4", "🎯", "Prediction", "Instant price estimation"),
    ]
    for col, (num, icon, title, desc) in zip(cols, steps):
        with col:
            flow_step(num, icon, title, desc)

    # ---- Feature cards ----
    st.markdown('<div class="section-heading">✨ Why AutoWorth AI</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("⚡", "Instant Predictions", "Sub-second price estimates, no waiting rooms."),
        ("🎯", "Data-Driven Accuracy", "Trained on real market listings, not guesswork."),
        ("📊", "Rich Analytics", "Explore trends across brands, fuel types & more."),
        ("🔒", "Reliable & Consistent", "The same trusted model powers every prediction."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            feature_card(icon, title, desc)

    # ---- CTA ----
    st.markdown('<div class="section-heading">🚀 Ready When You Are</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1.2rem;">
        <div>
            <div class="feature-title" style="font-size:1.25rem;">Get a valuation in under 30 seconds</div>
            <div class="feature-desc">Jump to the Predict Price page and answer a handful of quick questions.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    cta_col, _ = st.columns([1, 3])
    with cta_col:
        if st.button("🔮 Start Predicting →", key="home_cta"):
            st.session_state["nav_override"] = "Predict Price"
            st.rerun()

    render_footer()


# ============================================================================
# PAGE: PREDICT PRICE
# ============================================================================
def page_predict(model, encoders, metadata):
    st.markdown('<div class="eyebrow">🔮 Valuation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Predict Your Car\'s Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter your vehicle details on the left — your price estimate and confidence appear on the right.</div>', unsafe_allow_html=True)

    brands = sorted(metadata.get("brands", []))
    fuels = metadata.get("fuel", [])
    seller_types = metadata.get("seller_type", [])
    transmissions = metadata.get("transmission", [])
    owners = metadata.get("owner", [])
    current_year = metadata.get("current_year", 2026)

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🚗 Vehicle Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">All fields are read directly from the trained model\'s metadata.</div>', unsafe_allow_html=True)

        with st.form("predict_form"):
            st.markdown("**Identity**")
            c1, c2 = st.columns(2)
            with c1:
                brand = st.selectbox("🏷️ Brand", brands)
            with c2:
                fuel = st.selectbox("⛽ Fuel Type", fuels)

            st.markdown("---")
            st.markdown("**Ownership & Sale**")
            c3, c4 = st.columns(2)
            with c3:
                seller_type = st.selectbox("🧑‍💼 Seller Type", seller_types)
            with c4:
                owner = st.selectbox("👤 Ownership", owners)

            st.markdown("---")
            st.markdown("**Usage**")
            c5, c6 = st.columns(2)
            with c5:
                transmission = st.selectbox("⚙️ Transmission", transmissions)
            with c6:
                car_age = st.number_input("📅 Car Age (years)", min_value=0, max_value=40, value=5, step=1,
                                           help=f"Relative to current year {current_year}")
            km_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, max_value=1_000_000,
                                         value=40000, step=1000)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Predict Selling Price")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Prediction Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Your result card will populate here once you predict.</div>', unsafe_allow_html=True)

        if submitted:
            try:
                encoded_row = {
                    "km_driven": km_driven,
                    "fuel": safe_encode(encoders["fuel"], fuel),
                    "seller_type": safe_encode(encoders["seller_type"], seller_type),
                    "transmission": safe_encode(encoders["transmission"], transmission),
                    "owner": safe_encode(encoders["owner"], owner),
                    "brand": safe_encode(encoders["brand"], brand),
                    "car_age": car_age,
                }
                X_row = pd.DataFrame([[encoded_row[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

                with st.spinner("Analyzing car details..."):
                    predicted_price = max(float(model.predict(X_row)[0]), 0.0)
                    confidence = estimate_confidence(model, X_row)

                timestamp = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

                st.markdown(f"""
                <div class="result-hero">
                    <div class="result-label">Estimated Selling Price</div>
                    <div class="result-value">{format_inr(predicted_price)}</div>
                    <div class="result-caption">{brand} · {fuel} · {transmission} · {car_age} yrs · {km_driven:,} km</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card">
                    <div class="mini-stat-label">Prediction Confidence</div>
                    <div class="mini-stat-value" style="font-size:1.3rem;">{confidence}%</div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width:{confidence}%;"></div>
                    </div>
                    <div class="feature-desc" style="margin-top:0.5rem;">Derived from agreement across the forest's decision trees.</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="mini-stat-card">
                        <div class="mini-stat-label">Algorithm</div>
                        <div class="mini-stat-value" style="font-size:0.9rem;">Random Forest</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="mini-stat-card">
                        <div class="mini-stat-label">Status</div>
                        <div class="mini-stat-value" style="font-size:0.9rem;"><span class="status-pill">● Ready</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="mini-stat-card">
                        <div class="mini-stat-label">Predicted At</div>
                        <div class="mini-stat-value" style="font-size:0.8rem;">{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:2.6rem 1.5rem;">
                <div style="font-size:2.4rem; margin-bottom:0.6rem;">🧭</div>
                <div class="feature-title">No prediction yet</div>
                <div class="feature-desc">Fill in the vehicle details and click <b style="color:#e8ebf5;">Predict Selling Price</b> to see results here.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
def page_analytics(df):
    st.markdown('<div class="eyebrow">📊 Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Trends and patterns mined from the CarDekho used car dataset.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Dataset file not found. Place `CAR DETAILS FROM CAR DEKHO.csv` inside the `dataset/` folder to enable analytics charts.")
        return

    plot_template = "plotly_dark"
    color_seq = px.colors.sequential.Plasma

    # ---- KPI row ----
    total_cars = len(df)
    avg_price = format_inr(df["selling_price"].mean()) if "selling_price" in df.columns else "—"
    max_price = format_inr(df["selling_price"].max()) if "selling_price" in df.columns else "—"
    min_price = format_inr(df["selling_price"].min()) if "selling_price" in df.columns else "—"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("🚘", "Total Cars", f"{total_cars:,}")
    with k2:
        kpi_card("💰", "Average Price", avg_price)
    with k3:
        kpi_card("📈", "Maximum Price", max_price)
    with k4:
        kpi_card("📉", "Minimum Price", min_price)

    def chart_container(chart_title):
        st.markdown(f"<div class='glass-card' style='padding:1.2rem 1.3rem;'><div class='feature-title' style='margin-bottom:0.6rem;'>{chart_title}</div>", unsafe_allow_html=True)

    def close_chart_container():
        st.markdown("</div>", unsafe_allow_html=True)

    common_layout = dict(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10), height=330, font=dict(family="Inter"))

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        chart_container("💰 Selling Price Distribution")
        if "selling_price" in df.columns:
            fig = px.histogram(df, x="selling_price", nbins=40, color_discrete_sequence=["#8b5cf6"])
            fig.update_layout(**common_layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'selling_price' not found in dataset.")
        close_chart_container()

    with c2:
        chart_container("⛽ Fuel Type Distribution")
        if "fuel" in df.columns:
            counts = df["fuel"].value_counts().reset_index()
            counts.columns = ["fuel", "count"]
            fig = px.pie(counts, names="fuel", values="count", hole=0.55, color_discrete_sequence=color_seq)
            fig.update_layout(**common_layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'fuel' not found in dataset.")
        close_chart_container()

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        chart_container("⚙️ Transmission Distribution")
        if "transmission" in df.columns:
            counts = df["transmission"].value_counts().reset_index()
            counts.columns = ["transmission", "count"]
            fig = px.bar(counts, x="transmission", y="count", color="transmission", color_discrete_sequence=color_seq)
            fig.update_layout(**common_layout, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'transmission' not found in dataset.")
        close_chart_container()

    with c4:
        chart_container("👤 Owner Distribution")
        if "owner" in df.columns:
            counts = df["owner"].value_counts().reset_index()
            counts.columns = ["owner", "count"]
            fig = px.bar(counts, x="count", y="owner", orientation="h", color="owner", color_discrete_sequence=color_seq)
            fig.update_layout(**common_layout, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'owner' not found in dataset.")
        close_chart_container()

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        chart_container("🏆 Top 10 Brands")
        brand_col = "brand" if "brand" in df.columns else ("name" if "name" in df.columns else None)
        if brand_col:
            temp = df.copy()
            if brand_col == "name":
                temp["brand"] = temp["name"].astype(str).str.split().str[0]
                brand_col = "brand"
            counts = temp[brand_col].value_counts().nlargest(10).reset_index()
            counts.columns = ["brand", "count"]
            fig = px.bar(counts, x="count", y="brand", orientation="h", color="count", color_continuous_scale=color_seq)
            fig.update_layout(**common_layout, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand information not found in dataset.")
        close_chart_container()

    with c6:
        chart_container("📅 Cars by Year")
        if "year" in df.columns:
            counts = df["year"].value_counts().sort_index().reset_index()
            counts.columns = ["year", "count"]
            fig = px.line(counts, x="year", y="count", markers=True, color_discrete_sequence=["#ec4899"])
            fig.update_layout(**common_layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'year' not found in dataset.")
        close_chart_container()

    # Row 4
    chart_container("🧑‍💼 Cars by Seller Type")
    if "seller_type" in df.columns:
        counts = df["seller_type"].value_counts().reset_index()
        counts.columns = ["seller_type", "count"]
        fig = px.bar(counts, x="seller_type", y="count", color="seller_type", color_discrete_sequence=color_seq, text="count")
        fig.update_layout(**{**common_layout, "height": 360}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'seller_type' not found in dataset.")
    close_chart_container()


# ============================================================================
# PAGE: ABOUT
# ============================================================================
def page_about():
    st.markdown('<div class="eyebrow">ℹ️ Project Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">About AutoWorth AI</div>', unsafe_allow_html=True)

    colA, colB = st.columns([1.35, 1], gap="large")

    with colA:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title" style="font-size:1.15rem;">🚗 The Project</div>
            <p class="feature-desc" style="margin-top:0.55rem;">
            AutoWorth AI is a machine learning powered web application designed to predict
            the fair resale value of used cars. It was developed as an academic Machine
            Learning project, applying a full ML pipeline — data preprocessing, label
            encoding, model training and deployment — inside an interactive,
            production-style Streamlit dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading" style="margin-top:1.8rem; font-size:1.2rem;">🕒 Project Workflow</div>', unsafe_allow_html=True)
        timeline = [
            ("Data Collection", "Sourced real-world used car listings from the CarDekho dataset."),
            ("Data Cleaning & Preprocessing", "Handled missing values and standardized categorical fields."),
            ("Feature Encoding", "Applied Label Encoding to brand, fuel, seller type, transmission and owner."),
            ("Model Training", "Trained a 500-tree Random Forest Regressor on the processed features."),
            ("Evaluation", "Validated performance and consistency of price predictions."),
            ("Deployment", "Wrapped the model in this premium Streamlit dashboard for interactive use."),
        ]
        for i, (title, desc) in enumerate(timeline):
            is_last = i == len(timeline) - 1
            line_html = "" if is_last else '<div class="timeline-line"></div>'
            st.markdown(f"""
            <div class="timeline-item">
                <div style="position:relative;">
                    <div class="timeline-dot"></div>
                    {line_html}
                </div>
                <div>
                    <div class="timeline-title">{title}</div>
                    <div class="timeline-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading" style="font-size:1.2rem;">🌍 SDG Alignment</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <p class="feature-desc">This project aligns with the following UN Sustainable Development Goals:</p>
            <div style="margin-top:0.6rem;">
                <span class="tag-chip">🏭 SDG 9 — Industry, Innovation &amp; Infrastructure</span>
                <span class="tag-chip">🏙️ SDG 11 — Sustainable Cities &amp; Communities</span>
                <span class="tag-chip">♻️ SDG 12 — Responsible Consumption</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading" style="font-size:1.2rem;">🔭 Future Scope</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            feature_card("🖼️", "Image-Based Valuation", "Estimate condition and price directly from uploaded car photos.")
            feature_card("🌐", "Live Market Sync", "Pull real-time listings to keep predictions current.")
        with f2:
            feature_card("📱", "Mobile App", "A companion app for on-the-go valuations.")
            feature_card("🧠", "Explainable AI", "Show feature-level impact behind every prediction.")

    with colB:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="width:74px; height:74px; border-radius:50%; margin:0 auto 0.9rem auto;
                        background: linear-gradient(135deg,#6366f1,#ec4899); display:flex;
                        align-items:center; justify-content:center; font-size:2rem;">👨‍💻</div>
            <div class="feature-title" style="font-size:1.1rem;">Amaan Ali Shaikh</div>
            <div class="feature-desc" style="margin-bottom:0.9rem;">PRN: 125BTCM2005</div>
            <div style="text-align:left;">
                <div class="sidebar-info-card" style="background:rgba(255,255,255,0.03);">
                    <div class="sidebar-info-label">Subject</div>
                    <div class="sidebar-info-value">Machine Learning</div>
                </div>
                <div class="sidebar-info-card" style="background:rgba(255,255,255,0.03);">
                    <div class="sidebar-info-label">Role</div>
                    <div class="sidebar-info-value">ML Engineer &amp; Developer</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="feature-title" style="margin-bottom:0.8rem;">🛠️ Technology Stack</div>', unsafe_allow_html=True)
        stack = [
            ("🐍", "Python", "Core programming language"),
            ("🎈", "Streamlit", "Interactive web application framework"),
            ("🐼", "Pandas", "Data manipulation & analysis"),
            ("🔢", "NumPy", "Numerical computations"),
            ("🤖", "Scikit-learn", "Machine learning model"),
            ("📊", "Plotly", "Interactive data visualizations"),
            ("📦", "Joblib", "Model serialization"),
        ]
        for icon, name, desc in stack:
            st.markdown(f"""
            <div class="glass-card" style="padding:0.75rem 1rem; margin-bottom:0.55rem; display:flex; align-items:center; gap:0.75rem;">
                <div style="font-size:1.3rem;">{icon}</div>
                <div>
                    <div style="font-weight:700; color:var(--text-1); font-size:0.92rem;">{name}</div>
                    <div style="color:var(--text-3); font-size:0.78rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="feature-title" style="margin-bottom:0.6rem;">🧬 ML Pipeline</div>', unsafe_allow_html=True)
        pipeline_steps = ["Raw Data", "Cleaning", "Encoding", "Random Forest", "Prediction"]
        pipeline_html = " → ".join([f'<span class="tag-chip">{s}</span>' for s in pipeline_steps])
        st.markdown(f'<div class="glass-card">{pipeline_html}</div>', unsafe_allow_html=True)

    render_footer()


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    load_css()

    model, encoders, metadata, error = load_artifacts()
    df, _ = load_dataset()

    page = render_nav()

    if error:
        st.error(f"⚠️ Failed to load model artifacts: {error}\n\nPlease ensure `car_price_model.pkl`, "
                  f"`encoders.pkl`, and `metadata.pkl` are present in the `model/` directory.")
        if page not in ["Home", "About"]:
            return

    if page == "Home":
        page_home(metadata or {}, df)
    elif page == "Predict Price":
        if metadata and encoders and model:
            page_predict(model, encoders, metadata)
        else:
            st.warning("Prediction is unavailable because model artifacts could not be loaded.")
    elif page == "Analytics":
        page_analytics(df)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()