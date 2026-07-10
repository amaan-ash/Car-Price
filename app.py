"""
AutoWorth AI - Used Car Price Prediction
==========================================
A premium, production-quality Streamlit dashboard that predicts used car
selling prices using a pre-trained Random Forest Regression model.

Author   : Amaan Ali Shaikh
PRN      : 125BTCM2005
Subject  : Machine Learning
Dataset  : CarDekho Used Car Dataset

NOTE: This application only performs INFERENCE. It loads pre-trained
artifacts (model, encoders, metadata) and never retrains the model.
"""

import os
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
    page_title="AutoWorth AI | Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS & PATHS
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

# Exact feature order the model was trained on
FEATURE_ORDER = ["km_driven", "fuel", "seller_type", "transmission", "owner", "brand", "car_age"]


# ============================================================================
# CUSTOM CSS - PREMIUM SAAS DASHBOARD LOOK
# ============================================================================
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at 10% 0%, #131a2b 0%, #0b0f1a 45%, #05070d 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1424 0%, #090c16 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #e8ebf5 !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* ---------- Hero Section ---------- */
        .hero-container {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            border-radius: 28px;
            padding: 3.2rem 3rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(99, 102, 241, 0.35);
            position: relative;
            overflow: hidden;
        }
        .hero-container::before {
            content: "";
            position: absolute;
            top: -50%; right: -10%;
            width: 500px; height: 500px;
            background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            backdrop-filter: blur(10px);
            padding: 0.4rem 1rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 0.6rem 0;
            line-height: 1.15;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            color: rgba(255,255,255,0.92);
            max-width: 680px;
            line-height: 1.6;
            font-weight: 400;
        }

        /* ---------- Generic Card ---------- */
        .glass-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            backdrop-filter: blur(12px);
            transition: all 0.25s ease;
            height: 100%;
        }
        .glass-card:hover {
            border-color: rgba(139, 92, 246, 0.5);
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(139, 92, 246, 0.15);
        }

        .feature-icon {
            font-size: 2.2rem;
            margin-bottom: 0.6rem;
        }
        .feature-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f1f2f9;
            margin-bottom: 0.4rem;
        }
        .feature-desc {
            font-size: 0.9rem;
            color: #9ca3b8;
            line-height: 1.5;
        }

        /* ---------- Metric Cards ---------- */
        .metric-card {
            background: linear-gradient(145deg, rgba(99,102,241,0.14), rgba(236,72,153,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            text-align: left;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #9ca3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 0.2rem;
            font-family: 'Poppins', sans-serif;
        }

        /* ---------- Section Heading ---------- */
        .section-heading {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f1f2f9;
            margin: 2.2rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-sub {
            color: #9ca3b8;
            font-size: 0.95rem;
            margin-bottom: 1.2rem;
        }

        /* ---------- Workflow Steps ---------- */
        .step-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
            position: relative;
        }
        .step-num {
            width: 34px; height: 34px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #ec4899);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin: 0 auto 0.6rem auto;
        }

        /* ---------- Prediction Result Card ---------- */
        .result-card {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 24px;
            padding: 2.4rem;
            text-align: center;
            box-shadow: 0 20px 50px rgba(16, 185, 129, 0.35);
            margin-top: 1.2rem;
        }
        .result-label {
            color: rgba(255,255,255,0.85);
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .result-value {
            color: #ffffff;
            font-size: 3.2rem;
            font-weight: 800;
            margin: 0.3rem 0;
            font-family: 'Poppins', sans-serif;
        }
        .result-caption {
            color: rgba(255,255,255,0.85);
            font-size: 0.95rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.4);
        }

        /* ---------- Form container ---------- */
        .form-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 22px;
            padding: 2rem 2.2rem;
        }

        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }

        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #ec4899 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 1.5rem;
            font-weight: 700;
            font-size: 1.05rem;
            width: 100%;
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.35);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 36px rgba(139, 92, 246, 0.5);
        }

        /* ---------- Footer ---------- */
        .footer-box {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #6b7280;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.07);
            margin-top: 3rem;
        }

        /* Sidebar brand */
        .sidebar-brand {
            font-size: 1.4rem;
            font-weight: 800;
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }
        .sidebar-tag {
            font-size: 0.78rem;
            color: #8b91a7;
            margin-bottom: 1.4rem;
        }

        [data-testid="stMetricValue"] {
            font-family: 'Poppins', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# ARTIFACT LOADING (cached)
# ============================================================================
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, encoders and metadata. Returns (model, encoders, metadata, error)."""
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
    """Load the CarDekho dataset for analytics. Returns (df, error)."""
    path = _first_existing(DATASET_CANDIDATES)
    if not path:
        return None, "Dataset file not found."
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, str(e)


def safe_encode(encoder, value):
    """Encode a categorical value, falling back safely if unseen."""
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        classes = list(encoder.classes_)
        return int(encoder.transform([classes[0]])[0])


def predict_price(model, encoders, brand, fuel, seller_type, transmission, owner, car_age, km_driven):
    """Build the feature vector in the exact training order and predict."""
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
# SIDEBAR NAVIGATION
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🚗 AutoWorth AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tag">Smart Used Car Valuation</div>', unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔮 Predict Price", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.82rem; color:#8b91a7; line-height:1.7;">
            <b style="color:#e8ebf5;">Model</b><br>Random Forest Regressor<br><br>
            <b style="color:#e8ebf5;">Dataset</b><br>CarDekho Used Cars<br><br>
            <b style="color:#e8ebf5;">Status</b><br><span class="status-pill">● Ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("© 2026 AutoWorth AI · Built by Amaan Ali Shaikh")
    return page


# ============================================================================
# PAGE: HOME
# ============================================================================
def page_home(metadata, df):
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Powered by Machine Learning</div>
        <div class="hero-title">Know Your Car's True Worth,<br>Instantly.</div>
        <div class="hero-subtitle">
            AutoWorth AI uses a trained Random Forest Regression model on real-world
            CarDekho listings to estimate the fair market resale value of any used car —
            in seconds, with zero guesswork.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Dataset statistics ----
    st.markdown('<div class="section-heading">📈 Dataset at a Glance</div>', unsafe_allow_html=True)
    total_records = len(df) if df is not None else "—"
    total_brands = len(metadata.get("brands", []))
    total_fuel = len(metadata.get("fuel", []))
    avg_price = format_inr(df["selling_price"].mean()) if (df is not None and "selling_price" in df.columns) else "—"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Listings", "Car Brands", "Fuel Types", "Avg. Selling Price"],
        [total_records, total_brands, total_fuel, avg_price],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Algorithm / description ----
    st.markdown('<div class="section-heading">🧠 About The Project</div>', unsafe_allow_html=True)
    colA, colB = st.columns([1.4, 1])
    with colA:
        st.markdown("""
        <div class="glass-card">
            <p class="feature-desc" style="font-size:0.98rem;">
            Buying or selling a used car often means guessing at a fair price. AutoWorth AI
            removes that uncertainty by learning pricing patterns from thousands of real
            CarDekho listings — factoring in brand, fuel type, transmission, ownership
            history, mileage, and age — to deliver a data-driven price estimate you can trust.
            </p>
            <p class="feature-desc" style="font-size:0.98rem; margin-top:0.6rem;">
            The prediction engine is a <b style="color:#e8ebf5;">Random Forest Regression</b>
            model — an ensemble of decision trees that captures complex, non-linear
            relationships between a car's attributes and its resale value.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-icon">🌲</div>
            <div class="feature-title">Random Forest Regression</div>
            <div class="feature-desc">
            An ensemble learning method that builds multiple decision trees and
            averages their outputs for robust, low-variance predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- ML Workflow ----
    st.markdown('<div class="section-heading">⚙️ Machine Learning Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1", "📥", "Data Collection", "CarDekho used car listings"),
        ("2", "🧹", "Preprocessing", "Cleaning & label encoding"),
        ("3", "🌲", "Model Training", "Random Forest Regressor"),
        ("4", "🎯", "Prediction", "Instant price estimation"),
    ]
    cols = st.columns(4)
    for col, (num, icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style="font-size:1.6rem;">{icon}</div>
                <div class="feature-title" style="margin-top:0.3rem;">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Feature cards ----
    st.markdown('<div class="section-heading">✨ Why AutoWorth AI</div>', unsafe_allow_html=True)
    features = [
        ("⚡", "Instant Predictions", "Get a price estimate in under a second, no waiting."),
        ("🎯", "Data-Driven Accuracy", "Trained on real market listings, not guesswork."),
        ("📊", "Rich Analytics", "Explore market trends across brands, fuel & more."),
        ("🔒", "Reliable & Consistent", "Same trusted model powers every prediction."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Get started ----
    st.markdown('<div class="section-heading">🚀 Get Started</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
        <div>
            <div class="feature-title" style="font-size:1.2rem;">Ready to value your car?</div>
            <div class="feature-desc">Head to the Predict Price page and fill in a few quick details.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-box">
        Built with ❤️ using Streamlit &nbsp;|&nbsp; AutoWorth AI © 2026 &nbsp;|&nbsp;
        Developed by Amaan Ali Shaikh
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: PREDICT PRICE
# ============================================================================
def page_predict(model, encoders, metadata):
    st.markdown('<div class="section-heading">🔮 Predict Your Car\'s Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in your car\'s details below and let the model estimate a fair selling price.</div>', unsafe_allow_html=True)

    brands = sorted(metadata.get("brands", []))
    fuels = metadata.get("fuel", [])
    seller_types = metadata.get("seller_type", [])
    transmissions = metadata.get("transmission", [])
    owners = metadata.get("owner", [])
    current_year = metadata.get("current_year", 2026)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox("🏷️ Brand", brands)
            fuel = st.selectbox("⛽ Fuel Type", fuels)
            transmission = st.selectbox("⚙️ Transmission", transmissions)
        with col2:
            seller_type = st.selectbox("🧑‍💼 Seller Type", seller_types)
            owner = st.selectbox("👤 Ownership", owners)
            car_age = st.number_input("📅 Car Age (years)", min_value=0, max_value=40, value=5, step=1,
                                       help=f"Based on current year {current_year}")
        with col3:
            km_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, max_value=1_000_000,
                                         value=40000, step=1000)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="glass-card" style="padding:1rem;">
                <div class="feature-desc">🤖 <b style="color:#e8ebf5;">Algorithm:</b> Random Forest Regression</div>
                <div class="feature-desc" style="margin-top:0.3rem;">📌 <b style="color:#e8ebf5;">Model Status:</b> <span class="status-pill">● Ready</span></div>
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀 Predict Selling Price")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        try:
            with st.spinner("Analyzing car details..."):
                predicted_price = predict_price(
                    model, encoders, brand, fuel, seller_type,
                    transmission, owner, car_age, km_driven
                )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimated Selling Price</div>
                <div class="result-value">{format_inr(predicted_price)}</div>
                <div class="result-caption">Based on {brand} · {fuel} · {transmission} · {car_age} yrs · {km_driven:,} km</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Algorithm Used</div>
                    <div class="metric-value" style="font-size:1.2rem;">Random Forest</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Status</div>
                    <div class="metric-value" style="font-size:1.2rem;">● Ready</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence Basis</div>
                    <div class="metric-value" style="font-size:1.2rem;">500 Trees</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
def page_analytics(df):
    st.markdown('<div class="section-heading">📊 Market Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore trends and patterns from the CarDekho used car dataset.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Dataset file not found. Place `CAR DETAILS FROM CAR DEKHO.csv` inside the `dataset/` folder to enable analytics charts.")
        return

    plot_template = "plotly_dark"
    color_seq = px.colors.sequential.Plasma

    # Row 1: Selling Price Distribution & Fuel Type Distribution
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 💰 Selling Price Distribution")
        if "selling_price" in df.columns:
            fig = px.histogram(df, x="selling_price", nbins=40, color_discrete_sequence=["#8b5cf6"])
            fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'selling_price' not found in dataset.")

    with c2:
        st.markdown("##### ⛽ Fuel Type Distribution")
        if "fuel" in df.columns:
            counts = df["fuel"].value_counts().reset_index()
            counts.columns = ["fuel", "count"]
            fig = px.pie(counts, names="fuel", values="count", hole=0.5, color_discrete_sequence=color_seq)
            fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'fuel' not found in dataset.")

    # Row 2: Transmission Distribution & Owner Distribution
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### ⚙️ Transmission Distribution")
        if "transmission" in df.columns:
            counts = df["transmission"].value_counts().reset_index()
            counts.columns = ["transmission", "count"]
            fig = px.bar(counts, x="transmission", y="count", color="transmission",
                         color_discrete_sequence=color_seq)
            fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'transmission' not found in dataset.")

    with c4:
        st.markdown("##### 👤 Owner Distribution")
        if "owner" in df.columns:
            counts = df["owner"].value_counts().reset_index()
            counts.columns = ["owner", "count"]
            fig = px.bar(counts, x="count", y="owner", orientation="h", color="owner",
                         color_discrete_sequence=color_seq)
            fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'owner' not found in dataset.")

    # Row 3: Top Brands & Cars by Year
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("##### 🏆 Top 10 Brands")
        brand_col = "brand" if "brand" in df.columns else ("name" if "name" in df.columns else None)
        if brand_col:
            temp = df.copy()
            if brand_col == "name":
                temp["brand"] = temp["name"].astype(str).str.split().str[0]
                brand_col = "brand"
            counts = temp[brand_col].value_counts().nlargest(10).reset_index()
            counts.columns = ["brand", "count"]
            fig = px.bar(counts, x="count", y="brand", orientation="h", color="count",
                         color_continuous_scale=color_seq)
            fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand information not found in dataset.")

    with c6:
        st.markdown("##### 📅 Cars by Year")
        if "year" in df.columns:
            counts = df["year"].value_counts().sort_index().reset_index()
            counts.columns = ["year", "count"]
            fig = px.line(counts, x="year", y="count", markers=True, color_discrete_sequence=["#ec4899"])
            fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'year' not found in dataset.")

    # Row 4: Cars by Seller Type
    st.markdown("##### 🧑‍💼 Cars by Seller Type")
    if "seller_type" in df.columns:
        counts = df["seller_type"].value_counts().reset_index()
        counts.columns = ["seller_type", "count"]
        fig = px.bar(counts, x="seller_type", y="count", color="seller_type",
                     color_discrete_sequence=color_seq, text="count")
        fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10), height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'seller_type' not found in dataset.")


# ============================================================================
# PAGE: ABOUT
# ============================================================================
def page_about():
    st.markdown('<div class="section-heading">ℹ️ About This Project</div>', unsafe_allow_html=True)

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title" style="font-size:1.15rem;">🚗 AutoWorth AI</div>
            <p class="feature-desc" style="margin-top:0.5rem;">
            AutoWorth AI is a machine learning powered web application designed to predict
            the fair resale value of used cars. It was developed as an academic Machine
            Learning project, applying a full ML pipeline — from data preprocessing and
            label encoding to model training and deployment — inside an interactive,
            production-style Streamlit dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title" style="font-size:1.05rem;">👨‍💻 Developer</div>
            <p class="feature-desc" style="margin-top:0.4rem;">
            <b style="color:#e8ebf5;">Name:</b> Amaan Ali Shaikh<br>
            <b style="color:#e8ebf5;">PRN:</b> 125BTCM2005<br>
            <b style="color:#e8ebf5;">Subject:</b> Machine Learning
            </p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
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
            <div class="glass-card" style="padding:0.8rem 1.1rem; margin-bottom:0.6rem; display:flex; align-items:center; gap:0.8rem;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div>
                    <div style="font-weight:700; color:#f1f2f9; font-size:0.95rem;">{name}</div>
                    <div style="color:#9ca3b8; font-size:0.8rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-box">
        AutoWorth AI © 2026 &nbsp;|&nbsp; Machine Learning Academic Project &nbsp;|&nbsp;
        Developed by Amaan Ali Shaikh (PRN: 125BTCM2005)
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    load_css()

    model, encoders, metadata, error = load_artifacts()
    df, _ = load_dataset()

    page = render_sidebar()

    if error:
        st.error(f"⚠️ Failed to load model artifacts: {error}\n\nPlease ensure `car_price_model.pkl`, "
                  f"`encoders.pkl`, and `metadata.pkl` are present in the `model/` directory.")
        if page not in ["🏠 Home", "ℹ️ About"]:
            return

    if page == "🏠 Home":
        page_home(metadata or {}, df)
    elif page == "🔮 Predict Price":
        if metadata and encoders and model:
            page_predict(model, encoders, metadata)
        else:
            st.warning("Prediction is unavailable because model artifacts could not be loaded.")
    elif page == "📊 Analytics":
        page_analytics(df)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()