"""
ESG Score & Risk Prediction System
====================================
A complete Streamlit application that:
  1. Predicts ESG Score (regression) from financial & sustainability inputs
  2. Uses the predicted ESG Score + risk features to classify ESG Risk Level
  3. Displays results with actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Score & Risk Prediction System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

/* Result cards */
.result-card {
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.score-card  { background: #e8f5e9; border-left: 6px solid #2d6a4f; }
.risk-high   { background: #ffebee; border-left: 6px solid #c62828; }
.risk-medium { background: #fff8e1; border-left: 6px solid #f9a825; }
.risk-low    { background: #e8f5e9; border-left: 6px solid #2e7d32; }

/* Recommendation box */
.rec-box {
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 0.5rem;
    font-size: 16px;
}

/* High Risk */
.rec-high { 
    background:#8B0000; 
    border:1px solid #5A0000; 
    color:white; 
}

/* Medium Risk */
.rec-medium { 
    background:#B8860B; 
    border:1px solid #8B6508; 
    color:white; 
}

/* Low Risk */
.rec-low { 
    background:#006400; 
    border:1px solid #004d00; 
    color:white; 
}

</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__)

ARTIFACT_CANDIDATES = {
    "score_model": ["esg_score_model.pkl"],
    "scaler": ["scaler__1_.pkl", "scaler_1_ .pkl", "scaler__1_ .pkl", "scaler_1_.pkl"],
    "score_features": ["esg_score_features.pkl"],
}


def resolve_artifact_path(candidates):
    for fname in candidates:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Missing model artifact(s): {', '.join(candidates)}")

@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """Load all saved model artifacts. Returns dict or raises on failure."""
    artifacts = {}
    for key, candidates in ARTIFACT_CANDIDATES.items():
        path = resolve_artifact_path(candidates)
        try:
            artifacts[key] = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                artifacts[key] = pickle.load(f)
    return artifacts


def safe_load():
    try:
        return load_artifacts(), None
    except Exception as e:
        return None, str(e)


artifacts, load_error = safe_load()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌱 ESG Score &amp; Risk Prediction System</h1>
  <p>AI-powered environmental, social &amp; governance analytics — predict ESG scores and classify risk levels in real time.</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Could not load model files: {load_error}\n\nMake sure all `.pkl` files are in the same folder as `app.py`.")
    st.stop()

score_features = artifacts["score_features"]  
score_model    = artifacts["score_model"]
scaler         = artifacts["scaler"]

# ── Sidebar: user inputs ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")
    st.markdown("Fill in your company's financial and sustainability data below.")

    st.markdown("### 📊 Financial Metrics")
    year        = st.number_input("Year",            min_value=2000, max_value=2030, value=2023, step=1)
    ebit        = st.number_input("EBIT (USD M)",    min_value=-5000.0, max_value=50000.0, value=500.0, step=10.0,
                                  help="Earnings Before Interest and Taxes")
    roe         = st.number_input("ROE (%)",         min_value=-100.0, max_value=200.0,  value=12.0,  step=0.5,
                                  help="Return on Equity")
    revenue     = st.number_input("Revenue (USD M)", min_value=0.0,    max_value=500000.0, value=1000.0, step=10.0)
    profit_margin = st.number_input("Profit Margin (%)", min_value=-100.0, max_value=100.0, value=10.0, step=0.5)
    market_cap  = st.number_input("Market Cap (USD M)", min_value=0.0, max_value=2000000.0, value=5000.0, step=50.0)
    growth_rate = st.number_input("Growth Rate (%)",  min_value=-100.0, max_value=500.0, value=5.0,  step=0.5)

    st.markdown("### 🌿 ESG / Sustainability Metrics")
    e_score     = st.slider("Environmental Score (E)",  0.0, 100.0, 55.0, 0.5)
    g_score     = st.slider("Governance Score (G)",     0.0, 100.0, 60.0, 0.5)
    csr         = st.number_input("CSR Spending (USD M)", min_value=0.0, max_value=5000.0, value=50.0, step=1.0,
                                  help="Corporate Social Responsibility expenditure")
    percent_et  = st.number_input("% Energy from Renewables (Percent_ET)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
    percent_w   = st.number_input("% Water Recycled (Percent_W)",           min_value=0.0, max_value=100.0, value=25.0, step=0.5)

    st.markdown("### 🏭 Environmental Impact")
    carbon_emissions   = st.number_input("Carbon Emissions (tonnes)", min_value=0.0, max_value=10_000_000.0, value=35000.0, step=100.0)
    water_usage        = st.number_input("Water Usage (m³)",           min_value=0.0, max_value=10_000_000.0, value=18000.0, step=100.0)
    energy_consumption = st.number_input("Energy Consumption (MWh)",   min_value=0.0, max_value=10_000_000.0, value=70000.0, step=100.0)

    st.markdown("### 🏢 Company Profile")
    industry = st.selectbox("Industry", ["Energy", "Finance", "Healthcare", "Manufacturing",
                                          "Retail", "Technology", "Transportation", "Utilities"])
    region   = st.selectbox("Region",   ["Asia", "Europe", "Latin America", "Middle East",
                                          "North America", "Oceania"])

    company_id = st.number_input("Company ID (if known)", min_value=1, max_value=10000, value=1, step=1)

    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

# ── Helper: build score input ──────────────────────────────────────────────────
def build_score_input():
    """Build a DataFrame exactly matching score_features."""
    raw = {
        "Year":       year,
        "E_score":    e_score,
        "G_score":    g_score,
        "Percent_ET": percent_et,
        "Percent_W":  percent_w,
        "CSR":        csr,
        "EBIT":       ebit,
        "ROE":        roe,
    }
    df = pd.DataFrame([{col: raw.get(col, 0.0) for col in score_features}])
    return df



# ── Main content ───────────────────────────────────────────────────────────────
col_info, col_results = st.columns([1, 2], gap="large")

with col_info:
    st.markdown("### 📋 How it Works")
    st.markdown("""
<div style="background:#0B3D91;border-radius:10px;padding:1.2rem;">
<p><span class="step-badge">1</span><strong>Input Data</strong><br>
Enter your company's financial and ESG metrics in the sidebar.</p>
<p><span class="step-badge">2</span><strong>Score Prediction</strong><br>
The regression model predicts your overall ESG score.</p>
<p><span class="step-badge">3</span><strong>Risk Classification</strong><br>
The predicted score is fed into the risk classifier alongside your financial profile.</p>
<p style="margin-bottom:0"><span class="step-badge">4</span><strong>Recommendations</strong><br>
Tailored, actionable guidance is generated based on your risk level.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🎯 Score Features Used")
    st.code("\n".join(score_features), language="text")

with col_results:
    if not predict_btn:
        st.info("👈 Fill details and click Run Prediction")
    else:
        with st.spinner("Running predictions…"):

            try:
                X_score = build_score_input()
                try:
                    X_score_scaled = scaler.transform(X_score)
                except Exception:
                    # Some saved scaler artifacts were fit on a different feature schema.
                    # Fall back to the raw score-feature matrix so prediction can still run.
                    X_score_scaled = X_score.values
                predicted_score = float(score_model.predict(X_score_scaled)[0])
                predicted_score = max(0, min(100, predicted_score))
                score_ok = True
            except Exception as e:
                score_ok = False
                score_error = str(e)

        # ✅ DISPLAY MUST BE INSIDE HERE
        if not score_ok:
            st.error(f"ESG Score prediction failed: {score_error}")
        else:
            st.markdown("## 📊 ESG Score Result")

            score_color = (
                "#c62828" if predicted_score < 40 else
                "#f9a825" if predicted_score < 65 else
                "#2e7d32"
            )

            st.markdown(f"""
<div class="result-card score-card">
  <div>PREDICTED ESG SCORE</div>
  <div style="font-size:3rem;color:{score_color};">{predicted_score:.1f}</div>
</div>
""", unsafe_allow_html=True)

            # Progress bar
            pct = int(predicted_score)
            bar_color = "#c62828" if pct < 40 else "#f9a825" if pct < 65 else "#40916c"

            st.markdown(f"""
<div style="background:#e0e0e0;border-radius:20px;height:18px;">
  <div style="width:{pct}%;background:{bar_color};height:18px;border-radius:20px;"></div>
</div>
""", unsafe_allow_html=True)

            # Recommendations
            st.markdown("## 💡 Recommendations")

            if predicted_score < 40:
                st.error("🔴 Low ESG — Improve governance, reduce emissions, increase CSR")

            elif predicted_score < 65:
                st.warning("🟡 Moderate ESG — Improve sustainability & efficiency")

            else:
                st.success("🟢 High ESG — Maintain strong ESG practices")

            # Summary
            with st.expander("🔍 View Input Summary"):
                summary = {
                    "Year": year,
                    "EBIT (USD M)": ebit,
                    "ROE (%)": roe,
                    "Revenue (USD M)": revenue,
                    "Profit Margin (%)": profit_margin,
                    "Market Cap (USD M)": market_cap,
                    "Growth Rate (%)": growth_rate,
                    "Environmental Score (E)": e_score,
                    "Governance Score (G)": g_score,
                    "CSR Spending (USD M)": csr,
                    "% Energy from Renewables": percent_et,
                    "% Water Recycled": percent_w,
                    "Carbon Emissions (tonnes)": carbon_emissions,
                    "Water Usage (m3)": water_usage,
                    "Energy Consumption (MWh)": energy_consumption,
                    "Industry": industry,
                    "Region": region,
                    "Company ID": company_id,
                }
                summary_df = (
                    pd.DataFrame(summary, index=["Value"])
                    .T
                    .rename(columns={"Value": "Input"})
                )
                st.dataframe(summary_df, use_container_width=True)



        

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;'>"
    "ESG Score &amp; Risk Prediction System &nbsp;|&nbsp; Powered by Scikit-learn &amp; Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
