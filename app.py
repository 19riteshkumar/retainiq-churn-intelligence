"""
Enterprise Churn Prediction & Autonomous AI Retention Agent
FILE: app.py  -  Streamlit dashboard (dual-mode data ingestion + Gemini AI agent)

Run:  streamlit run app.py     (after running  python train.py  once)
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None
    genai_client = None
    try:
        import google.generativeai as genai_legacy  # fallback for older installs
    except Exception:
        genai_legacy = None
else:
    genai_legacy = None
    genai_client = None

BASE_DIR = Path(__file__).resolve().parent
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
RAW_DEFAULTS = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
}
BINARY_LOOKUP = {
    "yes": 1, "no": 0, "male": 1, "female": 0,
    "1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0,
}
RISK_COLORS = {"High Risk": "#d62728", "Medium Risk": "#ff9f1c", "Low Risk": "#2ca02c"}
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-1.5-flash"]

st.set_page_config(
    page_title="Churn Prediction & AI Retention Agent",
    page_icon="📉",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17252f;
        --muted: #65747d;
        --saffron: #e8752a;
        --teal: #087f8c;
        --mint: #e8f5f2;
        --line: #d9e5e3;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fbfa 0%, #eef7f5 52%, #fff8f1 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #17343a;
        border-right: 1px solid #285159;
    }
    [data-testid="stSidebar"] * { color: #edf8f5 !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] > div { background: #24515a; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--line);
        border-top: 4px solid var(--teal);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(23, 52, 58, 0.07);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(23, 52, 58, 0.06);
    }
    .hero {
        background: linear-gradient(110deg, #17343a 0%, #087f8c 68%, #e8752a 100%);
        border-radius: 18px;
        padding: 30px 34px;
        margin: 10px 0 26px;
        color: white;
        box-shadow: 0 14px 32px rgba(23, 52, 58, 0.18);
    }
    .hero h1 { color: white !important; margin: 0 0 8px; font-size: 2.15rem; }
    .hero p { margin: 0; color: #dff5f1; font-size: 1.05rem; }
    .hero-tag {
        display: inline-block;
        margin-top: 18px;
        padding: 6px 12px;
        border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px;
        color: #fff4df;
        font-size: .82rem;
        font-weight: 600;
    }
    .section-label {
        color: var(--saffron);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: -8px;
    }
    .stButton > button {
        border-radius: 9px;
        border: 1px solid #b7d7d2;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_core_artifacts():
    model = joblib.load(BASE_DIR / "model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")
    return model, scaler, columns


@st.cache_resource(show_spinner="Loading validation batch...")
def load_sample_batch():
    return joblib.load(BASE_DIR / "sample_test_batch.pkl")


def artifacts_ready() -> bool:
    needed = ["model.pkl", "scaler.pkl", "columns.pkl", "sample_test_batch.pkl"]
    return all((BASE_DIR / f).exists() for f in needed)


# ----------------------------------------------------------------------
# Helper: risk segmentation
# ----------------------------------------------------------------------
def get_risk_tier(prob: float) -> str:
    """High Risk (>= 70%), Medium Risk (40% - 69%), Low Risk (< 40%)."""
    if prob >= 0.70:
        return "High Risk"
    if prob >= 0.40:
        return "Medium Risk"
    return "Low Risk"


# ----------------------------------------------------------------------
# Preprocessing for raw (Telco-style) data -> model-ready features
# ----------------------------------------------------------------------
def _encode_binary(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(BINARY_LOOKUP)
    return mapped.fillna(0).astype(int)


def preprocess_raw(df_in: pd.DataFrame, feature_cols, scaler):
    """
    Cleans raw customer data, imputes missing columns/values, one-hot encodes,
    aligns to the training schema (missing dummy columns filled with 0) and
    scales numeric columns with the saved scaler.

    Returns (X_scaled, customer_ids, cleaned_raw_df, report_dict)
    """
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reset_index(drop=True)
    report = {"missing_columns": [], "imputed_values": 0, "ignored_columns": []}

    # Customer id
    id_col = next((c for c in df.columns if c.lower() == "customerid"), None)
    if id_col is not None:
        ids = df[id_col].astype(str)
    else:
        ids = pd.Series([f"CUST-{i + 1:05d}" for i in range(len(df))])
        report["missing_columns"].append("customerID (auto-generated)")

    # Numeric columns
    for col in ["tenure", "MonthlyCharges"]:
        if col not in df.columns:
            df[col] = RAW_DEFAULTS[col]
            report["missing_columns"].append(col)
        df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")
        n_na = int(df[col].isna().sum())
        if n_na:
            median = df[col].median()
            df[col] = df[col].fillna(RAW_DEFAULTS[col] if pd.isna(median) else median)
            report["imputed_values"] += n_na

    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
        report["missing_columns"].append("TotalCharges (derived: tenure x MonthlyCharges)")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce"
    )
    n_na = int(df["TotalCharges"].isna().sum())
    if n_na:
        median = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(
            median if not pd.isna(median) else df["tenure"] * df["MonthlyCharges"]
        )
        report["imputed_values"] += n_na

    # SeniorCitizen is numeric in the training data; the other binary fields
    # remain categorical so get_dummies reproduces the training schema.
    if "SeniorCitizen" not in df.columns:
        df["SeniorCitizen"] = RAW_DEFAULTS["SeniorCitizen"]
        report["missing_columns"].append("SeniorCitizen")
    df["SeniorCitizen"] = _encode_binary(df["SeniorCitizen"])

    for col in BINARY_COLS:
        if col not in df.columns:
            df[col] = RAW_DEFAULTS[col]
            report["missing_columns"].append(col)
        df[col] = df[col].fillna(RAW_DEFAULTS[col]).astype(str).str.strip()

    # Categorical columns
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = RAW_DEFAULTS[col]
            report["missing_columns"].append(col)
        n_na = int(df[col].isna().sum())
        if n_na:
            df[col] = df[col].fillna(RAW_DEFAULTS[col])
            report["imputed_values"] += n_na
        df[col] = df[col].astype(str).str.strip()

    # Match train.py: one-hot encode all string columns and drop the first level.
    model_frame = df[["SeniorCitizen"] + BINARY_COLS + NUMERIC_COLS + CAT_COLS]
    X = pd.get_dummies(model_frame, drop_first=True, dtype=int)
    X = X.reindex(columns=list(feature_cols), fill_value=0).astype(float)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    known = set(RAW_DEFAULTS) | {"TotalCharges", "Churn"} | ({id_col} if id_col else set())
    report["ignored_columns"] = [c for c in df_in.columns if str(c).strip() not in known]
    return X, ids, df, report


def score_dataframe(model, X: pd.DataFrame, ids) -> pd.DataFrame:
    probs = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "CustomerID": list(ids),
        "Churn_Probability": np.round(probs, 4),
    })
    out["Risk_Tier"] = out["Churn_Probability"].apply(get_risk_tier)
    return out


# ----------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------
def render_kpis(scored: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(scored):,}")
    c2.metric("🔴 High Risk", f"{(scored['Risk_Tier'] == 'High Risk').sum():,}")
    c3.metric("🟠 Medium Risk", f"{(scored['Risk_Tier'] == 'Medium Risk').sum():,}")
    c4.metric("🟢 Low Risk", f"{(scored['Risk_Tier'] == 'Low Risk').sum():,}")


def plot_risk_distribution(scored: pd.DataFrame):
    order = ["High Risk", "Medium Risk", "Low Risk"]
    counts = scored["Risk_Tier"].value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(order, counts.values, color=[RISK_COLORS[t] for t in order])
    ax.set_title("Customer Risk Distribution", fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


def plot_feature_importance(model, columns, top_n: int = 10):
    imp = pd.Series(model.feature_importances_, index=list(columns)).sort_values(ascending=False).head(top_n)
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color="#1f77b4")
    ax.set_title(f"Top {top_n} Drivers of Churn (XGBoost Importance)", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render_scored_table(scored: pd.DataFrame, key: str) -> None:
    tiers = st.multiselect(
        "Filter by risk tier",
        ["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"],
        key=f"filter_{key}",
    )
    view = scored[scored["Risk_Tier"].isin(tiers)].sort_values("Churn_Probability", ascending=False)
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "Churn_Probability": st.column_config.ProgressColumn(
                "Churn_Probability", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.caption(f"Showing {len(view):,} of {len(scored):,} customers")


def risk_badge(prob: float) -> None:
    tier = get_risk_tier(prob)
    color = RISK_COLORS[tier]
    st.markdown(
        f"<div style='display:inline-block;padding:8px 20px;border-radius:24px;"
        f"background:{color};color:white;font-size:1.2rem;font-weight:700;'>"
        f"{tier} &nbsp;|&nbsp; {prob:.1%} churn probability</div>",
        unsafe_allow_html=True,
    )
    st.progress(float(min(max(prob, 0.0), 1.0)))


# ----------------------------------------------------------------------
# Gemini AI helpers
# ----------------------------------------------------------------------
def call_gemini(api_key: str, prompt: str):
    """Returns (text, error_message) using the available Google Gemini SDK."""
    if genai is None and genai_legacy is None:
        return None, "The `google-genai` package is not installed. Run: pip install google-genai"
    if not api_key:
        return None, "No Gemini API key provided. Enter it in the sidebar to enable AI generation."

    last_error = "Unknown error"
    for model_name in GEMINI_MODELS:
        try:
            if genai is not None:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
            else:
                genai_legacy.configure(api_key=api_key)
                response = genai_legacy.GenerativeModel(model_name).generate_content(prompt)
                text = getattr(response, "text", None)
            if text:
                return text, None
            last_error = f"{model_name} returned an empty or blocked response."
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
    return None, f"Gemini request failed. {last_error}"


def build_segment_summary(raw_df: pd.DataFrame, probs: np.ndarray) -> dict:
    df = raw_df.copy().reset_index(drop=True)
    df["prob"] = np.asarray(probs)
    high = df[df["prob"] >= 0.70]
    stats = {
        "total_customers": int(len(df)),
        "high_risk_customers": int(len(high)),
        "high_risk_share_pct": round(100 * len(high) / max(len(df), 1), 1),
    }
    if len(high):
        stats["avg_tenure_months"] = round(float(high["tenure"].mean()), 1)
        stats["avg_monthly_charges"] = round(float(high["MonthlyCharges"].mean()), 2)
        stats["monthly_revenue_at_risk"] = round(float(high["MonthlyCharges"].sum()), 2)
        for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling"]:
            if col in high.columns:
                stats[f"{col}_mix_pct"] = (
                    (high[col].value_counts(normalize=True) * 100).round(1).head(4).to_dict()
                )
    return stats


def generate_batch_summary(api_key: str, stats: dict):
    prompt = f"""You are a senior customer-retention strategist writing for the executive team of a telecom company.
Below are aggregate statistics for the HIGH-RISK churn segment (churn probability >= 70%) produced by our ML model:

{stats}

Write an executive summary with these sections (use Markdown headers, be concise and specific):
1. Headline Finding (2 sentences)
2. Who Is at Risk (customer profile drawn from the statistics)
3. Revenue Exposure (use the monthly revenue at risk figure and annualize it)
4. Top 3 Recommended Retention Actions (each with expected impact and owner)
5. 30-Day Action Plan

Only use the numbers provided; do not invent statistics."""
    return call_gemini(api_key, prompt)


def generate_playbook(api_key: str, customer: dict, prob: float):
    tier = get_risk_tier(prob)
    prompt = f"""You are an autonomous AI Retention Agent for a telecom provider.
A machine-learning model has scored this customer.

CUSTOMER PROFILE:
{customer}

CHURN PROBABILITY: {prob:.1%}
RISK TIER: {tier}

Produce a retention playbook with exactly these three sections, using Markdown headers:

## 1. Root Cause Analysis
Explain, in 3-5 bullet points, why this customer is likely leaving, referencing specific profile attributes
(contract type, tenure, pricing, support, payment method, billing style, etc.).

## 2. Retention Offer
Recommend a targeted offer (discounts, contract-upgrade incentives, service bundles, free support/security add-ons).
Include concrete terms (percentage or amount, duration) and a short justification of expected impact vs. cost.

## 3. Personalized Outreach Email Draft
Write a ready-to-send email with a Subject line, greeting, body, clear call to action and signature
from "Customer Success Team". Warm, human tone, under 180 words. Use the placeholder [Customer Name] for the name.

Do not mention the churn probability or the ML model in the customer-facing email."""
    return call_gemini(api_key, prompt)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
api_key = st.sidebar.text_input(
    "GEMINI_API_KEY",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get a key at https://aistudio.google.com/app/apikey",
)
st.sidebar.caption(
    "ℹ️ ML predictions work fully offline. The key is only used for AI Retention generation."
)
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Data Ingestion Mode",
    [
        "Mode 1: Internal Demo (Pre-loaded Validation Batch)",
        "Mode 2: External CSV File Ingestion",
        "Mode 3: Single Customer Simulator & AI Agent",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Model: XGBoost + SMOTE  |  AI: Google Gemini")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📉 Churn Intelligence Hub</h1>
        <p>Turn customer signals into timely retention action with explainable ML and AI-assisted playbooks.</p>
        <span class="hero-tag">INDIA TELCO DEMO • XGBOOST + GEMINI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_ready():
    st.error(
        "Model artifacts not found. Run `python train.py` in this folder first to create "
        "`model.pkl`, `scaler.pkl`, `columns.pkl` and `sample_test_batch.pkl`."
    )
    st.stop()

model, scaler, feature_columns = load_core_artifacts()

# ======================================================================
# MODE 1: Internal Demo
# ======================================================================
if mode.startswith("Mode 1"):
    st.header("Internal Demo: Pre-loaded Validation Batch")
    batch = load_sample_batch()
    customer_ids = batch.get("CustomerID", batch.get("customer_ids"))
    X_batch = batch["X_test"].copy()
    if list(X_batch.columns) != list(feature_columns):
        X_batch = X_batch.reindex(columns=list(feature_columns), fill_value=0)
    scored = score_dataframe(model, X_batch, customer_ids)

    render_kpis(scored)
    st.markdown("")

    raw_demo = batch.get("X_test_raw")
    if isinstance(raw_demo, pd.DataFrame):
        st.subheader("Indian Customer Data Preview")
        preview_cols = [
            "customerID", "City", "State", "tenure", "MonthlyCharges",
            "InternetService", "Contract", "PaymentMethod", "TechSupport",
        ]
        preview = raw_demo[[col for col in preview_cols if col in raw_demo.columns]].copy()
        preview = preview.rename(columns={"customerID": "CustomerID"})
        st.dataframe(preview.head(20), hide_index=True, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = plot_risk_distribution(scored)
        st.pyplot(fig1)
        plt.close(fig1)
    with col_b:
        fig2 = plot_feature_importance(model, feature_columns)
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Customer Risk Table")
    render_scored_table(scored, key="m1")

    st.subheader("🤖 AI Batch Summary")
    if api_key:
        if st.button("Generate Executive Summary of High-Risk Segment", key="m1_ai"):
            raw = batch.get("X_test_raw")
            if raw is None:
                raw = pd.DataFrame({
                    "tenure": np.nan,
                    "MonthlyCharges": np.nan,
                    "Contract": "Month-to-month",
                    "InternetService": "Fiber optic",
                    "PaymentMethod": "Electronic check",
                    "TechSupport": "No",
                    "PaperlessBilling": "Yes",
                })
            stats = build_segment_summary(raw, scored["Churn_Probability"].values)
            with st.spinner("Gemini is analysing the high-risk segment..."):
                text, err = generate_batch_summary(api_key, stats)
            if err:
                st.error(err)
            else:
                st.markdown(text)
    else:
        st.info("Enter your Gemini API key in the sidebar to unlock the AI executive summary.")

# ======================================================================
# MODE 2: External CSV
# ======================================================================
elif mode.startswith("Mode 2"):
    st.header("External CSV File Ingestion")
    st.write(
        "Upload any unseen customer CSV. Columns are validated against the training schema; "
        "missing columns are imputed and missing dummy features are filled with 0."
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is None:
        st.info("Awaiting a CSV file. Tip: use the same layout as the Telco Customer Churn dataset.")
    else:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file as CSV: {exc}")
            st.stop()

        if raw_df.empty:
            st.error("The uploaded file contains no rows.")
            st.stop()

        try:
            X_ext, ids, clean_df, report = preprocess_raw(raw_df, feature_columns, scaler)
            scored = score_dataframe(model, X_ext, ids)
        except Exception as exc:
            st.error(f"Failed to process the uploaded data: {exc}")
            st.stop()

        st.success(f"Processed {len(raw_df):,} rows and {raw_df.shape[1]} columns.")
        with st.expander("Schema validation report", expanded=bool(report["missing_columns"])):
            if report["missing_columns"]:
                st.warning("Missing columns were auto-filled: " + ", ".join(report["missing_columns"]))
            else:
                st.write("✅ All expected columns are present.")
            st.write(f"Missing values imputed: **{report['imputed_values']}**")
            if report["ignored_columns"]:
                st.write("Ignored (not used by the model): " + ", ".join(map(str, report["ignored_columns"])))

        render_kpis(scored)
        st.markdown("")

        col_a, col_b = st.columns(2)
        with col_a:
            fig1 = plot_risk_distribution(scored)
            st.pyplot(fig1)
            plt.close(fig1)
        with col_b:
            fig2 = plot_feature_importance(model, feature_columns)
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Scored Customers")
        render_scored_table(scored, key="m2")

        export_df = raw_df.reset_index(drop=True).copy()
        export_df["Churn_Probability"] = scored["Churn_Probability"].values
        export_df["Risk_Tier"] = scored["Risk_Tier"].values
        st.download_button(
            "⬇️ Download Scored CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="scored_customers.csv",
            mime="text/csv",
        )

        st.subheader("🤖 AI Batch Summary")
        if api_key:
            if st.button("Generate Executive Summary of High-Risk Segment", key="m2_ai"):
                stats = build_segment_summary(clean_df, scored["Churn_Probability"].values)
                with st.spinner("Gemini is analysing the high-risk segment..."):
                    text, err = generate_batch_summary(api_key, stats)
                if err:
                    st.error(err)
                else:
                    st.markdown(text)
        else:
            st.info("Enter your Gemini API key in the sidebar to unlock the AI executive summary.")

# ======================================================================
# MODE 3: Single customer simulator + AI agent
# ======================================================================
else:
    st.header("Single Customer Simulator & Autonomous AI Retention Agent")

    left, right = st.columns(2)
    with left:
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0, step=0.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with right:
        tech = st.selectbox("Tech Support", ["No", "Yes"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with st.expander("Additional demographics (optional)"):
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    def build_customer() -> dict:
        record = dict(RAW_DEFAULTS)
        record.update({
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": round(tenure * monthly, 2),
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": tech,
            "PaymentMethod": payment,
            "PaperlessBilling": paperless,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
        })
        if internet == "No":
            for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies"]:
                record[col] = "No internet service"
        return record

    def run_prediction() -> dict:
        record = build_customer()
        X_one, _, _, _ = preprocess_raw(pd.DataFrame([record]), feature_columns, scaler)
        prob = float(model.predict_proba(X_one)[0, 1])
        return {"record": record, "prob": prob}

    b1, b2 = st.columns(2)
    predict_clicked = b1.button("🔮 Predict Churn Risk", type="primary")
    playbook_clicked = b2.button("🤖 Generate AI Retention Playbook (Gemini)")

    if predict_clicked or playbook_clicked:
        st.session_state["sim_result"] = run_prediction()
        st.session_state["sim_playbook"] = None

    result = st.session_state.get("sim_result")
    if result is None:
        st.info("Set the customer parameters above, then click **Predict Churn Risk**.")
    else:
        st.subheader("Churn Risk Assessment")
        risk_badge(result["prob"])

        if playbook_clicked:
            if not api_key:
                st.warning(
                    "No Gemini API key provided. The churn prediction above works offline; "
                    "enter your key in the sidebar to generate the AI retention playbook."
                )
            else:
                summary_fields = {
                    k: result["record"][k]
                    for k in ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
                              "TechSupport", "PaymentMethod", "PaperlessBilling", "SeniorCitizen",
                              "Partner", "Dependents"]
                }
                with st.spinner("Gemini is building the retention playbook..."):
                    text, err = generate_playbook(api_key, summary_fields, result["prob"])
                if err:
                    st.error(err)
                else:
                    st.session_state["sim_playbook"] = text

        if st.session_state.get("sim_playbook"):
            st.markdown("---")
            st.subheader("AI Retention Playbook")
            st.markdown(st.session_state["sim_playbook"])
