# =======================Import Functions=====================
import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from scipy.fftpack import dct
import shap
import matplotlib.pyplot as plt
import ollama
# import re
from scipy.fftpack import dct
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="EEG-based ADHD Screening with XAI",
    layout="wide"
)

# ===================== PATHS =====================
BASE = Path("results")
MODEL_PATH = BASE / "best_model.joblib"
CCO_SELECTOR_PATH = BASE / "selected_features_cco.joblib"
PREPROC_PATH = BASE / "preprocessing_config.json"
FEATURE_NAMES_PATH = BASE / "feature_names.json"

# ===================== LOAD ARTIFACTS =====================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(CCO_SELECTOR_PATH)
    config = json.load(open(PREPROC_PATH))
    feature_names = json.load(open(FEATURE_NAMES_PATH))
    return model, selector, config, feature_names


model, cco_selector, config, feature_names = load_artifacts()


# ===================== LLaMA PROMPT =====================
LLAMA_XAI_PROMPT = """
You are an Explainable AI assistant designed for an EEG-based ADHD screening system.

CONTEXT:
You are given the output of a machine learning model that analyzes EEG-derived features
to detect patterns that may be associated with ADHD. Your role is to explain the model's
output clearly, responsibly, and without making medical claims.

The explanation must be suitable for two audiences:
1) Healthcare professionals
2) Parents or caregivers

INPUT:
- Prediction label: {prediction}  (0 = Non-ADHD pattern, 1 = ADHD-indicative pattern)
- Model confidence score: {confidence}
- Important EEG features (with SHAP contributions): {features}

TASK:
Generate THREE clearly separated sections using the exact headings below.

=== CLINICIAN EXPLANATION ===
- Explain about the condition of child in simple language.
- Write 2 to 3 sentences
- Use technical but neutral language
- Reference ONLY the EEG features provided
- Describe findings as model-detected patterns, not diagnosis
- Mention the confidence score

=== PARENT-FRIENDLY EXPLANATION ===
- Write 3 to 4 bullet points
- Use simple, reassuring language
- Avoid medical jargon (or explain briefly)
- Clearly state this is a screening result, not a diagnosis

=== MEDICAL DISCLAIMER ===
- One short paragraph
- State clearly that this is NOT a diagnostic tool
- Emphasize the need for professional clinical evaluation
- Mention limitations of AI-based screening systems

CRITICAL CONSTRAINTS:
- Do NOT hallucinate or invent information
- Do NOT diagnose or confirm ADHD
- Do NOT suggest treatments or interventions
- Do NOT reference EEG channels or features not provided
- Do NOT add external medical facts or statistics

IMPORTANT:
- Output plain text only
- Use the exact section headings as provided
- Do NOT include markdown or JSON
"""


# ===================== PREPROCESSING =====================
def preprocess_eeg(df):
    """
    Reproduce training feature extraction:
    DCT + Time-domain + PSD band features
    Output shape: (n_epochs, 812)
    """


    # -----------------------------
    # Channel selection (Ch4–Ch17)
    # -----------------------------
    channels = [f"Ch{i}" for i in range(4, 18)]
    df = df[channels]
    data = df.values

    # -----------------------------
    # Epoching
    # -----------------------------
    epoch_len = config["epoch_length"]   # e.g. 256
    sfreq = config["sampling_rate"]      # 128
    n_epochs = data.shape[0] // epoch_len
    data = data[:n_epochs * epoch_len]
    epochs = data.reshape(n_epochs, epoch_len, len(channels))

    # ======================================================
    #  DCT FEATURES (same as notebook)
    # ======================================================
    n_coef = config["dct_coefficients"]  # e.g. 40
    dct_feats = []

    for ep in epochs:
        coeffs = dct(ep, axis=0, norm="ortho")[:n_coef, :]
        dct_feats.append(coeffs.flatten())

    dct_feats = np.array(dct_feats)

    # ======================================================
    #  TIME-DOMAIN FEATURES (same list as training)
    # ======================================================
    time_feats = []

    for ep in epochs:
        row = []
        for ch in range(ep.shape[1]):
            x = ep[:, ch]

            row.extend([
                np.mean(x),
                np.var(x),
                np.sqrt(np.mean(x**2)),     # RMS
                np.min(x),
                np.max(x),
                np.ptp(x),
                skew(x),
                kurtosis(x),
                np.mean(x[:-1] * x[1:] < 0),  # ZCR
                np.var(x),                  # Hjorth activity
                np.sqrt(np.var(np.diff(x)) / np.var(x)),  # mobility
                np.sqrt(
                    np.var(np.diff(np.diff(x))) /
                    np.var(np.diff(x))
                )                            # complexity
            ])
        time_feats.append(row)

    time_feats = np.array(time_feats)

    # ======================================================
    # 3️⃣ PSD BAND FEATURES (same as training)
    # ======================================================
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    psd_feats = []

    for ep in epochs:
        row = []
        for ch in range(ep.shape[1]):
            f, Pxx = welch(ep[:, ch], fs=sfreq, nperseg=sfreq*2)

            for (lo, hi) in bands.values():
                mask = (f >= lo) & (f < hi)
                row.append(np.trapz(Pxx[mask], f[mask]))

            # spectral entropy
            P = Pxx / np.sum(Pxx)
            row.append(-np.sum(P * np.log2(P + 1e-12)))

        psd_feats.append(row)

    psd_feats = np.array(psd_feats)

    # ======================================================
    # CONCATENATE (ORDER MATTERS!)
    # ======================================================
    X = np.hstack([dct_feats, time_feats, psd_feats])

    # -----------------------------
    # HARD ASSERT (IMPORTANT)
    # -----------------------------
    if X.shape[1] != 812:
        raise ValueError(f"Feature mismatch: expected 812, got {X.shape[1]}")

    return X





# ===================== UI =====================
st.title(" EEG-based ADHD Screening with Explainable AI")

st.markdown("""
This application demonstrates a **research prototype** for EEG-based ADHD screening  
using Machine Learning, SHAP, and a locally hosted LLaMA 3 (LLM) model.

 **This is NOT a medical diagnostic tool.**
""")

uploaded_file = st.file_uploader("Upload EEG CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("EEG file uploaded successfully")

    if st.button("Run Screening"):
        with st.spinner("Processing EEG data..."):
            X = preprocess_eeg(df)


            # Feature selection
            # X_sel = X[:, selected_idx]
            
            # Safety: ensure indices are within bounds
            selected_idx = cco_selector["selected_indices"]

            X_sel = X[:, selected_idx]   # (epochs, 181)


            # Aggregate across epochs
            X_mean = X_sel.mean(axis=0).reshape(1, -1)

            selected_names = [feature_names[i] for i in selected_idx]

            # Prediction
            pred = int(model.predict(X_mean)[0])
            prob = float(model.predict_proba(X_mean)[0].max())

        label = "ADHD-indicative pattern" if pred == 1 else "Non-ADHD pattern"

        st.subheader(" Screening Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Model Confidence:** {prob:.3f}")

        # ===================== SHAP =====================
        st.subheader("🔍 Feature Attribution (SHAP)")

        # Extract classifier from pipeline
        rf_model = model.named_steps["clf"]

        explainer = shap.TreeExplainer(rf_model)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_mean)

        

        if isinstance(shap_values, list):
            shap_vec = shap_values[1][0]   # shape (181,)
        else:
            shap_vec = shap_values[0][:, pred] if shap_values.ndim == 3 else shap_values[:, pred]

        # Build pandas Series (NOW 1D)
        shap_series = pd.Series(shap_vec, index=selected_names)

        top_shap = shap_series.abs().sort_values(ascending=False).head(10)

        # Plot
        fig, ax = plt.subplots()
        top_shap.sort_values().plot.barh(ax=ax)
        ax.set_title("Top SHAP Feature Contributions")
        ax.set_xlabel("SHAP value (impact)")
        st.pyplot(fig)




        # ===================== LLaMA XAI =====================
        st.subheader("Explaination by LLaMA-3")

        # Build feature payload safely (already correct)
        feature_payload = [
            {"channel": i, "contribution": float(v)}
            for i, v in top_shap.items()
        ]

        prompt = LLAMA_XAI_PROMPT.format(
            prediction=pred,
            confidence=round(prob, 3),
            features=feature_payload
        )

        with st.spinner("Generating explanation..."):
            response = ollama.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.6}
            )

        llama_text = response["message"]["content"]

        st.markdown(llama_text)