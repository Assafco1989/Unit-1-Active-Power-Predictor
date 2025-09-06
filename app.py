# app.py
# Unit-1 Active Power Predictor — Streamlit app using ONNX + scaler.json

import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional ONNX tooling (not required now for Model Info)
try:
    import onnx
    from onnx import numpy_helper
    HAVE_ONNX = True
except Exception:
    HAVE_ONNX = False

# ONNX Runtime for inference
try:
    import onnxruntime as ort
except Exception as e:
    st.error("onnxruntime is not installed. Install with: pip install onnxruntime")
    raise

# ------------------------- Page config & minimal theming -------------------------
st.set_page_config(page_title="Unit-1 Active Power Predictor", layout="wide")

# Small CSS: center header/footer, add top spacing so logo is fully visible
st.markdown("""
<style>
.centered { text-align: center; }
.small-note { color: #777; font-size: 0.9rem; }
.footer {
  position: fixed; left: 0; right: 0; bottom: 8px;
  text-align: center; color: #444; font-weight: 600;
}
.block-container { padding-top: 2.2rem; } /* bumped from 1.2rem to push hero down */
.hero-spacer { height: 22px; }            /* extra spacer above the logo */
</style>
""", unsafe_allow_html=True)

# ------------------------- Paths & helpers -------------------------
APP_DIR = os.getcwd()

def here(fname: str) -> str:
    return os.path.join(APP_DIR, fname)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path="u1_mw.onnx", scaler_path="scaler.json", config_path="config.json"):
    # Load scaler (feature list & z-score params)
    scaler = load_json(here(scaler_path))
    feat_names = scaler["feature_names"]
    mu = np.array(scaler["mean"], dtype=np.float64)
    stdev = np.array(scaler["std"], dtype=np.float64)
    stdev[stdev == 0] = 1.0

    # Load config if present (for hidden sizes, etc.)
    cfg = {}
    try:
        cfg = load_json(here(config_path))
    except Exception:
        pass

    # Load ONNX model for inference
    onnx_path = here(model_path)
    if not os.path.isfile(onnx_path):
        q = here("u1_mw_int8.onnx")
        if os.path.isfile(q):
            onnx_path = q
        else:
            raise FileNotFoundError("No ONNX model found (u1_mw.onnx or u1_mw_int8.onnx).")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return {
        "sess": sess,
        "onnx_path": onnx_path,
        "scaler": scaler,
        "feat_names": feat_names,
        "mu": mu,
        "stdev": stdev,
        "config": cfg,
    }

def standardize(X: np.ndarray, mu: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    s = stdev.copy(); s[s == 0] = 1.0
    return (X - mu) / s

def predict(sess: ort.InferenceSession, X_std_f32: np.ndarray) -> np.ndarray:
    return sess.run(["mw_pred"], {"features": X_std_f32})[0].reshape(-1)

def gauge_figure(value: float, vmin: float = 100.0, vmax: float = 285.0):
    value = float(np.clip(value, vmin, vmax))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': " MW", 'font': {'size': 28}},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [vmin, (vmin+vmax)/2], "color": "#e0f3ff"},
                {"range": [(vmin+vmax)/2, vmax], "color": "#d9ffe0"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": value}
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "U1 Active Power", 'font': {'size': 16}}
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ------------------------- HERO (logo + title + footer) -------------------------
with st.container():
    # Top logo centered with extra spacer so it's fully visible
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="hero-spacer"></div>', unsafe_allow_html=True)
        if os.path.isfile(here("APCO_Logo.png")):
            st.image("APCO_Logo.png", use_container_width=False)
        st.markdown('<h1 class="centered">⚡Unit-1 Active Power Predictor </h1>', unsafe_allow_html=True)
        st.markdown('<div class="centered small-note">Enter features, predict MW, and explore correlations.</div>', unsafe_allow_html=True)

# Footer (fixed)
st.markdown('<div class="footer">Developed by ATTARAT AI TEAM</div>', unsafe_allow_html=True)

# ------------------------- Load artifacts once -------------------------
try:
    art = load_artifacts("u1_mw.onnx", "scaler.json", "config.json")
except Exception as e:
    st.error(f"Failed to load model/scaler/config: {e}")
    st.stop()

feat_names = art["feat_names"]
mu = art["mu"]
stdev = art["stdev"]
sess = art["sess"]
cfg = art["config"] or {}

# ------------------------- Choose 9 UI features -------------------------
# Your requested top 3 first:
top3_requested = ["Steam FLOW (t/h)", "HRH PRESSURE (Mpa)", "HRH TEMP. (C)"]
top3_present = [f for f in top3_requested if f in feat_names]
if len(top3_present) < len(top3_requested):
    missing = [f for f in top3_requested if f not in feat_names]
    st.info(f"Note: The following requested features were not found in scaler.json and will be skipped: {missing}")

# For the remaining slots, use the original order from scaler.json (robust) minus those already chosen
remaining = [f for f in feat_names if f not in top3_present]
ui_feats = (top3_present + remaining)[:9]

# ------------------------- TABS -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["MW Prediction", "Correlation", "Model Info", "Analysis & Diagnostic"])

# === Tab 1: MW Prediction ===
with tab1:
    st.subheader("MW Prediction")
    st.caption("Enter values for the **nine** key features below (others default to their mean from scaler.json).")

    cols = st.columns(3)
    inputs = {}
    for i, fname in enumerate(ui_feats):
        col = cols[i % 3]
        idx = feat_names.index(fname)
        m = float(mu[idx]); s = float(stdev[idx])
        lo = m - 5.0 * (s if s > 1e-6 else 1.0)
        hi = m + 5.0 * (s if s > 1e-6 else 1.0)
        step = 0.001
        inputs[fname] = col.number_input(fname, value=round(m, 3), min_value=float(lo), max_value=float(hi),
                                         step=step, format="%.3f", key=f"inp_{fname}")

    st.write("")  # spacing

    predict_clicked = st.button("🔮 Predict", type="primary")

    if predict_clicked:
        try:
            # Build full feature vector: provided UI values for 9 features; means for the rest
            x = []
            for j, f in enumerate(feat_names):
                if f in inputs:
                    x.append(float(inputs[f]))
                else:
                    x.append(float(mu[j]))
            X = np.array([x], dtype=np.float64)
            Xs = standardize(X, mu, stdev).astype(np.float32)
            pred = float(predict(sess, Xs)[0])

            st.success(f"**U1 Active Power = {pred:,.3f} MW**")
            st.plotly_chart(gauge_figure(pred, vmin=100.0, vmax=285.0),
                            use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# === Tab 2: Correlation ===
with tab2:
    st.subheader("Correlation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pearson Correlation Matrix**")
        if os.path.isfile(here("Pearson_Correlation.png")):
            st.image("Pearson_Correlation.png", use_container_width=True)
        else:
            st.info("Pearson_Correlation.png not found.")
    with c2:
        st.markdown("**Spearman Correlation Matrix**")
        if os.path.isfile(here("Spearman_Correlation.png")):
            st.image("Spearman_Correlation.png", use_container_width=True)
        else:
            st.info("Spearman_Correlation.png not found.")

    st.write("")
    st.markdown('<div class="centered"><b>Correlation with target U1 Active Power (MW)</b></div>', unsafe_allow_html=True)
    if os.path.isfile(here("Correlation_with_target.png")):
        cc1, cc2, cc3 = st.columns([1, 2, 1])
        with cc2:
            st.image("Correlation_with_target.png", use_container_width=True)
    else:
        st.info("Correlation_with_target.png not found.")

# === Tab 3: Model Info ===
with tab3:
    st.subheader("Model Info")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**General**")
        st.write(f"**Model file:** `{os.path.basename(art['onnx_path'])}`")
        st.write(f"**Algorithm:** Neural Network (MLP) exported to ONNX")
        st.write(f"**Features used (total):** {len(feat_names)}")
        st.write(f"**Hidden layers:** {cfg.get('hidden', 'N/A')}")
        st.write(f"**Dropout:** {cfg.get('dropout', 'N/A')}")
        st.write(f"**Learning rate:** {cfg.get('lr', 'N/A')}")
        st.write(f"**Weight decay:** {cfg.get('weight_decay', 'N/A')}")
        if "target" in cfg:
            st.write(f"**Target:** {cfg['target']}")
        if "min_target" in cfg:
            st.write(f"**Min target filter:** {cfg['min_target']}")

    with colB:
        st.markdown("**Notes**")
        st.write("- Inputs are standardized using `scaler.json` (z-score).")
        st.write("- ONNX Runtime is used for fast, portable inference.")
        st.write("- For batch prediction or CSV/XLSX inputs, we can add a tab later if needed.")

    # Removed: features list & feature-importance table (as requested)

# === Tab 4: Analysis ===
with tab4:
    st.subheader("Analysis & Diagnostic")
    st.markdown('<div class="centered"><i>Will be add as a next stage</i></div>', unsafe_allow_html=True)
