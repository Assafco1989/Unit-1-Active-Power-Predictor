# app.py
# Unified App — Gross MW Predictor + Corrected SRC (Compute) + Correlation + Model Info
# -------------------------------------------------------------------------------------
# Tabs (in order):
#   1) Gross MW Predictor   — ONNX model-based MW prediction (your original Tab 1, retitled)
#   2) Compute (Corrected SRC) — Tab-1 from SRC.py (Kt, Kgf, Kpf; auto loading; Net/Gross basis)
#   3) Correlation
#   4) Model Info
#
# Notes:
# - SRC helpers (Kt/Kgf/Kpf/interp/extrap) are embedded below to avoid importing SRC.py
#   (SRC.py contains its own Streamlit app at top level).

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
st.set_page_config(page_title="Unit-1 Gross MW — Predictor", layout="wide")
st.markdown("""
<style>
.centered { text-align: center; }
.small-note { color: #777; font-size: 0.9rem; }
.footer {
  position: fixed; left: 0; right: 0; bottom: 8px;
  text-align: center; color: #444; font-weight: 600;
}
.block-container { padding-top: 2.2rem; }
.hero-spacer { height: 22px; }
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
    scaler = load_json(here(scaler_path))
    feat_names = scaler["feature_names"]
    mu = np.array(scaler["mean"], dtype=np.float64)
    stdev = np.array(scaler["std"], dtype=np.float64)
    stdev[stdev == 0] = 1.0

    cfg = {}
    try:
        cfg = load_json(here(config_path))
    except Exception:
        pass

    onnx_path = here(model_path)
    if not os.path.isfile(onnx_path):
        q = here("u1_mw_int8.onnx")
        if os.path.isfile(q):
            onnx_path = q
        else:
            raise FileNotFoundError("No ONNX model found (u1_mw.onnx or u1_mw_int8.onnx).")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return {"sess": sess, "onnx_path": onnx_path, "scaler": scaler,
            "feat_names": feat_names, "mu": mu, "stdev": stdev, "config": cfg}

def standardize(X: np.ndarray, mu: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    s = stdev.copy(); s[s == 0] = 1.0
    return (X - mu) / s

def predict(sess: ort.InferenceSession, X_std_f32: np.ndarray) -> np.ndarray:
    return sess.run(["mw_pred"], {"features": X_std_f32})[0].reshape(-1)

def gauge_figure(value: float, title: str = "U1 Gross Power"):
    vmin, vmax = 100.0, 285.0
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
        title={'text': title, 'font': {'size': 16}}
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ------------------------- HERO (logo + title + footer) -------------------------
with st.container():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="hero-spacer"></div>', unsafe_allow_html=True)
        if os.path.isfile(here("APCO_Logo.png")):
            st.image("APCO_Logo.png", use_container_width=False)
        st.markdown('<h1 class="centered">⚡ Unit-1 Gross MW — Predictor & SRC Calculator</h1>', unsafe_allow_html=True)
        st.markdown('<div class="centered small-note">Predict Gross MW, compute Corrected SRC, review correlations & model info.</div>', unsafe_allow_html=True)

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

# =========================
# SRC helpers (from SRC.py)
# =========================
def kt_from_T(T: float):
    if T < 30.0:
        A, B, C, D = -0.000003, -0.0021, -0.0336, 4.4614
        pct = A*(T**3) + B*(T**2) + C*T + D
        return 1.0 + pct/100.0, "T < 30°C (cubic)"
    else:
        A, B, C = -0.016, 0.8099, -8.4259
        pct = A*(T**2) + B*T + C
        return 1.0 + pct/100.0, "30–40°C (quadratic)"

def kgf_from_Hz(Hz: float) -> float:
    F = Hz - 50.0
    A, B, C = 0.0091, -0.0565, -0.00000009
    pct = A*(F**2) + B*F + C
    return 1.0 + pct/100.0

def kpf_at_band(PF: float, band: int) -> float:
    if band == 25:   A, B, C = 0.0002, 0.1460, -0.1243
    elif band == 50: A, B, C = 0.0030, 0.3933, -0.3364
    elif band == 75: A, B, C = 0.0058, 0.7518, -0.6433
    else:            A, B, C = 0.0176, 1.2424, -1.0688
    pct = A*(PF**2) + B*PF + C
    return 1.0 + pct/100.0

def kpf_from_pf_loadpct(PF: float, LoadPct: float):
    K25 = kpf_at_band(PF, 25)
    K50 = kpf_at_band(PF, 50)
    K75 = kpf_at_band(PF, 75)
    K100 = kpf_at_band(PF, 100)
    if LoadPct < 25.0:
        return K25, {"mode":"clamped_below_25"}
    if LoadPct < 50.0:
        w = (LoadPct - 25.0) / 25.0
        return K25 + w*(K50 - K25), {"mode":"interpolate","low_anchor":25,"high_anchor":50,"w":float(w)}
    if LoadPct < 75.0:
        w = (LoadPct - 50.0) / 25.0
        return K50 + w*(K75 - K50), {"mode":"interpolate","low_anchor":50,"high_anchor":75,"w":float(w)}
    if LoadPct <= 100.0:
        w = (LoadPct - 75.0) / 25.0
        return K75 + w*(K100 - K75), {"mode":"interpolate","low_anchor":75,"high_anchor":100,"w":float(w)}
    slope = (K100 - K75) / 25.0
    return K100 + slope*(LoadPct - 100.0), {"mode":"extrapolate_gt_100","slope_per_pct":float(slope)}

def nearest_loading_band(load_pct: float) -> int:
    candidates = np.array([25, 50, 75, 100], dtype=float)
    return int(candidates[np.argmin(np.abs(candidates - load_pct))])

def compute_all(MW: float, T: float, Hz: float, PF: float, LoadPct: float):
    Kt, _ = kt_from_T(T)
    Kgf = kgf_from_Hz(Hz)
    Kpf, _ = kpf_from_pf_loadpct(PF, LoadPct)
    Kc = Kt * Kgf * Kpf
    SRC = MW / Kc if Kc != 0 else np.nan
    return {"Kt": Kt, "Kgf": Kgf, "Kpf": Kpf, "Kc": Kc, "SRC": SRC}

# ------------------------- Choose 9 UI features (as before) -------------------------
top3_requested = ["Steam FLOW (t/h)", "HRH PRESSURE (Mpa)", "HRH TEMP. (C)"]
top3_present = [f for f in top3_requested if f in feat_names]
remaining = [f for f in feat_names if f not in top3_present]
ui_feats = (top3_present + remaining)[:9]
missing = [f for f in top3_requested if f not in feat_names]
if missing:
    st.info(f"Note: The following requested features were not found in scaler.json and will be skipped: {missing}")

# ======================
# Tabs (new order/labels)
# ======================
tab1, tab2, tab3, tab4 = st.tabs(["Gross MW Predictor", "Compute (Corrected SRC)", "Correlation", "Model Info"])

# === Tab 1: Gross MW Predictor (retitled) ===
with tab1:
    st.subheader("Gross MW Predictor")
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

    st.write("")
    if st.button("🔮 Predict Gross MW", type="primary"):
        try:
            x = [float(inputs.get(f, mu[j])) for j, f in enumerate(feat_names)]
            X = np.array([x], dtype=np.float64)
            Xs = standardize(X, mu, stdev).astype(np.float32)
            pred = float(predict(sess, Xs)[0])
            st.success(f"**U1 Gross Power = {pred:,.3f} MW**")
            st.plotly_chart(gauge_figure(pred, title="U1 Gross Power"),
                            use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# === Tab 2: Compute (Corrected SRC) — from SRC.py Tab-1 ===
with tab2:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Inputs (SRC)")
        power_basis = st.radio("Power basis", ["Net MW", "Gross MW"], horizontal=True, key="src_basis")
        if power_basis == "Net MW":
            denom = 235.0
            mw_label = "Net MW"
            mw_default = 235.0
            mw_key = "mw_net"
        else:
            denom = 277.0
            mw_label = "Gross MW"
            mw_default = 277.0
            mw_key = "mw_gross"

        MW = st.number_input(mw_label, min_value=0.0, max_value=5000.0, value=mw_default, step=0.1, key=mw_key)
        T  = st.number_input("Ambient temp (°C)", min_value=-50.0, max_value=100.0, value=30.0, step=0.1, key="src_T")
        Hz = st.number_input("Grid frequency (Hz)", min_value=45.0, max_value=65.0, value=50.0, step=0.01, format="%.2f", key="src_Hz")
        PF = st.number_input("Power factor (0–1)", min_value=0.0, max_value=1.0, value=0.85, step=0.001, format="%.3f", key="src_PF")

        LoadPct_auto = 100.0 * (MW / denom) if denom > 0 else 0.0
        st.metric(f"Auto Load % ({mw_label} / {int(denom)})", f"{LoadPct_auto:.2f}%")
        st.caption(f"Nearest nominal band (reference): {nearest_loading_band(LoadPct_auto)}%")
        LoadPct_used = LoadPct_auto

        compute_btn = st.button("Compute SRC", type="primary", key="src_compute_btn")

    with right:
        st.subheader("Results")
        if compute_btn:
            Kt, regime = kt_from_T(T)
            if T > 40.0:
                st.warning("T > 40°C: extrapolating Kt with the 30–40°C formula per spec.")
            Kgf = kgf_from_Hz(Hz)
            Kpf, _ = kpf_from_pf_loadpct(PF, LoadPct_used)

            Kc = Kt * Kgf * Kpf
            SRC = MW / Kc if Kc != 0 else np.nan

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total Correction Factor, Kc", f"{Kc:.6f}")
            with m2:
                st.metric("Corrected SRC", f"{SRC:.3f} MW")

            st.divider()
            st.markdown("### Factor breakdown")
            F_signed = Hz - 50.0
            st.write(f"**Kt (Ambient)** = {Kt:.6f}  \\\\ Regime: **{regime}**")
            st.write(f"**Kgf (Frequency)** = {Kgf:.6f}  \\\\ Using F = Hz - 50 = {F_signed:+.3f}")
            st.write(f"**Kpf (PF)** = {Kpf:.6f}  \\\\ Load % used: **{LoadPct_used:.2f}%**; PF = {PF:.3f}")
            st.caption(f"Basis: **{mw_label}**, denominator = {int(denom)}")

            st.session_state["baseline_inputs"] = dict(NetMW=MW, T=T, Hz=Hz, PF=PF, LoadPct=LoadPct_used)
        else:
            st.info("Enter inputs and click **Compute SRC**.")

# === Tab 3: Correlation (unchanged) ===
with tab3:
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

# === Tab 4: Model Info (unchanged) ===
with tab4:
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
        if "target" in cfg: st.write(f"**Target:** {cfg['target']}")
        if "min_target" in cfg: st.write(f"**Min target filter:** {cfg['min_target']}")
    with colB:
        st.markdown("**Notes**")
        st.write("- Inputs are standardized using `scaler.json` (z-score).")
        st.write("- ONNX Runtime is used for fast, portable inference.")
        st.write("- For batch prediction or CSV/XLSX inputs, a new tab can be added later.")
