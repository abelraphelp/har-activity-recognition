"""Interactive HAR dashboard (Streamlit).

A multi-panel dashboard over the trained models:
- Live demo: pick a test recording (unseen subject), see the raw signals and the
  ensemble's live prediction with confidence.
- Model results: headline metrics, model comparison, per-class recall.
- Robustness: Leave-One-Subject-Out summary, PAMAP2 per-subject, significance tests.
- Key findings.

Accessibility-focused (light high-contrast theme, large text, colour-blind-safe plots).
Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd                 # pandas before torch (Windows import-order rule)
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import torch
import torch.nn.functional as F

from src.data import load_features, load_raw_signals, load_activity_map
from src.model import HAR_CNN

MODELS_DIR = ROOT / "models"

# Colour-blind-safe palette (Okabe-Ito) + model colours
AXIS_COLORS = ["#0072B2", "#E69F00", "#009E73"]
C_RF, C_CNN, C_LSTM, C_ENS = "#E69F00", "#0072B2", "#8B98AD", "#009E73"

# ---- pre-computed results (from the notebooks) ----
MODEL_NAMES = ["Random Forest", "CNN", "LSTM", "Ensemble"]
ACC = [0.9287, 0.9450, 0.8945, 0.9491]
F1  = [0.927, 0.9449, 0.8947, 0.950]
ACT = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
REC_RF  = [0.976, 0.915, 0.857, 0.886, 0.921, 1.000]
REC_CNN = [0.978, 0.962, 0.974, 0.823, 0.921, 0.950]
REC_ENS = [0.992, 0.975, 0.993, 0.833, 0.940, 0.968]
PAMAP_SUB = ["101", "102", "103", "104", "105", "106", "107", "108", "109"]
PAMAP_ACC = [0.788, 0.738, 0.810, 0.782, 0.863, 0.877, 0.818, 0.298, 0.833]

st.set_page_config(page_title="HAR Dashboard", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 17px; }
      h1 { font-size: 2.3rem !important; }
      .pred-card { padding: 1.0rem 1.2rem; border-radius: 14px; margin: 0.2rem 0 1rem 0;
                   font-size: 1.6rem; font-weight: 800; line-height: 1.3; }
      .pred-ok  { background:#E6F4EA; color:#1B5E20; border:3px solid #2E7D32; }
      .pred-bad { background:#FDECEA; color:#B71C1C; border:3px solid #C62828; }
      .info-line { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})


@st.cache_resource
def load_everything():
    id_to_activity = load_activity_map()
    class_names = [id_to_activity[i] for i in range(1, 7)]
    X_feat, y, subjects = load_features("test")
    X_raw = load_raw_signals("test")
    stats = np.load(MODELS_DIR / "standardizer.npz")
    mean, std = stats["mean"], stats["std"]
    X_raw_std = (X_raw - mean) / std
    rf = joblib.load(MODELS_DIR / "rf.joblib")
    cnn = HAR_CNN()
    cnn.load_state_dict(torch.load(MODELS_DIR / "cnn.pt", map_location="cpu"))
    cnn.eval()
    return class_names, X_feat, X_raw, X_raw_std, y, subjects, rf, cnn


def predict(idx, X_feat, X_raw_std, rf, cnn):
    rf_p = rf.predict_proba(X_feat[idx:idx + 1])[0]
    xt = torch.tensor(X_raw_std[idx:idx + 1], dtype=torch.float32).permute(0, 2, 1)
    with torch.no_grad():
        cnn_p = F.softmax(cnn(xt), dim=1).numpy()[0]
    return rf_p, cnn_p, (rf_p + cnn_p) / 2


# ---------------------------------------------------------------- header
st.title("Human Activity Recognition — Dashboard")
st.markdown(
    "<p class='info-line'>Recognising physical activity from wearable sensors, evaluated on "
    "people the models never saw. A building block for fall detection and elderly-care "
    "monitoring.</p>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best accuracy", "94.9%", "ensemble · unseen subjects")
k2.metric("Best macro-F1", "0.950")
k3.metric("Models compared", "4")
k4.metric("Subjects evaluated", "39", "UCI HAR 30 + PAMAP2 9")

demo_ready = (MODELS_DIR / "rf.joblib").exists()
tab_demo, tab_models, tab_robust, tab_find = st.tabs(
    ["Live demo", "Model results", "Robustness", "Key findings"])

# ---------------------------------------------------------------- live demo
with tab_demo:
    if not demo_ready:
        st.warning("Model files not found. Run the save-artifacts cell in "
                   "notebooks/02_modeling.ipynb, then reload.")
    else:
        class_names, X_feat, X_raw, X_raw_std, y, subjects, rf, cnn = load_everything()
        n = len(y)
        if "idx" not in st.session_state:
            st.session_state.idx = 0

        cA, cB = st.columns([1, 3])
        with cA:
            if st.button("Show a random recording", use_container_width=True):
                st.session_state.idx = int(np.random.randint(n))
        with cB:
            st.session_state.idx = st.slider("Recording number", 0, n - 1,
                                             st.session_state.idx)
        idx = st.session_state.idx

        true_label = class_names[y[idx] - 1]
        rf_p, cnn_p, ens_p = predict(idx, X_feat, X_raw_std, rf, cnn)
        pred_label = class_names[int(ens_p.argmax())]
        correct = pred_label == true_label

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.caption(f"Person #{subjects[idx]} · actual activity: "
                       f"**{true_label.replace('_', ' ').title()}**")
            t = np.arange(128) / 50.0
            fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            for ax, (title, s0) in zip(axes, [("Body acceleration", 0),
                                              ("Rotation (gyroscope)", 3),
                                              ("Total acceleration", 6)]):
                for j, a in enumerate("XYZ"):
                    ax.plot(t, X_raw[idx, :, s0 + j], label=a,
                            color=AXIS_COLORS[j], linewidth=2)
                ax.set_title(title); ax.legend(loc="upper right", ncol=3, frameon=False)
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("Time (seconds)")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with right:
            nice = pred_label.replace("_", " ").title()
            if correct:
                st.markdown(f"<div class='pred-card pred-ok'>&#10003; {nice}<br>"
                            f"<span style='font-size:1rem;font-weight:600'>Correct</span></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='pred-card pred-bad'>&#10007; {nice}<br>"
                            f"<span style='font-size:1rem;font-weight:600'>"
                            f"Actual: {true_label.replace('_',' ').title()}</span></div>",
                            unsafe_allow_html=True)
            st.markdown("**How confident is it?**")
            proba = pd.DataFrame({"probability": ens_p},
                                 index=[c.replace('_', ' ').title() for c in class_names])
            st.bar_chart(proba, horizontal=True, color=C_ENS)
            st.caption(
                f"Random Forest → {class_names[int(rf_p.argmax())].replace('_',' ').title()} "
                f"({rf_p.max():.0%})  ·  "
                f"Neural network → {class_names[int(cnn_p.argmax())].replace('_',' ').title()} "
                f"({cnn_p.max():.0%})")

# ---------------------------------------------------------------- model results
with tab_models:
    st.subheader("Model comparison (UCI HAR test set)")
    comp = pd.DataFrame({"Accuracy": ACC, "Macro-F1": F1}, index=MODEL_NAMES)
    st.bar_chart(comp, stack=False)

    st.subheader("Per-class recall")
    st.caption("RF is a posture specialist, the CNN a motion specialist — complementary.")
    rec = pd.DataFrame({"Random Forest": REC_RF, "CNN": REC_CNN, "Ensemble": REC_ENS},
                       index=ACT)
    st.bar_chart(rec, stack=False, color=[C_RF, C_CNN, C_ENS])
    st.dataframe(rec.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------------- robustness
with tab_robust:
    st.subheader("Leave-One-Subject-Out cross-validation (UCI HAR, 30 subjects)")
    loso = pd.DataFrame({
        "LOSO accuracy": ["0.940 ± 0.076", "0.936 ± 0.057", "0.910 ± 0.093"],
        "Worst subject": [0.73, 0.79, 0.66],
    }, index=["CNN", "Random Forest", "LSTM"])
    st.table(loso)

    st.subheader("Are the models really different? (significance tests)")
    sig = pd.DataFrame({
        "Test": ["McNemar (fixed split)", "Wilcoxon (LOSO)", "Wilcoxon (LOSO)"],
        "p-value": [0.0002, 0.60, 0.0045],
        "Verdict": ["CNN > RF", "tie", "CNN > LSTM"],
    }, index=["CNN vs RF", "CNN vs RF", "CNN vs LSTM"])
    st.table(sig)
    st.caption("A 'significant' fixed-split result (p=0.0002) dissolves into a tie under "
               "subject-level CV (p=0.60) — single splits can mislead.")

    st.subheader("PAMAP2 — per-subject generalization (harder dataset)")
    st.caption("~0.81 for typical subjects, but subject 108 collapses to 0.30 — a domain shift.")
    pamap = pd.DataFrame({"accuracy": PAMAP_ACC}, index=PAMAP_SUB)
    st.bar_chart(pamap, color=C_CNN)

# ---------------------------------------------------------------- findings
with tab_find:
    st.subheader("Key findings")
    c1, c2 = st.columns(2)
    with c1:
        st.success("**Complementary specialists** — the CNN fixed the baseline's worst "
                   "class (downstairs recall 0.86 → 0.97); the RF kept static postures "
                   "cleaner. The ensemble beats either alone.")
        st.info("**Per-subject variance is the real challenge** — on PAMAP2 most subjects "
                "reach ~0.81 but one collapses to 0.30. Aggregate accuracy hides it.")
        st.warning("**Single splits can lie** — a McNemar-significant CNN>RF result became "
                   "a tie under Leave-One-Subject-Out. Only subject-level CV is trustworthy.")
    with c2:
        st.info("**A sensor limit, not a bug** — sitting vs standing are motionless upright "
                "postures with near-identical gravity signatures; no model exceeds ~0.89 recall.")
        st.warning("**Transitions confirmed** — windows straddling an activity boundary score "
                   "0.075 (≈ chance) vs 0.837 for clean windows, across both datasets.")
        st.success("**Know when to stop tuning** — RF tuning gained nothing (within CV noise); "
                   "CNN tuning gained ~1 point. Neural nets are far more tuning-sensitive.")

st.divider()
st.caption("Built by Abel Raphel Pulikottil · RF + PyTorch CNN/LSTM + ensemble · "
           "UCI HAR & PAMAP2 · github.com/abelraphelp/har-activity-recognition")
