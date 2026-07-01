"""Interactive HAR demo (accessibility-focused UI).

Loads the trained Random Forest, 1-D CNN, and standardization stats, then lets you
browse test windows (unseen subjects), view the raw sensor signals, and see each
model's prediction plus the soft-voting ensemble.

Designed with an assistive-care audience in mind: light high-contrast theme, large
legible text, colour-blind-safe plot colours, and status shown with icons + words
(not colour alone).

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
import streamlit as st

from src.data import load_features, load_raw_signals, load_activity_map
from src.model import HAR_CNN

MODELS = ROOT / "models"

# Colour-blind-safe palette (Okabe-Ito) for the X/Y/Z axes.
AXIS_COLORS = ["#0072B2", "#E69F00", "#009E73"]
ACCENT = "#00695C"        # deep teal
ACCENT_LIGHT = "#B2DFDB"  # muted teal for non-predicted bars

st.set_page_config(page_title="HAR Activity Classifier", layout="wide")

# --- accessibility CSS: larger fonts, bigger controls, clear status cards ---
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 19px; }
      h1 { font-size: 2.6rem !important; }
      h2, h3 { font-size: 1.5rem !important; }
      .stButton > button {
          font-size: 1.15rem; font-weight: 600;
          padding: 0.6rem 1.1rem; border-radius: 10px;
      }
      .pred-card {
          padding: 1.1rem 1.3rem; border-radius: 14px; margin: 0.3rem 0 1rem 0;
          font-size: 1.7rem; font-weight: 800; line-height: 1.3;
      }
      .pred-ok   { background:#E6F4EA; color:#1B5E20; border:3px solid #2E7D32; }
      .pred-bad  { background:#FDECEA; color:#B71C1C; border:3px solid #C62828; }
      .info-line { font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "legend.fontsize": 11})


@st.cache_resource
def load_everything():
    id_to_activity = load_activity_map()
    class_names = [id_to_activity[i] for i in range(1, 7)]
    X_feat, y, subjects = load_features("test")
    X_raw = load_raw_signals("test")
    stats = np.load(MODELS / "standardizer.npz")
    mean, std = stats["mean"], stats["std"]
    X_raw_std = (X_raw - mean) / std
    rf = joblib.load(MODELS / "rf.joblib")
    cnn = HAR_CNN()
    cnn.load_state_dict(torch.load(MODELS / "cnn.pt", map_location="cpu"))
    cnn.eval()
    return class_names, X_feat, X_raw, X_raw_std, y, subjects, rf, cnn


def predict(idx, X_feat, X_raw_std, rf, cnn):
    rf_p = rf.predict_proba(X_feat[idx:idx + 1])[0]
    xt = torch.tensor(X_raw_std[idx:idx + 1], dtype=torch.float32).permute(0, 2, 1)
    with torch.no_grad():
        cnn_p = F.softmax(cnn(xt), dim=1).numpy()[0]
    return rf_p, cnn_p, (rf_p + cnn_p) / 2


# ---------------------------------------------------------------- app
st.title("🚶 Activity Recognition")
st.markdown(
    "<p class='info-line'>Recognising everyday movement from a wearable sensor — "
    "a building block for fall detection and elderly-care monitoring. "
    "Predictions are shown for people the model has never seen.</p>",
    unsafe_allow_html=True,
)

if not (MODELS / "rf.joblib").exists():
    st.error("Model files not found. Run the save-artifacts cell in "
             "notebooks/02_modeling.ipynb first.")
    st.stop()

class_names, X_feat, X_raw, X_raw_std, y, subjects, rf, cnn = load_everything()
n = len(y)

st.sidebar.header("Choose a recording")
if "idx" not in st.session_state:
    st.session_state.idx = 0
if st.sidebar.button("🎲  Show a random recording"):
    st.session_state.idx = int(np.random.randint(n))
idx = st.sidebar.slider("Recording number", 0, n - 1, st.session_state.idx)
st.session_state.idx = idx

true_label = class_names[y[idx] - 1]
st.sidebar.markdown(
    f"<p class='info-line'><b>Person #{subjects[idx]}</b><br>"
    f"Actual activity: <b>{true_label.replace('_', ' ').title()}</b></p>",
    unsafe_allow_html=True,
)

rf_p, cnn_p, ens_p = predict(idx, X_feat, X_raw_std, rf, cnn)
pred_label = class_names[int(ens_p.argmax())]
correct = pred_label == true_label

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Sensor signals (2.56-second window)")
    t = np.arange(128) / 50.0
    groups = [("Body acceleration", 0), ("Rotation (gyroscope)", 3),
              ("Total acceleration", 6)]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True)
    for ax, (title, start) in zip(axes, groups):
        for j, axis in enumerate("XYZ"):
            ax.plot(t, X_raw[idx, :, start + j], label=axis,
                    color=AXIS_COLORS[j], linewidth=2.2)
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=3, frameon=False)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("What the system predicts")
    nice_pred = pred_label.replace("_", " ").title()
    if correct:
        st.markdown(
            f"<div class='pred-card pred-ok'>✓ {nice_pred}<br>"
            f"<span style='font-size:1.05rem;font-weight:600'>Correct</span></div>",
            unsafe_allow_html=True)
    else:
        nice_true = true_label.replace("_", " ").title()
        st.markdown(
            f"<div class='pred-card pred-bad'>✗ {nice_pred}<br>"
            f"<span style='font-size:1.05rem;font-weight:600'>"
            f"Actual: {nice_true}</span></div>",
            unsafe_allow_html=True)

    st.markdown("**How confident is it?**")
    order = np.argsort(ens_p)
    labels = [class_names[i].replace("_", " ").title() for i in order]
    colors = [ACCENT if i == ens_p.argmax() else ACCENT_LIGHT for i in order]
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    ax2.barh(labels, ens_p[order], color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown(
        f"<p class='info-line'>The two models combined:<br>"
        f"• Random Forest → <b>{class_names[int(rf_p.argmax())].replace('_',' ').title()}</b> "
        f"({rf_p.max():.0%})<br>"
        f"• Neural network → <b>{class_names[int(cnn_p.argmax())].replace('_',' ').title()}</b> "
        f"({cnn_p.max():.0%})</p>",
        unsafe_allow_html=True)
