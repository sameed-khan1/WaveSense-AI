import os
import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="WaveSense AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Paths
# ==========================================================

MODEL_PATH = os.path.join(
    "outputs",
    "wavesense_best.pth"
)

OUTPUT_DIR = "outputs"

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main{
        background-color:#0E1117;
    }

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
    }

    .title{
        text-align:center;
        font-size:42px;
        font-weight:bold;
        color:#4FC3F7;
    }

    .subtitle{
        text-align:center;
        font-size:20px;
        color:white;
    }

    .card{
        background:#1E1E1E;
        padding:20px;
        border-radius:15px;
        border:1px solid #333333;
        margin-bottom:20px;
    }

    .metric{
        text-align:center;
        font-size:26px;
        color:#00E5FF;
        font-weight:bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=90
    )

    st.title("WaveSense AI")

    st.markdown("---")

    st.success("CNN + GRU")

    st.write("WiFi Gesture Recognition")

    st.markdown("---")

    st.subheader("Navigation")

    st.info(
        """
Dashboard

Visualization

Model Performance

Dataset Explorer

About Model
"""
    )

    st.markdown("---")

    st.subheader("Model Status")

    if os.path.exists(MODEL_PATH):

        st.success("Model Loaded")

    else:

        st.error("No trained model found")

    st.markdown("---")

    st.caption("Version 1.0")

# ==========================================================
# Main Page
# ==========================================================

st.markdown(
    "<div class='title'>WaveSense AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>WiFi-Based Human Gesture Recognition using Deep Learning</div>",
    unsafe_allow_html=True
)

st.write("")

# ==========================================================
# Top Metrics
# ==========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
<div class='card'>
<div class='metric'>CNN</div>
<center>Feature Extraction</center>
</div>
""",
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
<div class='card'>
<div class='metric'>GRU</div>
<center>Temporal Learning</center>
</div>
""",
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
<div class='card'>
<div class='metric'>10</div>
<center>Gesture Classes</center>
</div>
""",
        unsafe_allow_html=True
    )

st.write("")

# ==========================================================
# About
# ==========================================================

st.header("Project Overview")

st.write(
"""
WaveSense AI is a deep learning system for recognizing human
gestures using WiFi Channel State Information (CSI).

The application uses a hybrid CNN + GRU architecture to learn
both spatial and temporal information from WiFi signals.

The trained model predicts one of ten gesture classes and
provides confidence scores along with animated signal
visualizations.
"""
)

st.write("")

# ==========================================================
# Features
# ==========================================================

st.header("Features")

left, right = st.columns(2)

with left:

    st.success("Upload WiFi MAT files")

    st.success("Gesture Prediction")

    st.success("Confidence Score")

    st.success("Probability Distribution")

    st.success("Animated WiFi Heatmap")

with right:

    st.success("Motion Trajectory")

    st.success("Confusion Matrix")

    st.success("Accuracy Curves")

    st.success("Training Report")

    st.success("Dataset Explorer")

st.write("")

# ==========================================================
# Workflow
# ==========================================================

st.header("Deep Learning Pipeline")

st.code(
"""
WiFi CSI Data
      │
      ▼
Preprocessing
      │
      ▼
CNN
      │
      ▼
GRU
      │
      ▼
Fully Connected Layer
      │
      ▼
Gesture Prediction
      │
      ▼
Visualization Dashboard
"""
)

st.write("")

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "WaveSense AI • Built with PyTorch and Streamlit"
)