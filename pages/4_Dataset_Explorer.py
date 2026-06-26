import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import load_raw_tensor

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Dataset Explorer")

st.write(
"""
Explore any WiFi CSI MAT file and inspect its signal,
statistics, and individual frames.
"""
)

st.divider()

# ==========================================================
# Upload
# ==========================================================

uploaded_file = st.file_uploader(

    "Upload MAT File",

    type=["mat"]

)

if uploaded_file is None:

    st.info("Upload a MAT file to begin.")

    st.stop()

tmp = tempfile.NamedTemporaryFile(

    delete=False,

    suffix=".mat"

)

tmp.write(uploaded_file.read())

tmp.close()

tensor = load_raw_tensor(tmp.name)

height, width, frames = tensor.shape
# ==========================================================
# Dataset Statistics
# ==========================================================

st.subheader("Dataset Statistics")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Height", height)

with c2:
    st.metric("Width", width)

with c3:
    st.metric("Frames", frames)

with c4:
    st.metric("Total Values", tensor.size)

st.divider()
# ==========================================================
# Signal Statistics
# ==========================================================

st.subheader("Signal Statistics")

left, right = st.columns(2)

with left:

    st.metric("Minimum", f"{tensor.min():.4f}")

    st.metric("Mean", f"{tensor.mean():.4f}")

with right:

    st.metric("Maximum", f"{tensor.max():.4f}")

    st.metric("Std", f"{tensor.std():.4f}")
# ==========================================================
# Frame Viewer
# ==========================================================

st.divider()

st.subheader("Frame Viewer")

frame = st.slider(

    "Select Frame",

    0,

    frames-1,

    0

)

current = tensor[:, :, frame]

fig = go.Figure()

fig.add_trace(

    go.Heatmap(

        z=current,

        colorscale="Viridis"

    )

)

fig.update_layout(

    template="plotly_dark",

    title=f"Frame {frame+1}",

    height=600

)

st.plotly_chart(

    fig,

    use_container_width=True
)
# ==========================================================
# Histogram
# ==========================================================

st.divider()

st.subheader("Signal Histogram")

hist = go.Figure()

hist.add_trace(

    go.Histogram(

        x=tensor.flatten(),

        nbinsx=50

    )

)

hist.update_layout(

    template="plotly_dark",

    xaxis_title="Normalized Signal",

    yaxis_title="Frequency",

    height=450

)

st.plotly_chart(

    hist,

    use_container_width=True
)
# ==========================================================
# Current Frame Information
# ==========================================================

st.divider()

st.subheader("Current Frame")

y, x = np.unravel_index(

    np.argmax(current),

    current.shape

)

col1, col2 = st.columns(2)

with col1:

    st.metric(

        "Maximum Signal",

        f"{current.max():.4f}"

    )

    st.metric(

        "Minimum Signal",

        f"{current.min():.4f}"

    )

with col2:

    st.metric(

        "Peak Position",

        f"({y}, {x})"

    )

    st.metric(

        "Average",

        f"{current.mean():.4f}"

    )
# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "WaveSense AI • Dataset Explorer"
)