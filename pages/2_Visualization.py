import os
import tempfile
import time

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import load_raw_tensor

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Visualization",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 WiFi Signal Visualization")

st.write(
"""
Visualize the WiFi CSI signal frame-by-frame and observe
how the strongest reflection moves during the gesture.
"""
)

st.divider()

# ==========================================================
# Upload MAT
# ==========================================================

uploaded_file = st.file_uploader(

    "Upload MAT File",

    type=["mat"]

)

if uploaded_file is None:

    st.info("Upload a MAT file.")

    st.stop()

tmp = tempfile.NamedTemporaryFile(

    delete=False,

    suffix=".mat"

)

tmp.write(uploaded_file.read())

tmp.close()

mat_path = tmp.name

# ==========================================================
# Load Tensor
# ==========================================================

try:

    tensor = load_raw_tensor(mat_path)

except Exception as e:

    st.error(str(e))

    st.stop()

frames = tensor.shape[2]

height = tensor.shape[0]

width = tensor.shape[1]

st.success("Signal loaded successfully.")
# ==========================================================
# Playback Controls
# ==========================================================

left, right = st.columns(2)

with left:

    speed = st.slider(

        "Animation Speed (ms)",

        50,

        1000,

        250,

        50

    )

with right:

    autoplay = st.checkbox(

        "Play Automatically",

        value=True

    )

frame_slider = st.slider(

    "Frame",

    0,

    frames-1,

    0

)

st.divider()
# ==========================================================
# Heatmap
# ==========================================================

heatmap_placeholder = st.empty()

trajectory_placeholder = st.empty()

info_placeholder = st.empty()

trajectory_x = []

trajectory_y = []
# ==========================================================
# Animation
# ==========================================================

def draw_frame(frame):

    current = tensor[:, :, frame]

    fig = go.Figure()

    fig.add_trace(

        go.Heatmap(

            z=current,

            colorscale="Viridis"

        )

    )

    fig.update_layout(

        title=f"Frame {frame+1}/{frames}",

        template="plotly_dark",

        height=600

    )

    heatmap_placeholder.plotly_chart(

        fig,

        use_container_width=True

    )

    y, x = np.unravel_index(

        np.argmax(current),

        current.shape

    )

    trajectory_x.append(x)

    trajectory_y.append(y)

    fig2 = go.Figure()

    fig2.add_trace(

        go.Scatter(

            x=trajectory_x,

            y=trajectory_y,

            mode="lines+markers"

        )

    )

    fig2.update_layout(

        template="plotly_dark",

        title="Motion Trajectory",

        yaxis=dict(

            autorange="reversed"

        ),

        height=500

    )

    trajectory_placeholder.plotly_chart(

        fig2,

        use_container_width=True

    )

    info_placeholder.info(

        f"""

Current Frame : {frame+1}

Maximum Signal : {current.max():.4f}

Minimum Signal : {current.min():.4f}

Mean Signal : {current.mean():.4f}

"""
    )
# ==========================================================
# Playback
# ==========================================================

if autoplay:

    for frame in range(frames):

        draw_frame(frame)

        time.sleep(speed/1000)

else:

    draw_frame(frame_slider)