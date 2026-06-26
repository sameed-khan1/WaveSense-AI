import os
import tempfile
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go

from model import WidarNet
from dataset import WidarDataset
from utils import preprocess_mat

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_PATH = r"D:\LinkedIn Projects\New folder\BVP"

MODEL_PATH = os.path.join(
    "outputs",
    "wavesense_best.pth"
)

# ==========================================================
# CSS
# ==========================================================

st.markdown(
"""
<style>

.metric-card{

    background:#1f2937;

    padding:18px;

    border-radius:15px;

    text-align:center;

    border:1px solid #333;

}

.big{

    font-size:32px;

    font-weight:bold;

    color:#00E5FF;

}

.small{

    color:white;

    font-size:18px;

}

</style>
""",
unsafe_allow_html=True
)

# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_resource
def load_dataset():

    return WidarDataset(DATASET_PATH)

dataset = load_dataset()

num_classes = len(dataset.label_map)

inverse_map = dataset.inverse_label_map

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_model():

    model = WidarNet(
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    return model

model = load_model()

# ==========================================================
# Header
# ==========================================================

st.title("📊 WaveSense AI Dashboard")

st.write(
"""
Upload a WiFi CSI MAT file and let the trained CNN + GRU
model recognize the gesture.
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

    st.info("Please upload a MAT file.")

    st.stop()

# ==========================================================
# Save Uploaded File
# ==========================================================

tmp = tempfile.NamedTemporaryFile(

    delete=False,

    suffix=".mat"

)

tmp.write(uploaded_file.read())

tmp.close()

mat_path = tmp.name

# ==========================================================
# Preprocess
# ==========================================================

try:

    x = preprocess_mat(mat_path)

except Exception as e:

    st.error(str(e))

    st.stop()

x = x.to(DEVICE)

st.success("File loaded successfully.")
# ==========================================================
# Prediction
# ==========================================================

with st.spinner("Running AI prediction..."):

    with torch.no_grad():

        outputs = model(x)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

predicted_class = prediction.item()

gesture = inverse_map[predicted_class]

confidence = confidence.item() * 100

# ==========================================================
# Prediction Cards
# ==========================================================

st.divider()

st.subheader("Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
<div class="metric-card">
<div class="big">{gesture}</div>
<div class="small">Predicted Gesture</div>
</div>
""",
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
<div class="metric-card">
<div class="big">{confidence:.2f}%</div>
<div class="small">Confidence</div>
</div>
""",
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
<div class="metric-card">
<div class="big">{num_classes}</div>
<div class="small">Total Classes</div>
</div>
""",
        unsafe_allow_html=True
    )

# ==========================================================
# Probability Distribution
# ==========================================================

st.divider()

st.subheader("Class Probability Distribution")

probabilities = probabilities.cpu().numpy()[0]

labels = []

for i in range(num_classes):

    labels.append(
        f"Gesture {inverse_map[i]}"
    )

fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=labels,

        y=probabilities * 100,

        text=[
            f"{p:.2f}%"
            for p in probabilities * 100
        ],

        textposition="outside"

    )

)

fig.update_layout(

    title="Prediction Confidence",

    xaxis_title="Gesture",

    yaxis_title="Probability (%)",

    height=500,

    template="plotly_dark"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# Top 3 Predictions
# ==========================================================

st.divider()

st.subheader("Top Predictions")

top3 = np.argsort(probabilities)[::-1][:3]

for rank, idx in enumerate(top3, start=1):

    st.write(

        f"**{rank}. Gesture {inverse_map[idx]}**"

        f" — {probabilities[idx]*100:.2f}%"

    )
# ==========================================================
# Sample Statistics
# ==========================================================

st.divider()

st.subheader("Sample Information")

try:

    sample = x.squeeze().cpu().numpy()

    height = sample.shape[0]
    width = sample.shape[1]
    frames = sample.shape[2]

    signal_min = float(sample.min())
    signal_max = float(sample.max())
    signal_mean = float(sample.mean())
    signal_std = float(sample.std())

except Exception:

    height = width = frames = 0

    signal_min = signal_max = 0

    signal_mean = signal_std = 0

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Height",
        height
    )

with col2:

    st.metric(
        "Width",
        width
    )

with col3:

    st.metric(
        "Frames",
        frames
    )

with col4:

    st.metric(
        "Classes",
        num_classes
    )

st.write("")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Minimum Signal",
        f"{signal_min:.4f}"
    )

    st.metric(
        "Mean Signal",
        f"{signal_mean:.4f}"
    )

with col2:

    st.metric(
        "Maximum Signal",
        f"{signal_max:.4f}"
    )

    st.metric(
        "Std Deviation",
        f"{signal_std:.4f}"
    )

# ==========================================================
# Model Information
# ==========================================================

st.divider()

st.subheader("Model Information")

info1, info2 = st.columns(2)

with info1:

    st.info(
        """
Model Architecture

• CNN Feature Extractor

• Batch Normalization

• GRU Sequence Learning

• Fully Connected Classifier
"""
    )

with info2:

    st.info(
        f"""
Training Device

{DEVICE}

Validation Accuracy

79.08%

Output Classes

{num_classes}
"""
    )

# ==========================================================
# Status
# ==========================================================

st.divider()

st.subheader("Prediction Status")

if confidence >= 90:

    st.success("Very High Confidence Prediction")

elif confidence >= 75:

    st.success("High Confidence Prediction")

elif confidence >= 50:

    st.warning("Moderate Confidence Prediction")

else:

    st.error("Low Confidence Prediction")

# ==========================================================
# Prediction Report
# ==========================================================

report = f"""
WaveSense AI Prediction Report
==============================

Predicted Gesture : {gesture}

Confidence : {confidence:.2f} %

Tensor Shape : {sample.shape}

Signal Minimum : {signal_min:.4f}

Signal Maximum : {signal_max:.4f}

Signal Mean : {signal_mean:.4f}

Signal Std : {signal_std:.4f}

Model : CNN + GRU

Validation Accuracy : 79.08 %
"""

st.download_button(

    label="Download Prediction Report",

    data=report,

    file_name="prediction_report.txt",

    mime="text/plain"

)
# ==========================================================
# Confidence Gauge
# ==========================================================

st.divider()

st.subheader("Confidence Gauge")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%"},
        title={"text": "Prediction Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00E5FF"},
            "steps": [
                {"range": [0, 50], "color": "#8B0000"},
                {"range": [50, 75], "color": "#DAA520"},
                {"range": [75, 100], "color": "#228B22"}
            ]
        }
    )
)

gauge.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(
    gauge,
    use_container_width=True
)

# ==========================================================
# Signal Histogram
# ==========================================================

st.divider()

st.subheader("Signal Distribution")

hist = go.Figure()

hist.add_trace(
    go.Histogram(
        x=sample.flatten(),
        nbinsx=40
    )
)

hist.update_layout(
    template="plotly_dark",
    xaxis_title="Normalized Signal",
    yaxis_title="Frequency",
    height=400
)

st.plotly_chart(
    hist,
    use_container_width=True
)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "WaveSense AI • CNN + GRU • Built with PyTorch & Streamlit"
)