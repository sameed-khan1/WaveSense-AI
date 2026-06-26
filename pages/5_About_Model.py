import streamlit as st

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(

    page_title="About WaveSense AI",

    page_icon="🤖",

    layout="wide"

)

st.title("🤖 About WaveSense AI")

st.write(
"""
WaveSense AI is an end-to-end Deep Learning system that
recognizes human gestures using WiFi Channel State Information (CSI)
instead of cameras or wearable sensors.
"""
)

st.divider()
# ==========================================================
# Project Overview
# ==========================================================

st.header("Project Overview")

st.info(
"""
WaveSense AI analyzes WiFi CSI signals and predicts
human gestures using a hybrid CNN + GRU neural network.

The model processes Doppler velocity spectrum data
extracted from WiFi signals and learns temporal
patterns corresponding to different human activities.
"""
)
# ==========================================================
# Architecture
# ==========================================================

st.header("Model Architecture")

st.markdown("""
### CNN Feature Extractor

- Conv2D
- Batch Normalization
- ReLU
- MaxPooling

### Temporal Learning

- 2-Layer GRU
- Hidden Size = 256

### Classifier

- Fully Connected Layers
- Dropout
- Softmax Prediction
""")
# ==========================================================
# Dataset
# ==========================================================

st.header("Dataset")

st.markdown("""
**Dataset:** Widar 3.0 BVP

Characteristics

- Doppler Velocity Spectrum

- 10 Gesture Classes

- 43,657 Samples

- 20 Temporal Frames

- CSI-based Human Activity Recognition
""")
# ==========================================================
# Training Details
# ==========================================================

st.header("Training")

c1, c2 = st.columns(2)

with c1:

    st.metric("Training Samples", "34,925")

    st.metric("Validation Samples", "8,732")

with c2:

    st.metric("Best Validation Accuracy", "79.08%")

    st.metric("Framework", "PyTorch")
# ==========================================================
# Technologies
# ==========================================================

st.header("Technology Stack")

st.markdown("""
- Python

- PyTorch

- NumPy

- SciPy

- Streamlit

- Plotly

- Matplotlib

- Scikit-learn
""")
# ==========================================================
# Future Improvements
# ==========================================================

st.header("Future Work")

st.markdown("""
- Transformer-based sequence models

- Real-time WiFi sensing

- Live CSI stream processing

- Multi-person activity recognition

- Edge AI deployment

- Mobile application integration
""")
# ==========================================================
# Author
# ==========================================================

st.header("Developer")

st.success("""
Developed by

Sameedullah Khan

AI • Machine Learning • Computer Vision • Deep Learning
""")
# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "WaveSense AI © 2026"
)