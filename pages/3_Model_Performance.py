import os
import matplotlib.image as mpimg
import streamlit as st

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

OUTPUT_DIR = "outputs"

CONFUSION = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

LOSS = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

ACCURACY = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)

REPORT = os.path.join(
    OUTPUT_DIR,
    "training_report.txt"
)

LABELS = os.path.join(
    OUTPUT_DIR,
    "label_mapping.txt"
)

MODEL = os.path.join(
    OUTPUT_DIR,
    "wavesense_best.pth"
)

st.title("📈 Model Performance")

st.write(
"""
This page summarizes the performance of the trained
CNN + GRU model.
"""
)

st.divider()
# ==========================================================
# Model Status
# ==========================================================

left, right = st.columns(2)

with left:

    if os.path.exists(MODEL):

        st.success("✅ Trained model found")

    else:

        st.error("❌ Model not found")

with right:

    if os.path.exists(REPORT):

        st.success("✅ Training report found")

    else:

        st.error("❌ Training report missing")
# ==========================================================
# Training Curves
# ==========================================================

st.divider()

st.header("Training Curves")

c1, c2 = st.columns(2)

with c1:

    if os.path.exists(LOSS):

        st.subheader("Loss Curve")

        st.image(
            LOSS,
            use_container_width=True
        )
    else:
        st.warning("Loss curve not found.")

with c2:

    if os.path.exists(ACCURACY):

        st.subheader("Accuracy Curve")

        st.image(
            ACCURACY,
            use_container_width=True
        )
    else:
        st.warning("Accuracy Curve not found.")
# ==========================================================
# Confusion Matrix
# ==========================================================

st.divider()

st.header("Confusion Matrix")

if os.path.exists(CONFUSION):

    st.image(
        CONFUSION,
        use_container_width=True
    )

else:

    st.warning("Confusion matrix not found.")
# ==========================================================
# Training Report
# ==========================================================

st.divider()

st.header("Training Report")

if os.path.exists(REPORT):

    with open(REPORT, "r", encoding="utf-8") as f:

        report = f.read()

    st.text(report)

    st.download_button(

        "Download Training Report",

        report,

        file_name="training_report.txt"

    )

else:

    st.warning("Training report not available.")
# ==========================================================
# Label Mapping
# ==========================================================

st.divider()

st.header("Label Mapping")

if os.path.exists(LABELS):

    with open(LABELS, "r", encoding="utf-8") as f:

        labels = f.read()

    st.text(labels)

    st.download_button(

        "Download Label Mapping",

        labels,

        file_name="label_mapping.txt"

    )

else:

    st.warning("Label mapping not available.")
# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "WaveSense AI • Model Evaluation Dashboard"
)