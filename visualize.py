import os
import numpy as np
import torch

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from model import WidarNet
from dataset import WidarDataset
from utils import preprocess_mat
from utils import load_raw_tensor


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = r"D:\LinkedIn Projects\New folder\BVP"

MODEL_PATH = r"outputs\wavesense_best.pth"

SAMPLE_FILE = r"D:\LinkedIn Projects\New folder\BVP\BVP\20181211-VS\6-link\user8\user8-1-1-1-2-1-1e-07-100-20-100000-L0.mat"

OUTPUT_DIR = "outputs"

SAVE_GIF = True

SAVE_MP4 = False

FPS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("--------------------------------------------")
print("WaveSense AI Dashboard")
print("--------------------------------------------")
print("Loading dataset...")
# ============================================================
# Load Dataset
# ============================================================

dataset = WidarDataset(DATASET_PATH)

num_classes = len(dataset.label_map)

inverse_map = dataset.inverse_label_map

print("Loading trained model...")


# ============================================================
# Load Model
# ============================================================

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


# ============================================================
# Load Sample
# ============================================================

print("Loading WiFi sample...")

input_tensor = preprocess_mat(
    SAMPLE_FILE
).to(DEVICE)

raw_tensor = load_raw_tensor(
    SAMPLE_FILE
)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )


predicted_class = prediction.item()

gesture = inverse_map[predicted_class]

confidence = confidence.item() * 100


print("--------------------------------------------")
print("Prediction")
print("--------------------------------------------")
print(f"Gesture    : {gesture}")
print(f"Confidence : {confidence:.2f}%")
print("--------------------------------------------")


# ============================================================
# Dashboard Layout
# ============================================================

frames = raw_tensor.shape[2]

fig = plt.figure(

    figsize=(14, 7),

    facecolor="white"

)

# Left : WiFi Heatmap
heatmap_ax = fig.add_subplot(1, 2, 1)

# Right : Motion Trajectory
track_ax = fig.add_subplot(1, 2, 2)

fig.suptitle(

    "WaveSense AI Dashboard",

    fontsize=18,

    fontweight="bold"

)


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

heatmap = heatmap_ax.imshow(

    raw_tensor[:, :, 0],

    cmap="viridis",

    vmin=0,

    vmax=1,

    animated=True

)

plt.colorbar(

    heatmap,

    ax=heatmap_ax,

    fraction=0.046,

    pad=0.04

)

heatmap_ax.set_title("WiFi Heatmap")

heatmap_ax.set_xlabel("Angle Bin")

heatmap_ax.set_ylabel("Velocity Bin")


# ------------------------------------------------------------
# Motion Trajectory
# ------------------------------------------------------------

track_ax.set_title("Motion Trajectory")

track_ax.set_xlim(0, raw_tensor.shape[1])

track_ax.set_ylim(raw_tensor.shape[0], 0)

track_ax.grid(True)

track_ax.set_xlabel("Angle Bin")

track_ax.set_ylabel("Velocity Bin")


trajectory_x = []

trajectory_y = []

trajectory_plot, = track_ax.plot(

    [],

    [],

    "-o",

    linewidth=2,

    markersize=5

)
current_point = track_ax.scatter(
    [],
    [],
    s=120,
    c="red",
    edgecolors="black",
    zorder=5
)
# ------------------------------------------------------------
# Information Panel
# ------------------------------------------------------------

info_text = fig.text(

    0.50,

    0.02,

    "",

    ha="center",

    fontsize=12

)
# ============================================================
# Animation Function
# ============================================================

def update(frame):

    # --------------------------------------------------------
    # Update Heatmap
    # --------------------------------------------------------

    current = raw_tensor[:, :, frame]

    heatmap.set_array(current)

    # --------------------------------------------------------
    # Find strongest reflection
    # --------------------------------------------------------

    y, x = np.unravel_index(

        np.argmax(current),

        current.shape

    )

    trajectory_x.append(x)

    trajectory_y.append(y)

    # --------------------------------------------------------
    # Update Motion Trajectory
    # --------------------------------------------------------

    trajectory_plot.set_data(

        trajectory_x,

        trajectory_y

    )
    current_point.set_offsets([[x, y]])

    # --------------------------------------------------------
    # Update Dashboard Text
    # --------------------------------------------------------

    info_text.set_text(

        f"Gesture : {gesture}      "

        f"Confidence : {confidence:.2f}%      "

        f"Frame : {frame + 1}/{frames}"

    )

    # --------------------------------------------------------
    # Update Heatmap Title
    # --------------------------------------------------------

    heatmap_ax.set_title(

        f"WiFi Heatmap (Frame {frame+1})"

    )

    return (

        heatmap,

        trajectory_plot,

        current_point,

        info_text,

    )


# ============================================================
# Create Animation
# ============================================================

animation = FuncAnimation(

    fig,

    update,

    frames=frames,

    interval=250,

    repeat=True,

    blit=False

)
# ============================================================
# Save Animation
# ============================================================

if SAVE_GIF:

    print("\nSaving GIF...")

    try:

        gif_path = os.path.join(

            OUTPUT_DIR,

            "wavesense_dashboard.gif"

        )

        animation.save(

            gif_path,

            writer="pillow",

            fps=FPS

        )

        print("GIF saved successfully.")
        print(gif_path)

    except Exception as e:

        print("\nUnable to save GIF.")
        print(e)

        print("\nInstall Pillow:")

        print("pip install pillow")


# ============================================================
# Save MP4 (Optional)
# ============================================================

if SAVE_MP4:

    print("\nSaving MP4...")

    try:

        mp4_path = os.path.join(

            OUTPUT_DIR,

            "wavesense_dashboard.mp4"

        )

        animation.save(

            mp4_path,

            writer="ffmpeg",

            fps=FPS

        )

        print("MP4 saved successfully.")
        print(mp4_path)

    except Exception as e:

        print("\nUnable to save MP4.")
        print(e)

        print("\nInstall FFmpeg if you want MP4 export.")


# ============================================================
# Display Dashboard
# ============================================================

plt.tight_layout()

plt.show()

plt.close()

print("\nVisualization Finished.")