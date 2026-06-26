import os
import torch
import torch.nn.functional as F

from model import WidarNet
from dataset import WidarDataset
from utils import preprocess_mat


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATASET_PATH = r"D:\LinkedIn Projects\New folder\BVP"

MODEL_PATH = r"outputs\wavesense_best.pth"

# CHANGE THIS TO ANY .MAT FILE YOU WANT TO TEST
SAMPLE_FILE = r"D:\LinkedIn Projects\New folder\BVP\BVP\20181211-VS\6-link\user8\user8-1-1-1-2-1-1e-07-100-20-100000-L0.mat"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------
# Build label mapping
# ----------------------------------------------------

dataset = WidarDataset(DATASET_PATH)

num_classes = len(dataset.label_map)

inverse_map = dataset.inverse_label_map


# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

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

print("----------------------------------------")
print("WaveSense AI Prediction")
print("----------------------------------------")


# ----------------------------------------------------
# Preprocess sample
# ----------------------------------------------------

x = preprocess_mat(SAMPLE_FILE)

x = x.to(DEVICE)


# ----------------------------------------------------
# Predict
# ----------------------------------------------------

with torch.no_grad():

    output = model(x)

    probabilities = F.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )


predicted_class = prediction.item()

original_label = inverse_map[predicted_class]


print("\nPrediction")
print("----------------------------------------")

print("Predicted Class :", predicted_class)

print("Original Label  :", original_label)

print(f"Confidence      : {confidence.item()*100:.2f}%")

print("\nClass Probabilities")
print("----------------------------------------")

for i in range(num_classes):

    print(
        f"Class {inverse_map[i]:2d} : "
        f"{probabilities[0][i].item()*100:.2f}%"
    )