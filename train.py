import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import WidarDataset
from model import WidarNet


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATASET_PATH = r"D:\LinkedIn Projects\New folder\BVP"

OUTPUT_DIR = "outputs"

BATCH_SIZE = 64

NUM_EPOCHS = 100

LEARNING_RATE = 0.001

TRAIN_RATIO = 0.8

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("---------------------------------------")
print("WaveSense AI")
print("---------------------------------------")
print("Device :", DEVICE)
print("Epochs :", NUM_EPOCHS)
print("Batch Size :", BATCH_SIZE)
print("---------------------------------------")


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

dataset = WidarDataset(DATASET_PATH)

num_classes = len(dataset.label_map)

print("Number of classes :", num_classes)

train_size = int(
    TRAIN_RATIO * len(dataset)
)

val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(

    dataset,

    [train_size, val_size],

    generator=generator

)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available()

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()

)


# ----------------------------------------------------
# Model
# ----------------------------------------------------

model = WidarNet(

    num_classes=num_classes

).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)
# ----------------------------------------------------
# Training History
# ----------------------------------------------------

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

patience = 10
patience_counter = 0


print("\nStarting Training...\n")


# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    print("-" * 50)

    # ===========================
    # TRAINING
    # ===========================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += y.size(0)

        correct += (predicted == y).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    train_losses.append(train_loss)

    train_accuracies.append(train_accuracy)

    print(f"Train Loss     : {train_loss:.4f}")

    print(f"Train Accuracy : {train_accuracy:.2f}%")



    # ===========================
    # VALIDATION
    # ===========================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)

            loss = criterion(outputs, y)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # ===========================
    # Validation Statistics
    # ===========================

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = 100.0 * val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Validation Loss     : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_accuracy:.2f}%")

    # ===========================
    # Learning Rate Scheduler
    # ===========================

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate       : {current_lr:.6f}")

    # ===========================
    # Save Best Model
    # ===========================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(model.state_dict())

        torch.save(
            best_model_weights,
            os.path.join(
                OUTPUT_DIR,
                "wavesense_best.pth"
            )
        )

        patience_counter = 0

        print("Best model updated.")

    else:

        patience_counter += 1

        print(f"No improvement ({patience_counter}/{patience})")

    # ===========================
    # Early Stopping
    # ===========================

    if patience_counter >= patience:

        print("\nEarly stopping triggered.")

        break


print("\nTraining Finished.")
print(f"Best Validation Accuracy : {best_val_accuracy:.2f}%")

model.load_state_dict(best_model_weights)
model.eval()
 
# ----------------------------------------------------
# Final Evaluation
# ----------------------------------------------------

print("\nEvaluating best model...\n")

all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():

    for x, y in val_loader:

        x = x.to(DEVICE)

        outputs = model(x)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            y.numpy()
        )


# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------

cm = confusion_matrix(

    all_labels,

    all_predictions

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        dataset.inverse_label_map[i]
        for i in range(num_classes)
    ]

)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(

    cmap="Blues",

    ax=ax,

    xticks_rotation=45,

    colorbar=False

)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),

    dpi=300

)

plt.close()


# ----------------------------------------------------
# Loss Curve
# ----------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(

    train_losses,

    linewidth=2,

    label="Training"

)

plt.plot(

    val_losses,

    linewidth=2,

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    ),

    dpi=300

)

plt.close()


# ----------------------------------------------------
# Accuracy Curve
# ----------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(

    train_accuracies,

    linewidth=2,

    label="Training"

)

plt.plot(

    val_accuracies,

    linewidth=2,

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("Accuracy Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "accuracy_curve.png"
    ),

    dpi=300

)

plt.close()
# ----------------------------------------------------
# Calculate Final Accuracy
# ----------------------------------------------------

correct = np.sum(
    np.array(all_predictions) == np.array(all_labels)
)

total = len(all_labels)

final_accuracy = 100.0 * correct / total

print("\n----------------------------------------")
print("Training Summary")
print("----------------------------------------")
print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
print(f"Number of Classes  : {num_classes}")
print(f"Best Validation Accuracy : {best_val_accuracy:.2f}%")
print(f"Final Accuracy     : {final_accuracy:.2f}%")
print("----------------------------------------")


# ----------------------------------------------------
# Save Label Mapping
# ----------------------------------------------------

label_file = os.path.join(
    OUTPUT_DIR,
    "label_mapping.txt"
)

with open(label_file, "w") as f:

    f.write("WaveSense AI Label Mapping\n")
    f.write("==========================\n\n")

    for original_label, mapped_label in dataset.label_map.items():

        f.write(
            f"Original Label {original_label} -> Model Class {mapped_label}\n"
        )


# ----------------------------------------------------
# Save Training Report
# ----------------------------------------------------

report_file = os.path.join(
    OUTPUT_DIR,
    "training_report.txt"
)

with open(report_file, "w") as f:

    f.write("WaveSense AI Training Report\n")
    f.write("============================\n\n")

    f.write(f"Dataset Path : {DATASET_PATH}\n")
    f.write(f"Training Samples : {len(train_dataset)}\n")
    f.write(f"Validation Samples : {len(val_dataset)}\n")
    f.write(f"Classes : {num_classes}\n")
    f.write(f"Epochs Completed : {len(train_losses)}\n")
    f.write(f"Best Validation Accuracy : {best_val_accuracy:.2f}%\n")
    f.write(f"Final Accuracy : {final_accuracy:.2f}%\n")


print("\nFiles saved to:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
print("  wavesense_best.pth")
print("  confusion_matrix.png")
print("  loss_curve.png")
print("  accuracy_curve.png")
print("  label_mapping.txt")
print("  training_report.txt")

print("\nTraining completed successfully.")