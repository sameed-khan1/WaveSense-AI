import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class WidarDataset(Dataset):

    def __init__(self, root_dir, target_T=20):

        self.root_dir = root_dir
        self.target_T = target_T

        self.files = []
        self.raw_labels = []

        print("Scanning dataset...")

        for root, _, files in os.walk(root_dir):

            for file in files:

                if not file.endswith(".mat"):
                    continue

                path = os.path.join(root, file)

                try:
                    # Extract gesture label from filename
                    # Example: user1-3-1-1-1.mat -> label = 3
                    label = int(file.split("-")[1])

                except Exception:
                    continue

                self.files.append(path)
                self.raw_labels.append(label)

        unique = sorted(list(set(self.raw_labels)))

        # Original label -> model label
        self.label_map = {
            label: idx
            for idx, label in enumerate(unique)
        }

        # Model label -> original label
        self.inverse_label_map = {
            idx: label
            for label, idx in self.label_map.items()
        }

        self.labels = [
            self.label_map[label]
            for label in self.raw_labels
        ]

        print(f"Loaded samples : {len(self.files)}")
        print(f"Classes : {len(unique)}")
        print("Label Map :", self.label_map)
        print("Inverse Label Map :", self.inverse_label_map)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        while True:

            try:

                path = self.files[idx]
                label = self.labels[idx]

                mat = sio.loadmat(path)

                if "velocity_spectrum_ro" not in mat:

                    idx = (idx + 1) % len(self.files)
                    continue

                x = mat["velocity_spectrum_ro"]

                x = np.asarray(
                    x,
                    dtype=np.float32
                )

                # Skip empty arrays
                if x.size == 0:

                    idx = (idx + 1) % len(self.files)
                    continue

                # Must be (20,20,T)
                if x.ndim != 3:

                    idx = (idx + 1) % len(self.files)
                    continue

                # Normalize
                x_min = np.min(x)
                x_max = np.max(x)

                if abs(x_max - x_min) < 1e-6:

                    idx = (idx + 1) % len(self.files)
                    continue

                x = (x - x_min) / (x_max - x_min)

                # Fix temporal length
                T = x.shape[2]

                if T > self.target_T:

                    x = x[:, :, :self.target_T]

                elif T < self.target_T:

                    pad = self.target_T - T

                    x = np.pad(
                        x,
                        ((0, 0), (0, 0), (0, pad)),
                        mode="constant"
                    )

                x = torch.tensor(
                    x,
                    dtype=torch.float32
                )

                # Shape:
                # (20,20,20) -> (1,20,20,20)
                x = x.unsqueeze(0)

                y = torch.tensor(
                    label,
                    dtype=torch.long
                )

                return x, y

            except Exception as e:

                print(f"\nSkipping corrupted file:")
                print(self.files[idx])
                print(e)

                idx = (idx + 1) % len(self.files)
