import numpy as np
import scipy.io as sio
import torch


def preprocess_mat(mat_path, target_T=20):
    """
    Load and preprocess a WiFi .mat file.

    Returns
    -------
    tensor : torch.Tensor
        Shape: (1, 1, 20, 20, target_T)
    """

    mat = sio.loadmat(mat_path)

    if "velocity_spectrum_ro" not in mat:
        raise ValueError("velocity_spectrum_ro not found in mat file.")

    x = np.asarray(mat["velocity_spectrum_ro"], dtype=np.float32)

    if x.size == 0:
        raise ValueError("Empty WiFi tensor.")

    if x.ndim != 3:
        raise ValueError("Expected a 3D WiFi tensor.")

    x_min = np.min(x)
    x_max = np.max(x)

    if abs(x_max - x_min) < 1e-6:
        raise ValueError("Cannot normalize constant tensor.")

    # Normalize
    x = (x - x_min) / (x_max - x_min)

    # Fix temporal dimension
    T = x.shape[2]

    if T > target_T:
        x = x[:, :, :target_T]

    elif T < target_T:
        pad = target_T - T
        x = np.pad(
            x,
            ((0, 0), (0, 0), (0, pad)),
            mode="constant"
        )

    tensor = torch.tensor(
        x,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor


def load_raw_tensor(mat_path):

    mat = sio.loadmat(mat_path)

    if "velocity_spectrum_ro" not in mat:
        raise ValueError("velocity_spectrum_ro not found.")

    x = np.asarray(
        mat["velocity_spectrum_ro"],
        dtype=np.float32
    )

    if x.ndim != 3:
        raise ValueError("Expected a 3D tensor.")

    x_min = np.min(x)
    x_max = np.max(x)

    x = (x - x_min) / (x_max - x_min + 1e-6)

    return x