import numpy as np
import torch

def generate_heatmaps(landmarks, h, w, sigma=2.0):
    """
    landmarks: (N_landmarks, 2) normalized coordinates (0-1 range)
    h, w: output heatmap size
    """
    num_landmarks = landmarks.shape[0]
    heatmaps = np.zeros((num_landmarks, h, w), dtype=np.float32)

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    for i, (lx, ly) in enumerate(landmarks):
        cx = lx * w
        cy = ly * h

        heatmaps[i] = np.exp(
            - ((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)
        )

    return torch.tensor(heatmaps)


def get_max_preds(batch_heatmaps):
    """
    batch_heatmaps: numpy array (B, 106, H, W)
    Returns:
        coords: (B, 106, 2) normalized (0-1)
        maxvals: confidence scores
    """
    assert isinstance(batch_heatmaps, np.ndarray), "Input must be numpy array"
    B, L, H, W = batch_heatmaps.shape

    heatmaps_reshaped = batch_heatmaps.reshape(B, L, -1)
    idx = np.argmax(heatmaps_reshaped, axis=2)
    maxvals = np.max(heatmaps_reshaped, axis=2)

    coords = np.zeros((B, L, 2), dtype=np.float32)

    coords[:, :, 0] = (idx % W) / W     # x normalized
    coords[:, :, 1] = (idx // W) / H    # y normalized

    return coords, maxvals