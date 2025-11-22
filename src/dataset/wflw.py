import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils.heatmap import generate_heatmaps


class WFLW106Dataset(Dataset):
    def __init__(self, annotation_path, img_root, heatmap_size=64):
        with open(annotation_path, "r") as f:
            self.ann = json.load(f)

        self.img_root = img_root
        self.hm_size = heatmap_size

        self.keys = list(self.ann.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.ann[key]

        img = cv2.imread(f"{self.img_root}/{key}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        lm = np.array(item["landmarks_106"])
        lm = lm / 256.0  # normalize (assuming original image 256x256)

        heatmaps = generate_heatmaps(lm, self.hm_size, self.hm_size)

        return {
            "image": torch.tensor(img),
            "heatmaps": heatmaps
        }