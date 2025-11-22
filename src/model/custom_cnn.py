import torch
import torch.nn as nn
from .hourglass_blocks import Hourglass


class CustomBackbone(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.initial(x)


class Custom106Net(nn.Module):
    def __init__(self, num_landmarks=106, channels=32, depth=3, up_blocks=3):
        super().__init__()

        self.backbone = CustomBackbone(channels)
        self.hourglass = Hourglass(depth, channels)
        self.head = nn.Conv2d(channels, num_landmarks, 1)

    def forward(self, x):
        f = self.backbone(x)
        f = self.hourglass(f)
        out = self.head(f)
        return out