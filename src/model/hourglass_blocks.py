import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Hourglass(nn.Module):
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.down = nn.MaxPool2d(2)
        self.res = Residual(channels)

        if depth > 1:
            self.inner = Hourglass(depth - 1, channels)
        else:
            self.inner = Residual(channels)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        down = self.res(self.down(x))
        inner = self.inner(down)
        up = self.up(inner)
        return up + x