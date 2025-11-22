import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .losses import HeatmapLoss


class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.cfg = config

        self.loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2
        )

        self.loss_fn = HeatmapLoss()
        self.opt = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"]
        )

        self.device = config["device"]
        self.ema = EMA(model)
        self.best_loss = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            imgs = batch["image"].to(self.device)
            heatmaps = batch["heatmaps"].to(self.device)

            pred = self.model(imgs)

            loss = self.loss_fn(pred, heatmaps)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.ema.update(self.model)

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (len(pbar)+1)})

        return total_loss / len(self.loader)

    def fit(self, epochs):
        for e in range(1, epochs + 1):
            loss = self.train_epoch(e)
            print(f"Epoch {e} | Loss: {loss:.4f}")

            # Save best EMA model
            if loss < self.best_loss:
                self.best_loss = loss
                self.ema.apply_shadow(self.model)
                torch.save(self.model.state_dict(), "experiments/checkpoints/best_ema.pth")
                self.ema.restore(self.model)