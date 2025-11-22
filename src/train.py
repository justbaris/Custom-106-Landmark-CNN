import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataset, cfg):
        self.model = model
        self.cfg = cfg
        self.device = cfg["device"]
        self.epochs = cfg["epochs"]

        self.loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=1e-4
        )

        # Mixed precision
        self.scaler = GradScaler()

        # LR scheduler: cosine decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        self.criterion = torch.nn.MSELoss()

        self.best_loss = float("inf")
        self.patience_counter = 0
        self.early_stop_patience = cfg.get("early_stop_patience", 8)

        os.makedirs("experiments/checkpoints", exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        loss_ema = None

        pbar = tqdm(self.loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            imgs = batch["image"].to(self.device)
            heatmaps = batch["heatmaps"].to(self.device)

            self.optimizer.zero_grad()

            # AMP training
            with autocast():
                preds = self.model(imgs)
                loss = self.criterion(preds, heatmaps)

            # Update EMA
            loss_ema = loss.item() if loss_ema is None else \
                (loss_ema * 0.9 + loss.item() * 0.1)

            # backward with scaler
            self.scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            pbar.set_postfix({"loss (EMA)": f"{loss_ema:.4f}"})

        return loss_ema

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch(epoch)

            # LR decay
            self.scheduler.step()

            # Checkpoint — last
            torch.save(self.model.state_dict(), "experiments/checkpoints/last.pth")

            # Checkpoint — best
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(self.model.state_dict(), "experiments/checkpoints/best.pth")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            print(f"[INFO] Epoch {epoch} → EMA Loss: {loss:.4f} | Best: {self.best_loss:.4f}")

            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print("[INFO] Early stopping triggered.")
                break