from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device: str = "cuda"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_step(self, batch_x, batch_y):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(batch_x)
        loss = self.loss_fn(preds, batch_y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, batch_x, batch_y):
        self.model.eval()
        preds = self.model(batch_x)
        loss = self.loss_fn(preds, batch_y)
        return loss.item()

    def fit(self, train_loader, val_loader, epochs: int = 100, print_status: bool = False):
        print("Starting training...")
        for epoch in range(epochs):
            batch_train_losses = []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = self.train_step(xb, yb)
                batch_train_losses.append(loss)

            avg_train = np.mean(batch_train_losses)
            self.train_losses.append(avg_train)

            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                val_losses.append(self.val_step(xb, yb))

            avg_val = np.mean(val_losses)
            self.val_losses.append(avg_val)

            if print_status and epoch % 50 == 0:
                print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    @torch.no_grad()
    def test(self, test_loader):
        losses = []
        for xb, yb in test_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self.model(xb)
            losses.append(self.loss_fn(preds, yb).item())

        return np.mean(losses)

    def plot_losses(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def save(self, path: str = "model.pt"):
        torch.save(self.model.state_dict(), path)

