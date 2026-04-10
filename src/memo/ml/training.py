from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device: str = "cuda",
        patience: int = 50,
        min_delta: float = 0.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self._best_state_dict = None

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
        epochs_without_improvement = 0

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

            if avg_val < (self.best_val_loss - self.min_delta):
                self.best_val_loss = avg_val
                self.best_epoch = epoch
                self._best_state_dict = {
                    key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if print_status and epoch % 50 == 0:
                print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

            if epochs_without_improvement >= self.patience:
                if print_status:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best epoch: {self.best_epoch}, best val: {self.best_val_loss:.4f})"
                    )
                break

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

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

