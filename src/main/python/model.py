import torch.nn as nn
import torch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MembraneModel(nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device="cuda"):
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

    def fit(self, train_loader, val_loader, epochs=100, print_status=False):
        #print("Starting training...")
        for epoch in range(epochs):
            batch_train_losses = []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = self.train_step(xb, yb)
                batch_train_losses.append(loss)

            avg_train = np.mean(batch_train_losses)
            self.train_losses.append(avg_train)

            # validation
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
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save(self, path="model.pt"):
        torch.save(self.model.state_dict(), path)

class ModelEvaluator:
    def __init__(self, model_class, test_loader, output_dim, device="cuda"):
        self.model_class = model_class
        self.test_loader = test_loader
        self.output_dim = output_dim
        self.device = device
        self.results = {}

    def load_model(self, path):
        model = self.model_class(output_dim=self.output_dim).to(self.device)
        weights = torch.load(path, weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        return model

    def evaluate_metrics(self, model):
        preds = []
        targets = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = model(x)

                preds.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.vstack(preds)
        targets = np.vstack(targets)

        mse  = mean_squared_error(targets, preds)
        mae  = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mse)
        #r2   = r2_score(targets, preds)

        return {
            "mse":  float(mse),
            "mae":  float(mae),
            "rmse": float(rmse),
            #"r2":   float(r2)
        }

    def evaluate_from_json(self, json_path):
        with open(json_path) as f:
            results = json.load(f)

        summary = {}

        for loss_name, info in results.items():
            print(f"\nEvaluating loss: {loss_name}")

            # choose model path depending on output_dim
            path = info["model_xy_path"] if self.output_dim == 2 else info["model_f_path"]

            model = self.load_model(path)
            metrics = self.evaluate_metrics(model)

            summary[loss_name] = metrics
            print(metrics)

        self.results = summary
        return summary

    def find_best_loss(self, metric="mse"):
        if not self.results:
            raise ValueError("No results found. Run evaluate_from_json() first.")

        best_name = None
        best_value = float("inf")
        best_metrics = None

        for loss_name, metrics in self.results.items():
            val = metrics[metric]
            if val < best_value:
                best_value = val
                best_name = loss_name
                best_metrics = metrics

        print(f"\nBest loss function: Model_{best_name} with metric_{metric}: {best_value:.4f}")
        return {
            "best_loss": best_name,
            "best_value": best_value,
            "metrics": best_metrics
        }

class ModelPredictor:
    def __init__(self, model_class, model_path, output_dim, device="cuda", mins: pd.DataFrame=None, maxs: pd.DataFrame=None):
        self.model_class = model_class
        self.model_path = model_path
        self.output_dim = output_dim
        self.device = device
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.model = self.load_model()

    def load_model(self):
        model = self.model_class(output_dim=self.output_dim).to(self.device)
        weights = torch.load(self.model_path, weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        return model

    def min_max_normalize(self, input: torch.tensor) -> pd.DataFrame:
        norm_input = (input - self.mins) / (self.maxs - self.mins)
        norm_input = 2 * norm_input - 1  # Scale to [-1, 1]
        return norm_input

    @torch.no_grad()
    def predict(self, input_data):
        #norm_input = self.min_max_normalize(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        output_tensor = self.model(input_tensor)
        return output_tensor.cpu().numpy()