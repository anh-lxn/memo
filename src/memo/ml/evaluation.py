from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelEvaluator:
    def __init__(self, model_class, test_loader, output_dim: int, device: str = "cuda"):
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

        mse = mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mse)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
        }

    def evaluate_from_json(self, json_path):
        with open(json_path) as f:
            results = json.load(f)

        summary = {}
        for loss_name, info in results.items():
            print(f"\nEvaluating loss: {loss_name}")
            path = info["model_xy_path"] if self.output_dim == 2 else info["model_f_path"]
            model = self.load_model(path)
            metrics = self.evaluate_metrics(model)
            summary[loss_name] = metrics
            print(metrics)

        self.results = summary
        return summary

    def find_best_loss(self, metric: str = "mse"):
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
            "metrics": best_metrics,
        }

