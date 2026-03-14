from __future__ import annotations

import numpy as np
import torch

from memo.types import NormalizationStats, Prediction


class ModelPredictor:
    def __init__(
        self,
        model_class,
        model_path,
        output_dim: int,
        device: str = "cuda",
        mins=None,
        maxs=None,
    ):
        self.model_class = model_class
        self.model_path = model_path
        self.output_dim = output_dim
        self.device = device
        self.stats = None
        if mins is not None and maxs is not None:
            self.stats = NormalizationStats(mins=np.array(mins), maxs=np.array(maxs))
        self.model = self.load_model()

    def load_model(self):
        model = self.model_class(output_dim=self.output_dim).to(self.device)
        weights = torch.load(self.model_path, weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        return model

    def min_max_normalize(self, input_data):
        if self.stats is None:
            return input_data
        norm_input = (input_data - self.stats.mins) / (self.stats.maxs - self.stats.mins)
        norm_input = 2 * norm_input - 1
        return norm_input

    @torch.no_grad()
    def predict(self, input_data):
        norm_input = self.min_max_normalize(input_data)
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).to(self.device)
        output_tensor = self.model(input_tensor)
        return output_tensor.cpu().numpy()

    @torch.no_grad()
    def predict_structured(self, input_data) -> Prediction:
        raw_output = self.predict(input_data)
        values = np.atleast_1d(raw_output).astype(float)
        prediction = Prediction(raw_output=np.array(raw_output))
        if self.output_dim >= 1:
            prediction.x = values[0]
        if self.output_dim >= 2:
            prediction.y = values[1]
        if self.output_dim == 1:
            prediction.force = values[0]
            prediction.x = None
        elif self.output_dim >= 3:
            prediction.force = values[2]
        return prediction
