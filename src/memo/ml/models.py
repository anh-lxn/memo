from __future__ import annotations

import torch.nn as nn


class MembraneModel(nn.Module):
    def __init__(self, output_dim: int = 3, hidden_dims: tuple[int, ...] = (32, 64, 32), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 8

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

