from __future__ import annotations

import pandas as pd
import torch as pt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def prepare_data(
    data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    targets: str = "xyf",
    random_seed: int = 42,
) -> pt.Tensor:
    X = data[[f"Sensor R{i}" for i in range(1, 9)]].values

    if targets == "xy":
        y = data[["X", "Y"]].values
    elif targets == "f":
        y = data[["F"]].values
    else:
        y = data[["X", "Y", "F"]].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_seed,
        shuffle=True,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size: int = 32):
    X_train_tensor = pt.tensor(X_train, dtype=pt.float32)
    y_train_tensor = pt.tensor(y_train, dtype=pt.float32)
    X_val_tensor = pt.tensor(X_val, dtype=pt.float32)
    y_val_tensor = pt.tensor(y_val, dtype=pt.float32)
    X_test_tensor = pt.tensor(X_test, dtype=pt.float32)
    y_test_tensor = pt.tensor(y_test, dtype=pt.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

