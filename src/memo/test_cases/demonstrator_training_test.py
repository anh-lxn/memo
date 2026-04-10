from __future__ import annotations

"""
Dateiname: demonstrator_training_test.py
Beschreibung:
Standalone-Testskript fuer das Demonstrator-Training auf Basis der alten
Trainingslogik. Das Skript liest ein CSV mit X/Y/F + 8 Sensorwerten ein,
normalisiert die Sensorwerte wie im Legacy-Skript, trainiert das alte
Demonstrator-Modell und visualisiert zufaellige Testvorhersagen.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.transforms import Affine2D
from sklearn.model_selection import train_test_split

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENSOR_COLUMNS = tuple(f"Sensor R{i}" for i in range(1, 9))
TARGET_COLUMNS = ("X", "Y", "F")
SENSOR_POSITIONS = [
    (-315, 315),
    (0, 315),
    (315, 315),
    (-315, 0),
    (315, 0),
    (-315, -315),
    (0, -315),
    (315, -315),
]

#data/recorded_samples/3D_Messung_01_10N.csv
#data/interpolation/auswertung_gesamt_10N_original.csv

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Legacy-style demonstrator training test")
    parser.add_argument(
        "--csv",
        default="data/recorded_samples/3D_Messung_01_10N.csv",
        help="Pfad zur CSV-Datei relativ zum Repo oder als absoluter Pfad.",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Anzahl Trainingsepochen.")
    parser.add_argument("--lr", type=float, default=0.00205, help="Learning Rate fuer Adam.")
    parser.add_argument(
        "--percentage",
        type=float,
        default=1.0,
        help="Anteil des Datensatzes, der fuer Train/Val/Test verwendet wird.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed fuer reproduzierbare Splits.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Anzahl zufaelliger Testpunkte fuer die Visualisierung.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Speichert das trainierte Modell unter models/final_models/",
    )
    parser.add_argument(
        "--normalization",
        choices=("legacy", "raw", "channel_minmax", "compare"),
        default="compare",
        help=(
            "legacy = alte per-Sample-Normalisierung, "
            "raw = Rohwerte, "
            "channel_minmax = pro Sensorkanal min/max ueber den Trainingssatz, "
            "compare = trainiert alle drei Varianten nacheinander."
        ),
    )
    return parser.parse_args(argv)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_csv_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return resolve_repo_root() / path


def load_legacy_style_data(csv_path: Path):
    data = pd.read_csv(csv_path)
    missing_columns = [column for column in (*TARGET_COLUMNS, *SENSOR_COLUMNS) if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    load_pos_x = data["X"].astype(float).tolist()
    load_pos_y = data["Y"].astype(float).tolist()
    load_value = data["F"].astype(float).tolist()
    strains = [data[column].astype(float).tolist() for column in SENSOR_COLUMNS]
    return load_pos_x, load_pos_y, load_value, strains


def normalize_strain_data_legacy(strains: list[list[float]]) -> list[list[float]]:
    """
    Reproduziert die Legacy-Logik aus dem alten Demonstrator-Skript:
    Die Normalisierung erfolgt pro Messpunkt ueber alle 8 Sensoren.
    """
    strains_array = np.array(strains, dtype=float)
    min_vals = np.min(strains_array, axis=0)
    max_vals = np.max(strains_array, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized_strains = (strains_array - min_vals) / range_vals
    return normalized_strains.tolist()


def build_feature_matrix(strains: list[list[float]]) -> np.ndarray:
    return np.column_stack(strains).astype(float)


def minmax_scale_from_train(x_train: np.ndarray, x_other: np.ndarray) -> np.ndarray:
    mins = np.min(x_train, axis=0)
    maxs = np.max(x_train, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (x_other - mins) / ranges


def prepare_data_legacy(
    x_matrix: np.ndarray,
    load_pos_x: list[float],
    load_pos_y: list[float],
    load_value: list[float],
    percentage: float,
    seed: int,
):
    if len(x_matrix) != len(load_pos_x):
        raise ValueError("Input lengths do not match.")

    y_matrix = np.column_stack((load_pos_x, load_pos_y, load_value))

    if percentage >= 1.0:
        x_reduced = x_matrix
        y_reduced = y_matrix
    else:
        x_reduced, _, y_reduced, _ = train_test_split(
            x_matrix,
            y_matrix,
            train_size=percentage,
            shuffle=True,
            random_state=seed,
        )

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_reduced,
        y_reduced,
        train_size=0.8,
        shuffle=True,
        random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        shuffle=True,
        random_state=seed,
    )

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


class LegacyDemonstratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
        )

    def forward(self, x):
        return self.net(x)


def create_scatterplot(load_pos_x, load_pos_y, sensor_pos, rectangle_width=50, rectangle_height=20):
    rotation_angles = [45, 0, -45, 90, 45, 0, -45, 90]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color="yellow", alpha=0.4)
    ax.scatter(load_pos_x, load_pos_y, c="blue", alpha=0.5, label="Load Points")

    for index, ((sx, sy), angle) in enumerate(zip(sensor_pos, rotation_angles), start=1):
        trans = Affine2D().rotate_deg_around(sx, sy, angle) + ax.transData
        rect = patches.Rectangle(
            (sx - rectangle_width / 2, sy - rectangle_height / 2),
            rectangle_width,
            rectangle_height,
            edgecolor="red",
            facecolor="red",
            alpha=0.6,
            transform=trans,
        )
        ax.add_patch(rect)
        ax.text(
            sx,
            sy + 30,
            f"Sensor R{index}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.3", "alpha": 0.9},
        )

    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_aspect("equal")
    ax.set_xlabel("X-Achse [mm]")
    ax.set_ylabel("Y-Achse [mm]")
    ax.set_title("Lastpunkte (Blau), Sensorpatches (Rot)")
    ax.grid(True)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.legend()
    plt.show()


def create_scatterplot_testing(
    load_pos_x,
    load_pos_x_pred,
    load_pos_y,
    load_pos_y_pred,
    load_value,
    load_value_pred,
    sensor_pos,
    rectangle_width=50,
    rectangle_height=20,
):
    rotation_angles = [45, 0, -45, 90, 45, 0, -45, 90]
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color="yellow", alpha=0.4)

    for index, (x_value, y_value) in enumerate(zip(load_pos_x, load_pos_y), start=1):
        ax.scatter(x_value, y_value, c="blue", alpha=0.8)
        ax.text(
            x_value,
            y_value,
            str(index),
            fontsize=7,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.3", "alpha": 0.9},
        )

    for index, (x_value, y_value) in enumerate(zip(load_pos_x_pred, load_pos_y_pred), start=1):
        ax.scatter(x_value, y_value, c="green", alpha=0.5)
        ax.text(
            x_value,
            y_value,
            str(index),
            fontsize=7,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.3", "alpha": 0.9},
        )

    for index, ((sx, sy), angle) in enumerate(zip(sensor_pos, rotation_angles), start=1):
        trans = Affine2D().rotate_deg_around(sx, sy, angle) + ax.transData
        rect = patches.Rectangle(
            (sx - rectangle_width / 2, sy - rectangle_height / 2),
            rectangle_width,
            rectangle_height,
            edgecolor="red",
            facecolor="red",
            alpha=0.6,
            transform=trans,
        )
        ax.add_patch(rect)
        ax.text(
            sx,
            sy + 30,
            f"Sensor {index}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.3", "alpha": 0.9},
        )

    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_aspect("equal")
    ax.set_xlabel("X-Achse [mm]")
    ax.set_ylabel("Y-Achse [mm]")
    ax.set_title("Memo-2D-Plot - Lastpunkte (Blau), Vorhergesagte Lastpunkte (Gruen)")
    ax.grid(True)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    legend_box = fig.add_axes([0.76, 0.18, 0.2, 0.64], frame_on=True)
    legend_box.axis("off")
    for index, (force_true, force_pred) in enumerate(zip(load_value, load_value_pred), start=1):
        delta_force = force_true - force_pred
        color = "green" if -2 <= delta_force <= 2 else "red"
        legend_text = (
            f"Punkt {index}: F={force_true:.2f}N, "
            f"F_pred={force_pred:.2f}N, deltaF={delta_force:.2f}N"
        )
        legend_box.text(0, 0.95 - index * 0.08, legend_text, fontsize=9, ha="left", color=color)

    plt.show()


def test_model(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test.to(DEVICE))
        mae_x = torch.abs(predictions[:, 0] - y_test[:, 0].to(DEVICE)).mean().item()
        mae_y = torch.abs(predictions[:, 1] - y_test[:, 1].to(DEVICE)).mean().item()
        mae_f = torch.abs(predictions[:, 2] - y_test[:, 2].to(DEVICE)).mean().item()
        mse = nn.MSELoss()(predictions, y_test.to(DEVICE)).item()

    print(f"Test MSE: {mse:.4f}")
    print(f"Mittlere absolute Abweichung X [mm]: {mae_x:.4f}")
    print(f"Mittlere absolute Abweichung Y [mm]: {mae_y:.4f}")
    print(f"Mittlere absolute Abweichung F [N]: {mae_f:.4f}")
    return {
        "mse": mse,
        "mae_x": mae_x,
        "mae_y": mae_y,
        "mae_f": mae_f,
    }


def test_random_samples(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, num_samples: int = 10):
    sample_count = min(num_samples, len(x_test))
    random_indices = random.sample(range(len(x_test)), sample_count)
    x_sample = x_test[random_indices]
    y_sample = y_test[random_indices]

    model.eval()
    with torch.no_grad():
        predictions = model(x_sample.to(DEVICE)).cpu()

    return x_sample, y_sample, predictions


def train_model(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    learning_rate: float,
):
    model = LegacyDemonstratorModel().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    plot_epochs = []
    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(x_train.to(DEVICE))
        loss = loss_fn(train_pred, y_train.to(DEVICE))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            val_pred = model(x_val.to(DEVICE))
            val_loss = loss_fn(val_pred, y_val.to(DEVICE))
            val_losses.append(val_loss.item())

        if epoch % 500 == 0:
            plot_epochs.append(epoch)
            avg_train = float(np.mean(train_losses[-500:]))
            avg_val = float(np.mean(val_losses[-500:]))
            avg_train_losses.append(avg_train)
            avg_val_losses.append(avg_val)
            print(f"Epoch {epoch}: Avg Train-Loss = {avg_train:.4f}, Avg Val-Loss = {avg_val:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_epochs, avg_train_losses, label="Average Train Loss")
    plt.plot(plot_epochs, avg_val_losses, label="Average Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def save_model(model: nn.Module, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def prepare_variant_data(
    variant: str,
    strains: list[list[float]],
    load_pos_x: list[float],
    load_pos_y: list[float],
    load_value: list[float],
    percentage: float,
    seed: int,
):
    if variant == "legacy":
        x_matrix = build_feature_matrix(normalize_strain_data_legacy(strains))
        return prepare_data_legacy(
            x_matrix=x_matrix,
            load_pos_x=load_pos_x,
            load_pos_y=load_pos_y,
            load_value=load_value,
            percentage=percentage,
            seed=seed,
        )

    x_matrix = build_feature_matrix(strains)
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data_legacy(
        x_matrix=x_matrix,
        load_pos_x=load_pos_x,
        load_pos_y=load_pos_y,
        load_value=load_value,
        percentage=percentage,
        seed=seed,
    )

    if variant == "raw":
        return x_train, x_val, x_test, y_train, y_val, y_test

    if variant == "channel_minmax":
        x_train_np = x_train.numpy()
        x_val_np = x_val.numpy()
        x_test_np = x_test.numpy()
        return (
            torch.tensor(minmax_scale_from_train(x_train_np, x_train_np), dtype=torch.float32),
            torch.tensor(minmax_scale_from_train(x_train_np, x_val_np), dtype=torch.float32),
            torch.tensor(minmax_scale_from_train(x_train_np, x_test_np), dtype=torch.float32),
            y_train,
            y_val,
            y_test,
        )

    raise ValueError(f"Unknown normalization variant: {variant}")


def run_experiment(
    variant: str,
    strains: list[list[float]],
    load_pos_x: list[float],
    load_pos_y: list[float],
    load_value: list[float],
    args,
):
    print("\n" + "=" * 80)
    print(f"Normalization mode: {variant}")
    print("=" * 80)

    x_train, x_val, x_test, y_train, y_val, y_test = prepare_variant_data(
        variant=variant,
        strains=strains,
        load_pos_x=load_pos_x,
        load_pos_y=load_pos_y,
        load_value=load_value,
        percentage=args.percentage,
        seed=args.seed,
    )

    model = train_model(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    metrics = test_model(model, x_test, y_test)
    _, y_sample, y_pred = test_random_samples(model, x_test, y_test, num_samples=args.num_samples)

    create_scatterplot_testing(
        load_pos_x=y_sample[:, 0].cpu().numpy(),
        load_pos_x_pred=y_pred[:, 0].cpu().numpy(),
        load_pos_y=y_sample[:, 1].cpu().numpy(),
        load_pos_y_pred=y_pred[:, 1].cpu().numpy(),
        load_value=y_sample[:, 2].cpu().numpy(),
        load_value_pred=y_pred[:, 2].cpu().numpy(),
        sensor_pos=SENSOR_POSITIONS,
    )

    if args.save_model:
        output_path = resolve_repo_root() / "models" / "final_models" / f"demonstrator_training_test_{variant}.pt"
        save_model(model, output_path)

    return metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    csv_path = resolve_csv_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    load_pos_x, load_pos_y, load_value, strains = load_legacy_style_data(csv_path)
    create_scatterplot(load_pos_x, load_pos_y, SENSOR_POSITIONS)

    variants = ["legacy", "raw", "channel_minmax"] if args.normalization == "compare" else [args.normalization]
    results: dict[str, dict[str, float]] = {}

    for variant in variants:
        results[variant] = run_experiment(
            variant=variant,
            strains=strains,
            load_pos_x=load_pos_x,
            load_pos_y=load_pos_y,
            load_value=load_value,
            args=args,
        )

    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        for variant, metrics in results.items():
            print(
                f"{variant:>14} | "
                f"MSE={metrics['mse']:.4f} | "
                f"MAE_X={metrics['mae_x']:.4f} | "
                f"MAE_Y={metrics['mae_y']:.4f} | "
                f"MAE_F={metrics['mae_f']:.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
