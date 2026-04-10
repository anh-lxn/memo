from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


DEFAULT_CSV_PATH = Path(r"C:\Users\anh\Documents\Repositories\memo\data\recorded_samples\3D_Messung_02_15N.csv")
SENSOR_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]
BASELINE_VOLTAGE = 2.5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Erzeugt 2D-Heatmaps pro Sensor relativ zu einer Baseline-Spannung."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Pfad zur CSV-Datei mit X/Y/F und Sensor R1..R8.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=BASELINE_VOLTAGE,
        help="Baseline-Spannung in Volt, gegen die die Heatmaps gebildet werden.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optionaler Ausgabepfad fuer das Bild, z. B. figs/sensor_baseline_heatmaps.png",
    )
    return parser.parse_args(argv)


def load_measurements(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required_columns = {"X", "Y", *SENSOR_COLUMNS}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"CSV fehlt benoetigte Spalten: {', '.join(missing_columns)}")
    return data


def build_sensor_grid(data: pd.DataFrame, sensor_column: str, baseline: float) -> pd.DataFrame:
    sensor_data = data[["X", "Y", sensor_column]].copy()
    sensor_data["delta_to_baseline"] = sensor_data[sensor_column] - baseline
    heatmap_grid = sensor_data.pivot_table(
        index="Y",
        columns="X",
        values="delta_to_baseline",
        aggfunc="mean",
    )
    return heatmap_grid.sort_index().sort_index(axis=1)


def plot_sensor_heatmaps(data: pd.DataFrame, baseline: float, save_path: Path | None = None):
    grids = {sensor_column: build_sensor_grid(data, sensor_column, baseline) for sensor_column in SENSOR_COLUMNS}

    global_max_abs = max(
        float(np.nanmax(np.abs(grid.to_numpy(dtype=float))))
        for grid in grids.values()
    )
    if global_max_abs == 0.0:
        global_max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-global_max_abs, vcenter=0.0, vmax=global_max_abs)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(f"2D Heatmaps pro Sensor relativ zur Baseline {baseline:.2f} V", fontsize=16)

    image = None
    for axis, sensor_column in zip(axes.flat, SENSOR_COLUMNS):
        grid = grids[sensor_column]
        image = axis.imshow(
            grid.to_numpy(dtype=float),
            origin="lower",
            cmap="coolwarm",
            norm=norm,
            extent=[
                float(grid.columns.min()),
                float(grid.columns.max()),
                float(grid.index.min()),
                float(grid.index.max()),
            ],
            aspect="equal",
        )
        axis.set_title(sensor_column)
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
        axis.grid(False)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
        colorbar.set_label("Sensorwert - Baseline [V]")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Heatmap gespeichert unter: {save_path}")

    plt.show()


def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    save_path = Path(args.save) if args.save else None
    data = load_measurements(csv_path)
    plot_sensor_heatmaps(data=data, baseline=float(args.baseline), save_path=save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
