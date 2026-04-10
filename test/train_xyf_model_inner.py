from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "recorded_samples" / "3D_Messung_01_10N.csv",
    ROOT / "data" / "recorded_samples" / "3D_Messung_02_15N.csv",
]
FEATURE_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]
XY_COLUMNS = ["X", "Y"]
F_COLUMN = "F"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
TOLERANCES_MM = [10, 20, 40, 80]
MAX_ABS_COORD_MM = 120.0


@dataclass
class RegressionMetrics:
    mae: dict[str, float]
    rmse: dict[str, float]
    tolerance_accuracy_xy: dict[str, float]
    euclidean_accuracy_xy: dict[str, float]
    test_loss: float


def load_dataset() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in DATA_FILES]
    data = pd.concat(frames, ignore_index=True)
    inside_mask = data[XY_COLUMNS].abs().max(axis=1) <= MAX_ABS_COORD_MM
    filtered = data.loc[inside_mask].reset_index(drop=True)
    filtered["xy_group"] = filtered[XY_COLUMNS].astype(str).agg("_".join, axis=1)
    return filtered


def make_xy_features(data: pd.DataFrame) -> pd.DataFrame:
    sensors = data[FEATURE_COLUMNS].to_numpy()
    eps = 1e-9
    total = sensors.sum(axis=1, keepdims=True)
    normalized = sensors / (total + eps)
    centered = sensors - sensors.mean(axis=1, keepdims=True)

    features = pd.DataFrame(index=data.index)
    for idx in range(8):
        features[f"norm_r{idx + 1}"] = normalized[:, idx]
        features[f"center_r{idx + 1}"] = centered[:, idx]

    features["sensor_sum"] = total.ravel()
    features["sensor_std"] = sensors.std(axis=1)
    features["sensor_range"] = sensors.max(axis=1) - sensors.min(axis=1)

    for idx in range(8):
        next_idx = (idx + 1) % 8
        features[f"adj_diff_{idx + 1}_{next_idx + 1}"] = sensors[:, idx] - sensors[:, next_idx]

    for left, right in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        features[f"opp_diff_{left + 1}_{right + 1}"] = sensors[:, left] - sensors[:, right]

    return features


def snap_to_known_grid(predictions: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    snapped = predictions.copy()
    snapped[:, 0] = np.round(snapped[:, 0] / 40.0) * 40.0
    snapped[:, 1] = np.round(snapped[:, 1] / 40.0) * 40.0
    valid_positions = {tuple(position) for position in candidates.tolist()}

    for idx, point in enumerate(snapped):
        if tuple(point) in valid_positions:
            continue
        distances = np.sum((candidates - point) ** 2, axis=1)
        snapped[idx] = candidates[np.argmin(distances)]

    return snapped


def split_by_position(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(splitter.split(data, groups=data["xy_group"]))
    train_val = data.iloc[train_val_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(val_splitter.split(train_val, groups=train_val["xy_group"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    ranges = np.ptp(y_true, axis=0)
    safe_ranges = np.where(ranges == 0.0, 1.0, ranges)
    return float(np.mean(rmse / safe_ranges))


def evaluate_models(
    xy_model: Pipeline,
    f_model: ExtraTreesRegressor,
    x_xy_test: pd.DataFrame,
    x_f_test: pd.DataFrame,
    y_xy_test: np.ndarray,
    y_f_test: np.ndarray,
    grid_candidates: np.ndarray,
) -> tuple[RegressionMetrics, np.ndarray]:
    xy_pred = np.asarray(xy_model.predict(x_xy_test), dtype=float)
    xy_pred = snap_to_known_grid(xy_pred, grid_candidates)
    f_pred = f_model.predict(x_f_test).reshape(-1, 1)

    combined_pred = np.column_stack([xy_pred, f_pred.ravel()])
    combined_true = np.column_stack([y_xy_test, y_f_test.ravel()])

    mae_values = mean_absolute_error(combined_true, combined_pred, multioutput="raw_values")
    rmse_values = np.sqrt(mean_squared_error(combined_true, combined_pred, multioutput="raw_values"))

    tolerance_accuracy = {}
    euclidean_accuracy = {}
    for tolerance in TOLERANCES_MM:
        axis_hit = (
            (np.abs(xy_pred[:, 0] - y_xy_test[:, 0]) <= tolerance)
            & (np.abs(xy_pred[:, 1] - y_xy_test[:, 1]) <= tolerance)
        )
        euclidean_hit = np.linalg.norm(xy_pred - y_xy_test, axis=1) <= tolerance
        tolerance_accuracy[f"+/-{tolerance}mm"] = float(axis_hit.mean())
        euclidean_accuracy[f"{tolerance}mm_radius"] = float(euclidean_hit.mean())

    metrics = RegressionMetrics(
        mae={
            "X_mm": float(mae_values[0]),
            "Y_mm": float(mae_values[1]),
            "F_N": float(mae_values[2]),
        },
        rmse={
            "X_mm": float(rmse_values[0]),
            "Y_mm": float(rmse_values[1]),
            "F_N": float(rmse_values[2]),
        },
        tolerance_accuracy_xy=tolerance_accuracy,
        euclidean_accuracy_xy=euclidean_accuracy,
        test_loss=normalized_rmse(combined_true, combined_pred),
    )
    return metrics, combined_pred


def main() -> None:
    data = load_dataset()
    train, val, test = split_by_position(data)
    train_val = pd.concat([train, val], ignore_index=True)
    grid_candidates = data[XY_COLUMNS].drop_duplicates().to_numpy()

    xy_train_features = make_xy_features(train)
    xy_val_features = make_xy_features(val)
    xy_train_val_features = make_xy_features(train_val)
    xy_test_features = make_xy_features(test)

    y_xy_train = train[XY_COLUMNS].to_numpy()
    y_xy_val = val[XY_COLUMNS].to_numpy()
    y_xy_train_val = train_val[XY_COLUMNS].to_numpy()
    y_xy_test = test[XY_COLUMNS].to_numpy()

    f_train_features = train[FEATURE_COLUMNS].copy()
    f_val_features = val[FEATURE_COLUMNS].copy()
    f_train_val_features = train_val[FEATURE_COLUMNS].copy()
    f_test_features = test[FEATURE_COLUMNS].copy()

    y_f_train = train[F_COLUMN].to_numpy()
    y_f_val = val[F_COLUMN].to_numpy()
    y_f_train_val = train_val[F_COLUMN].to_numpy()
    y_f_test = test[F_COLUMN].to_numpy()

    xy_candidates = [
        (
            "extra_trees_snap",
            Pipeline([("regressor", ExtraTreesRegressor(n_estimators=1200, random_state=RANDOM_STATE))]),
        ),
        (
            "knn3_snap",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", KNeighborsRegressor(n_neighbors=3, weights="distance")),
                ]
            ),
        ),
    ]

    best_xy_name = ""
    best_xy_model = None
    best_xy_score = -1.0
    best_xy_val_loss = float("inf")
    for name, model in xy_candidates:
        model.fit(xy_train_features, y_xy_train)
        val_pred = np.asarray(model.predict(xy_val_features), dtype=float)
        val_pred = snap_to_known_grid(val_pred, grid_candidates)
        val_tol10 = float(
            np.mean(
                (np.abs(val_pred[:, 0] - y_xy_val[:, 0]) <= 10)
                & (np.abs(val_pred[:, 1] - y_xy_val[:, 1]) <= 10)
            )
        )
        val_loss = normalized_rmse(y_xy_val, val_pred)
        if val_tol10 > best_xy_score or (val_tol10 == best_xy_score and val_loss < best_xy_val_loss):
            best_xy_name = name
            best_xy_model = model
            best_xy_score = val_tol10
            best_xy_val_loss = val_loss

    assert best_xy_model is not None

    f_candidates = [
        ("extra_trees_600", ExtraTreesRegressor(n_estimators=600, random_state=RANDOM_STATE)),
        ("extra_trees_1200", ExtraTreesRegressor(n_estimators=1200, random_state=RANDOM_STATE)),
    ]

    best_f_name = ""
    best_f_model = None
    best_f_val_loss = float("inf")
    for name, model in f_candidates:
        model.fit(f_train_features, y_f_train)
        val_pred = model.predict(f_val_features).reshape(-1, 1)
        val_loss = normalized_rmse(y_f_val.reshape(-1, 1), val_pred)
        if val_loss < best_f_val_loss:
            best_f_name = name
            best_f_model = model
            best_f_val_loss = val_loss

    assert best_f_model is not None

    best_xy_model.fit(xy_train_val_features, y_xy_train_val)
    best_f_model.fit(f_train_val_features, y_f_train_val)

    metrics, predictions = evaluate_models(
        best_xy_model,
        best_f_model,
        xy_test_features,
        f_test_features,
        y_xy_test,
        y_f_test,
        grid_candidates,
    )

    output_dir = ROOT / "test" / "artifacts_inner"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_xy_model, output_dir / "xy_model.joblib")
    joblib.dump(best_f_model, output_dir / "f_model.joblib")

    prediction_frame = test[["timestamp", *XY_COLUMNS, F_COLUMN]].copy()
    prediction_frame["pred_X"] = predictions[:, 0]
    prediction_frame["pred_Y"] = predictions[:, 1]
    prediction_frame["pred_F"] = predictions[:, 2]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    results = {
        "dataset": {
            "files": [str(path) for path in DATA_FILES],
            "max_abs_coord_mm": MAX_ABS_COORD_MM,
            "n_samples_total": int(len(data)),
            "n_unique_positions": int(data["xy_group"].nunique()),
            "split": {
                "train_samples": int(len(train)),
                "val_samples": int(len(val)),
                "test_samples": int(len(test)),
            },
        },
        "models": {
            "xy_model": {
                "type": best_xy_name,
                "validation_tol10_accuracy": best_xy_score,
                "validation_loss": best_xy_val_loss,
                "features": "interior-only, normalized + centered sensor features + ring differences + 40mm grid snap",
            },
            "f_model": {
                "type": best_f_name,
                "validation_loss": best_f_val_loss,
                "features": "raw sensor values on interior points only",
            },
        },
        "metrics": asdict(metrics),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
