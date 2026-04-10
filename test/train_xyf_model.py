from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid


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
    data["xy_group"] = data[XY_COLUMNS].astype(str).agg("_".join, axis=1)
    return data


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


def train_best_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: list[dict[str, int | None]],
) -> tuple[ExtraTreesRegressor, dict[str, int | None], float]:
    best_model = None
    best_params = None
    best_score = float("inf")

    for params in ParameterGrid(param_grid):
        model = ExtraTreesRegressor(random_state=RANDOM_STATE, **params)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        y_val_2d = y_val if y_val.ndim > 1 else y_val.reshape(-1, 1)
        pred_2d = pred if pred.ndim > 1 else pred.reshape(-1, 1)
        score = normalized_rmse(y_val_2d, pred_2d)
        if score < best_score:
            best_model = model
            best_params = params
            best_score = score

    assert best_model is not None
    assert best_params is not None
    return best_model, best_params, best_score


def evaluate_models(
    xy_model: ExtraTreesRegressor,
    f_model: ExtraTreesRegressor,
    x_test: np.ndarray,
    y_xy_test: np.ndarray,
    y_f_test: np.ndarray,
) -> tuple[RegressionMetrics, np.ndarray]:
    xy_pred = xy_model.predict(x_test)
    f_pred = f_model.predict(x_test).reshape(-1, 1)
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

    x_train = train[FEATURE_COLUMNS].to_numpy()
    x_val = val[FEATURE_COLUMNS].to_numpy()
    x_train_val = pd.concat([train, val], ignore_index=True)[FEATURE_COLUMNS].to_numpy()
    x_test = test[FEATURE_COLUMNS].to_numpy()

    y_xy_train = train[XY_COLUMNS].to_numpy()
    y_xy_val = val[XY_COLUMNS].to_numpy()
    y_xy_train_val = pd.concat([train, val], ignore_index=True)[XY_COLUMNS].to_numpy()
    y_xy_test = test[XY_COLUMNS].to_numpy()

    y_f_train = train[F_COLUMN].to_numpy()
    y_f_val = val[F_COLUMN].to_numpy()
    y_f_train_val = pd.concat([train, val], ignore_index=True)[F_COLUMN].to_numpy()
    y_f_test = test[F_COLUMN].to_numpy()

    param_grid = [
        {
            "n_estimators": [300, 600, 1000],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 2],
        }
    ]

    _, best_xy_params, best_xy_val_loss = train_best_model(x_train, y_xy_train, x_val, y_xy_val, param_grid)
    _, best_f_params, best_f_val_loss = train_best_model(
        x_train,
        y_f_train,
        x_val,
        y_f_val,
        param_grid,
    )

    xy_model = ExtraTreesRegressor(random_state=RANDOM_STATE, **best_xy_params)
    f_model = ExtraTreesRegressor(random_state=RANDOM_STATE, **best_f_params)
    xy_model.fit(x_train_val, y_xy_train_val)
    f_model.fit(x_train_val, y_f_train_val)

    metrics, predictions = evaluate_models(xy_model, f_model, x_test, y_xy_test, y_f_test)

    output_dir = ROOT / "test" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xy_model, output_dir / "xy_extra_trees.joblib")
    joblib.dump(f_model, output_dir / "f_extra_trees.joblib")

    prediction_frame = test[["timestamp", *XY_COLUMNS, F_COLUMN]].copy()
    prediction_frame["pred_X"] = predictions[:, 0]
    prediction_frame["pred_Y"] = predictions[:, 1]
    prediction_frame["pred_F"] = predictions[:, 2]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    results = {
        "dataset": {
            "files": [str(path) for path in DATA_FILES],
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
                "type": "ExtraTreesRegressor",
                "best_params": best_xy_params,
                "validation_loss": best_xy_val_loss,
            },
            "f_model": {
                "type": "ExtraTreesRegressor",
                "best_params": best_f_params,
                "validation_loss": best_f_val_loss,
            },
        },
        "metrics": asdict(metrics),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
