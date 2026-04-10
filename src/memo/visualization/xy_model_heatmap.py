from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from memo.acquisition.readers import Ads1115Reader, CsvReplayReader, MockReader, SerialSensorReader, UnavailableSensorReader
from memo.ml.inference import ModelPredictor
from memo.ml.models import MembraneModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_models" / "model_xy.pt"
DEFAULT_NORMALIZATION_CSV = PROJECT_ROOT / "data" / "recorded_samples" / "3D_Messung_01_10N.csv"
SENSOR_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]
MEMBRANE_SIDE_LENGTH = 450.0
MEMBRANE_DIAGONAL = MEMBRANE_SIDE_LENGTH * np.sqrt(2.0)
OFFSET = 50.0
GRID_SPACING = 40.0
X_LIMITS = (-MEMBRANE_DIAGONAL / 2.0 - OFFSET, MEMBRANE_DIAGONAL / 2.0 + OFFSET)
Y_LIMITS = (-MEMBRANE_DIAGONAL / 2.0 - OFFSET, MEMBRANE_DIAGONAL / 2.0 + OFFSET)
UPDATE_MS = 60
HEATMAP_SIGMA_MM = 42.0
HEATMAP_INTENSITY = 1.0
TEMPORAL_SMOOTHING_ALPHA = 0.22
GRID_RESOLUTION = 220


def _load_normalization_stats(csv_path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if csv_path is None or not csv_path.exists():
        return None, None

    import pandas as pd

    data = pd.read_csv(csv_path)
    missing_columns = [column for column in SENSOR_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV fuer Normalisierung fehlt Sensor-Spalten: {', '.join(missing_columns)}")
    sensor_data = data[SENSOR_COLUMNS]
    return sensor_data.min().to_numpy(dtype=float), sensor_data.max().to_numpy(dtype=float)


def build_reader(replay_csv: str | None, sensor_port: str | None, sensor_baudrate: int, sensor_timeout: float, use_mock: bool):
    if use_mock:
        return MockReader(seed=7, baseline=2.5, noise_std=0.08)
    if replay_csv:
        return CsvReplayReader(replay_csv, loop=True)
    if sensor_port:
        return SerialSensorReader(
            port=sensor_port,
            baudrate=sensor_baudrate,
            timeout=sensor_timeout,
        )
    try:
        return Ads1115Reader()
    except RuntimeError as exc:
        return UnavailableSensorReader(
            reason=(
                "ADS1115 ist in dieser Umgebung nicht verfuegbar. "
                "Bitte mit --mock, --replay-csv oder --sensor-port starten. "
                f"Details: {exc}"
            )
        )


class XYModelHeatmapPlot(QWidget):
    def __init__(
        self,
        x_limits=X_LIMITS,
        y_limits=Y_LIMITS,
        grid_spacing: float = GRID_SPACING,
        membrane_size: float = MEMBRANE_DIAGONAL,
        resolution: int = GRID_RESOLUTION,
        parent=None,
    ):
        super().__init__(parent)
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.grid_spacing = float(grid_spacing)
        self.membrane_size = float(membrane_size)
        self.resolution = int(resolution)
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self._x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.resolution)
        self._y_values = np.linspace(self.y_limits[0], self.y_limits[1], self.resolution)
        self._xx, self._yy = np.meshgrid(self._x_values, self._y_values)
        self._half_size = self.membrane_size / 2.0
        self._inside_membrane = (np.abs(self._xx) + np.abs(self._yy)) <= self._half_size
        self._build_layout()
        self._initialize_plot()

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _grid_ticks(self, limits):
        start = np.ceil(limits[0] / self.grid_spacing) * self.grid_spacing
        stop = np.floor(limits[1] / self.grid_spacing) * self.grid_spacing
        ticks = np.arange(start, stop + (self.grid_spacing * 0.5), self.grid_spacing)
        if limits[0] <= 0 <= limits[1] and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))
        return ticks

    def _membrane_vertices(self) -> list[tuple[float, float]]:
        return [
            (0.0, self._half_size),
            (self._half_size, 0.0),
            (0.0, -self._half_size),
            (-self._half_size, 0.0),
        ]

    def _eyelet_positions(self) -> list[tuple[float, float]]:
        ring_scale = 0.9
        return [(corner_x * ring_scale, corner_y * ring_scale) for corner_x, corner_y in self._membrane_vertices()]

    def _draw_sensors(self):
        sensor_width = 18.0
        sensor_height = 33.0
        sensor_definitions = [
            {"name": "R1", "x": -30.0, "y": 200.0, "rotation": 0.0},
            {"name": "R2", "x": 45.0, "y": 182.0, "rotation": 90.0},
            {"name": "R3", "x": -200.0, "y": 30.0, "rotation": 90.0},
            {"name": "R4", "x": -182.0, "y": -45.0, "rotation": 0.0},
            {"name": "R5", "x": 182.0, "y": 45.0, "rotation": 0.0},
            {"name": "R6", "x": 200.0, "y": -30.0, "rotation": 90.0},
            {"name": "R7", "x": -45.0, "y": -182.0, "rotation": 90.0},
            {"name": "R8", "x": 30.0, "y": -200.0, "rotation": 0.0},
        ]
        for sensor_data in sensor_definitions:
            center_x = float(sensor_data["x"])
            center_y = float(sensor_data["y"])
            rotation = float(sensor_data["rotation"])
            sensor = Rectangle(
                (center_x - sensor_width / 2.0, center_y - sensor_height / 2.0),
                sensor_width,
                sensor_height,
                facecolor="#f5a623",
                edgecolor="#7a4d00",
                alpha=0.82,
                linewidth=1.0,
                zorder=3,
            )
            sensor.set_transform(Affine2D().rotate_deg_around(center_x, center_y, rotation) + self.axis.transData)
            self.axis.add_patch(sensor)
            self.axis.text(center_x, center_y, str(sensor_data["name"]), ha="center", va="center", fontsize=7, zorder=4)

    def _build_field(self, center_x: float, center_y: float, sigma: float, intensity: float):
        sigma = max(float(sigma), 1.0)
        intensity = max(float(intensity), 0.0)
        distance_sq = (self._xx - center_x) ** 2 + (self._yy - center_y) ** 2
        field = intensity * np.exp(-distance_sq / (2.0 * sigma**2))
        return np.where(self._inside_membrane, field, np.nan)

    def _initialize_plot(self):
        self.axis.clear()
        self.axis.set_title("XY Heatmap Vorhersage", pad=18)
        self.axis.set_xlim(*self.x_limits)
        self.axis.set_ylim(*self.y_limits)
        self.axis.set_xlabel("X [mm]")
        self.axis.set_ylabel("Y [mm]")
        self.axis.set_xticks(self._grid_ticks(self.x_limits))
        self.axis.set_yticks(self._grid_ticks(self.y_limits))
        self.axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.28)
        self.axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.25, zorder=1)
        self.axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.25, zorder=1)
        self.axis.set_aspect("equal", adjustable="box")

        membrane = Polygon(
            self._membrane_vertices(),
            closed=True,
            facecolor="#4a90e2",
            edgecolor="#1f4f82",
            alpha=0.10,
            linewidth=1.8,
            zorder=0,
        )
        self.axis.add_patch(membrane)
        self._draw_sensors()

        monitor = Rectangle(
            (-75.0, self._half_size + 50.0),
            150.0,
            45.0,
            facecolor="#c9d3df",
            edgecolor="#4b5563",
            linewidth=1.3,
            alpha=0.95,
            clip_on=False,
            zorder=2,
        )
        self.axis.add_patch(monitor)

        for corner_x, corner_y in self._eyelet_positions():
            self.axis.add_patch(
                Circle(
                    (corner_x, corner_y),
                    radius=7.0,
                    facecolor="white",
                    edgecolor="#203548",
                    linewidth=1.6,
                    zorder=4,
                )
            )

        empty = np.full((self.resolution, self.resolution), np.nan, dtype=float)
        self.image = self.axis.imshow(
            empty,
            origin="lower",
            extent=(self.x_limits[0], self.x_limits[1], self.y_limits[0], self.y_limits[1]),
            cmap="turbo",
            interpolation="bilinear",
            alpha=0.88,
            vmin=0.0,
            vmax=1.0,
            zorder=2,
        )
        self.prediction_marker = self.axis.scatter([], [], s=90, color="#b42318", edgecolors="white", linewidths=1.2, zorder=5)
        self.prediction_text = self.axis.text(
            0.02,
            0.98,
            "",
            transform=self.axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#1f2a37",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.92},
            zorder=6,
        )
        self.canvas.draw_idle()

    def update_prediction(self, x_value: float, y_value: float, sigma: float = HEATMAP_SIGMA_MM, intensity: float = HEATMAP_INTENSITY):
        x_value = float(np.clip(x_value, self.x_limits[0], self.x_limits[1]))
        y_value = float(np.clip(y_value, self.y_limits[0], self.y_limits[1]))
        field = self._build_field(x_value, y_value, sigma=sigma, intensity=intensity)
        self.image.set_data(field)
        self.prediction_marker.set_offsets(np.array([[x_value, y_value]], dtype=float))
        self.prediction_text.set_text(f"Predicted X={x_value:.1f} mm\nPredicted Y={y_value:.1f} mm")
        self.canvas.draw_idle()


class XYModelHeatmapWindow(QMainWindow):
    def __init__(self, reader, predictor: ModelPredictor, update_ms: int = UPDATE_MS):
        super().__init__()
        self.reader = reader
        self.predictor = predictor
        self.update_ms = int(update_ms)
        self._smoothed_xy: np.ndarray | None = None
        self._last_error: str | None = None

        self.setWindowTitle("MeMo XY Model Heatmap")
        self.resize(1500, 980)

        self.plot = XYModelHeatmapPlot()
        self.reader_label = QLabel(type(reader).__name__)
        self.model_label = QLabel(Path(self.predictor.model_path).name)
        self.position_label = QLabel("-")
        self.sensor_label = QLabel("-")
        self.status_label = QLabel("Bereit")

        self.timer = QTimer(self)
        self.timer.setInterval(self.update_ms)
        self.timer.timeout.connect(self.refresh_prediction)

        self._build_ui()
        self._apply_styles()
        self.refresh_prediction()
        self.timer.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        layout.addWidget(self.plot, stretch=1)

        info_card = QWidget()
        info_card.setObjectName("infoCard")
        info_layout = QGridLayout(info_card)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setHorizontalSpacing(24)
        info_layout.setVerticalSpacing(10)
        info_layout.addWidget(self._caption("Reader"), 0, 0)
        info_layout.addWidget(self.reader_label, 0, 1)
        info_layout.addWidget(self._caption("Model"), 0, 2)
        info_layout.addWidget(self.model_label, 0, 3)
        info_layout.addWidget(self._caption("Position"), 1, 0)
        info_layout.addWidget(self.position_label, 1, 1)
        info_layout.addWidget(self._caption("Sensoren"), 1, 2)
        info_layout.addWidget(self.sensor_label, 1, 3)
        info_layout.addWidget(self._caption("Status"), 2, 0)
        info_layout.addWidget(self.status_label, 2, 1, 1, 3)
        layout.addWidget(info_card, stretch=0)

    def _caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("captionLabel")
        return label

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f3f5f7;
            }
            QWidget#infoCard {
                background: #ffffff;
                border: 1px solid #d9e0e6;
                border-radius: 12px;
            }
            QLabel {
                color: #1f2a37;
                font-size: 14px;
            }
            QLabel#captionLabel {
                color: #5b6470;
                font-weight: 700;
            }
            """
        )

    def _smooth_prediction(self, xy_prediction: np.ndarray) -> np.ndarray:
        xy_prediction = np.asarray(xy_prediction, dtype=float).reshape(-1)
        if self._smoothed_xy is None:
            self._smoothed_xy = xy_prediction.copy()
        else:
            alpha = TEMPORAL_SMOOTHING_ALPHA
            self._smoothed_xy = ((1.0 - alpha) * self._smoothed_xy) + (alpha * xy_prediction)
        return self._smoothed_xy

    def refresh_prediction(self):
        try:
            frame = self.reader.read()
            sensor_values = np.asarray(frame.sensors, dtype=float)
            prediction = np.asarray(self.predictor.predict(sensor_values), dtype=float).reshape(-1)
            if prediction.size < 2:
                raise ValueError("Das XY-Modell liefert weniger als zwei Ausgabewerte.")
            smoothed_xy = self._smooth_prediction(prediction[:2])
            self.plot.update_prediction(smoothed_xy[0], smoothed_xy[1])
            self.position_label.setText(f"X={smoothed_xy[0]:.1f} mm, Y={smoothed_xy[1]:.1f} mm")
            self.sensor_label.setText(", ".join(f"{value:.3f}" for value in sensor_values[:8]))
            self.status_label.setText(f"Live-Vorhersage aktiv ({frame.source})")
            self._last_error = None
        except StopIteration:
            self.timer.stop()
            self.status_label.setText("Replay beendet")
        except Exception as exc:
            message = str(exc)
            if message != self._last_error:
                self.status_label.setText(f"Fehler: {message}")
                self._last_error = message

    def closeEvent(self, event: QCloseEvent):
        self.timer.stop()
        close_method = getattr(self.reader, "close", None)
        if callable(close_method):
            close_method()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MeMo XY Heatmap auf Basis des trainierten XY-Modells")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Pfad zu model_xy.pt")
    parser.add_argument(
        "--normalization-csv",
        default=str(DEFAULT_NORMALIZATION_CSV),
        help="CSV fuer Min/Max-Normalisierung der acht Sensorspalten",
    )
    parser.add_argument("--replay-csv", default=None, help="CSV-Replay statt Live-Hardware")
    parser.add_argument("--sensor-port", default=None, help="Serieller Port fuer 8 Sensorsignale")
    parser.add_argument("--sensor-baudrate", type=int, default=57600, help="Baudrate des seriellen Sensorreaders")
    parser.add_argument("--sensor-timeout", type=float, default=0.2, help="Timeout des seriellen Sensorreaders")
    parser.add_argument("--mock", action="store_true", help="Verwendet Mock-Sensordaten statt echter Hardware")
    parser.add_argument("--cpu", action="store_true", help="Erzwingt Vorhersage auf CPU")
    parser.add_argument("--update-ms", type=int, default=UPDATE_MS, help="UI-Updateintervall in Millisekunden")
    return parser.parse_args(argv)


def run_xy_model_heatmap(argv=None):
    args = parse_args(argv)
    mins, maxs = _load_normalization_stats(Path(args.normalization_csv) if args.normalization_csv else None)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    predictor = ModelPredictor(
        model_class=MembraneModel,
        model_path=str(Path(args.model_path)),
        output_dim=2,
        device=device,
        mins=mins,
        maxs=maxs,
    )
    reader = build_reader(
        replay_csv=args.replay_csv,
        sensor_port=args.sensor_port,
        sensor_baudrate=args.sensor_baudrate,
        sensor_timeout=args.sensor_timeout,
        use_mock=args.mock,
    )
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = XYModelHeatmapWindow(reader=reader, predictor=predictor, update_ms=args.update_ms)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(run_xy_model_heatmap())
