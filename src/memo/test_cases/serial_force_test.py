from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QProgressBar, QVBoxLayout, QWidget

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

from memo.acquisition.readers import SerialException, SerialForceReader


def to_utc_timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).timestamp()
    return value.astimezone(UTC).timestamp()


class ForcePlotWidget(QWidget):
    def __init__(self, history_seconds: float = 2.0, parent=None):
        super().__init__(parent)
        self.history_seconds = history_seconds
        self.timestamps: deque[datetime] = deque()
        self.values: deque[float] = deque()

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("left", "Kraft [N]")
        self.plot_widget.getAxis("bottom").setStyle(showValues=False, tickLength=0)
        self.plot_widget.setXRange(-self.history_seconds / 2.0, self.history_seconds / 2.0, padding=0.0)
        self.plot_widget.setYRange(0.0, 25.0, padding=0.05)
        self.plot_widget.getPlotItem().vb.setLimits(
            xMin=-self.history_seconds / 2.0,
            xMax=self.history_seconds / 2.0,
        )

        pen = pg.mkPen(color="#5f6b7a", width=2)
        self.curve = self.plot_widget.plot([], [], pen=pen)
        self.marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#0f8b6d"), pen=pg.mkPen("#0f8b6d"))
        self.plot_widget.addItem(self.marker)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)

    def append_value(self, timestamp: datetime, value: float):
        self.timestamps.append(timestamp)
        self.values.append(float(value))

        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.values.popleft()

        self._update_plot()

    def _update_plot(self):
        if not self.timestamps:
            self.curve.setData([], [])
            self.marker.setData([], [])
            return

        latest_ts = self.timestamps[-1].timestamp()
        x_values = np.array([ts.timestamp() - latest_ts for ts in self.timestamps], dtype=float)
        x_values += self.history_seconds / 2.0
        x_values -= self.history_seconds / 2.0
        y_values = np.array(self.values, dtype=float)

        self.curve.setData(x_values, y_values)
        self.marker.setData([0.0], [y_values[-1]])


class SerialForceTestWindow(QMainWindow):
    def __init__(self, port: str, baudrate: int, timeout: float, refresh_ms: int):
        super().__init__()
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")

        self.reader = SerialForceReader(port=port, baudrate=baudrate, timeout=timeout)
        self.refresh_ms = refresh_ms
        self._last_sample_id = 0
        self._display_history: deque[float] = deque(maxlen=5)
        self._display_value: float | None = None

        self.setWindowTitle("MeMo Serial Force Test")
        self.resize(1000, 620)

        self.force_value_label = QLabel("Kraft: -")
        self.status_label = QLabel("Status: Verbinde...")
        self.force_bar = QProgressBar()
        self.force_plot = ForcePlotWidget(history_seconds=2.0)
        self.exit_button = QPushButton("Exit")

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._refresh)
        self.exit_button.clicked.connect(self.close)

        self._build_ui()
        self._apply_styles()
        self._start_reader()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self.force_value_label)
        layout.addWidget(self.force_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.force_plot, stretch=1)
        layout.addWidget(self.exit_button)

    def _apply_styles(self):
        self.force_value_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.status_label.setStyleSheet("font-size: 16px; color: #444444;")
        self.exit_button.setFixedHeight(36)
        self.force_bar.setRange(0, 2500)
        self.force_bar.setValue(0)
        self.force_bar.setFormat("0.000 N")
        self.force_bar.setTextVisible(True)
        self.force_bar.setFixedHeight(26)
        self.force_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #b8c0cc;
                border-radius: 6px;
                background: #f3f6fa;
                text-align: center;
                font-size: 14px;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background-color: #0f8b6d;
                border-radius: 5px;
            }
            """
        )

    def _start_reader(self):
        try:
            self.reader.start()
        except (RuntimeError, SerialException, OSError) as exc:
            self.status_label.setText(f"Status: Verbindung fehlgeschlagen ({exc})")
            return

        self.status_label.setText(
            f"Status: Verbunden mit {self.reader.port} @ {self.reader.baudrate}"
        )
        self.timer.start(self.refresh_ms)

    def _filter_display_force(self, value: float) -> float:
        raw_value = float(value)

        if len(self._display_history) >= 3:
            history = np.asarray(self._display_history, dtype=float)
            median = float(np.median(history))
            mad = float(np.median(np.abs(history - median)))
            deviation_limit = max(0.20, 6.0 * mad)
            if abs(raw_value - median) > deviation_limit:
                raw_value = median + np.sign(raw_value - median) * deviation_limit

        self._display_history.append(raw_value)

        if self._display_value is None:
            self._display_value = raw_value
        else:
            alpha = 0.35
            self._display_value = ((1.0 - alpha) * self._display_value) + (alpha * raw_value)

        return float(self._display_value)

    def _refresh(self):
        now = datetime.now(UTC)
        now_ts = now.timestamp()
        error_text = self.reader.get_last_error()
        if error_text:
            self.status_label.setText(f"Status: Fehler bei serieller Verbindung ({error_text})")
            return

        new_samples = self.reader.get_samples_since(self._last_sample_id)
        if new_samples:
            self._last_sample_id = new_samples[-1][0]

        force_value = self.reader.get_latest_force()
        force_timestamp = self.reader.get_latest_force_timestamp()
        force_timestamp_ts = to_utc_timestamp(force_timestamp)
        stream_active = (
            force_timestamp_ts is not None
            and (now_ts - force_timestamp_ts) <= max(0.2, self.refresh_ms / 1000.0 * 3.0)
        )

        if force_value is None or not stream_active:
            self.force_value_label.setText("Kraft: -")
            self.force_bar.setValue(0)
            self.force_bar.setFormat("-.--- N")
            self.status_label.setText(f"Status: Warte auf Live-Daten auf {self.reader.port} @ {self.reader.baudrate}")
            return

        display_value = force_value
        for _, sample_timestamp, sample_value in new_samples:
            filtered_sample_value = self._filter_display_force(sample_value)
            self.force_plot.append_value(sample_timestamp, filtered_sample_value)
            display_value = filtered_sample_value

        if not new_samples:
            display_value = self._display_value if self._display_value is not None else force_value

        self.force_value_label.setText(f"Kraft: {display_value:.3f} N")
        force_bar_value = max(0, min(int(round(display_value * 100)), self.force_bar.maximum()))
        self.force_bar.setValue(force_bar_value)
        self.force_bar.setFormat(f"{display_value:.3f} N")
        self.status_label.setText(
            f"Status: Live-Daten aktiv auf {self.reader.port} @ {self.reader.baudrate}"
        )

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()
        self.reader.close()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live test window for serial force measurement")
    parser.add_argument("--port", default="COM3", help="Serial port of the force measurement device")
    parser.add_argument("--baudrate", type=int, default=57600, help="Baudrate of the force measurement device")
    parser.add_argument("--timeout", type=float, default=0.01, help="Serial read timeout in seconds")
    parser.add_argument("--refresh-ms", type=int, default=8, help="UI refresh interval in milliseconds")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if pg is None:
        print("pyqtgraph is not installed. Please install it first.")
        return 1

    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = SerialForceTestWindow(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        refresh_ms=args.refresh_ms,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
