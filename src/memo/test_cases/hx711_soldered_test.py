from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    import RPi.GPIO as GPIO
except ImportError:  # pragma: no cover - only available on Raspberry Pi
    GPIO = None


@dataclass
class Hx711Sample:
    raw_value: int
    value: float
    unit_label: str
    timestamp_monotonic: float


class Hx711Reader:
    """Minimal HX711 reader for direct Raspberry Pi hardware tests."""

    _GAIN_PULSES = {
        128: 1,
        64: 3,
        32: 2,
    }

    def __init__(
        self,
        data_pin: int,
        clock_pin: int,
        gain: int = 128,
        read_timeout: float = 1.0,
        stable_reads: int = 5,
        calibration_factor: float = 1.0,
        offset: float = 0.0,
        pulse_delay: float = 0.00002,
        use_bcm: bool = True,
    ):
        if GPIO is None:
            raise RuntimeError(
                "RPi.GPIO ist nicht installiert. Bitte auf dem Raspberry Pi installieren."
            )
        if gain not in self._GAIN_PULSES:
            raise ValueError("gain muss 128, 64 oder 32 sein.")
        if stable_reads < 1:
            raise ValueError("stable_reads muss mindestens 1 sein.")
        if calibration_factor == 0:
            raise ValueError("calibration_factor darf nicht 0 sein.")

        self.data_pin = int(data_pin)
        self.clock_pin = int(clock_pin)
        self.gain = int(gain)
        self.read_timeout = float(read_timeout)
        self.stable_reads = int(stable_reads)
        self.calibration_factor = float(calibration_factor)
        self.offset = float(offset)
        self.pulse_delay = float(pulse_delay)
        self.use_bcm = bool(use_bcm)

        self._is_setup = False

    def setup(self):
        if self._is_setup:
            return

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM if self.use_bcm else GPIO.BOARD)
        GPIO.setup(self.clock_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.data_pin, GPIO.IN)
        self._is_setup = True

        # First read configures the requested gain/channel combination.
        self.read_raw()

    def close(self):
        if not self._is_setup:
            return
        GPIO.cleanup((self.clock_pin, self.data_pin))
        self._is_setup = False

    def wait_until_ready(self):
        deadline = time.monotonic() + self.read_timeout
        while GPIO.input(self.data_pin) == GPIO.HIGH:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"HX711 nicht bereit auf DOUT pin {self.data_pin} innerhalb von {self.read_timeout:.2f}s."
                )
            time.sleep(0.001)

    def read_raw(self) -> int:
        self.setup()
        self.wait_until_ready()

        value = 0
        for _ in range(24):
            GPIO.output(self.clock_pin, GPIO.HIGH)
            time.sleep(self.pulse_delay)
            value = (value << 1) | int(GPIO.input(self.data_pin))
            GPIO.output(self.clock_pin, GPIO.LOW)
            time.sleep(self.pulse_delay)

        for _ in range(self._GAIN_PULSES[self.gain]):
            GPIO.output(self.clock_pin, GPIO.HIGH)
            time.sleep(self.pulse_delay)
            GPIO.output(self.clock_pin, GPIO.LOW)
            time.sleep(self.pulse_delay)

        if value & 0x800000:
            value -= 1 << 24

        return int(value)

    def read_raw_average(self) -> int:
        values = [self.read_raw() for _ in range(self.stable_reads)]
        return int(round(statistics.median(values)))

    def read(self, unit_label: str = "counts") -> Hx711Sample:
        raw_value = self.read_raw_average()
        converted_value = (raw_value - self.offset) / self.calibration_factor
        return Hx711Sample(
            raw_value=raw_value,
            value=float(converted_value),
            unit_label=unit_label,
            timestamp_monotonic=time.monotonic(),
        )

    def tare(self, samples: int = 15) -> int:
        if samples < 1:
            raise ValueError("samples fuer tare muss mindestens 1 sein.")
        raw_values = [self.read_raw() for _ in range(samples)]
        self.offset = float(statistics.median(raw_values))
        return int(round(self.offset))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Einfacher Live-Test fuer einen Soldered HX711 am Raspberry Pi."
    )
    parser.add_argument("--data-pin", type=int, required=True, help="GPIO fuer HX711 DOUT")
    parser.add_argument("--clock-pin", type=int, required=True, help="GPIO fuer HX711 SCK")
    parser.add_argument(
        "--gain",
        type=int,
        default=128,
        choices=(128, 64, 32),
        help="HX711 gain/channel setting: 128 oder 64 fuer Kanal A, 32 fuer Kanal B",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="Abstand zwischen Ausgaben in Sekunden")
    parser.add_argument("--samples", type=int, default=5, help="Anzahl Rohmessungen pro Ausgabe")
    parser.add_argument("--tare", action="store_true", help="Beim Start automatisch nullen")
    parser.add_argument("--tare-samples", type=int, default=20, help="Anzahl Messungen fuer tare")
    parser.add_argument(
        "--calibration-factor",
        type=float,
        default=1.0,
        help="Teiler fuer Umrechnung der Raw-Counts in Zielwert",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Manueller Offset fuer die Umrechnung, falls nicht --tare verwendet wird",
    )
    parser.add_argument("--unit", default="counts", help="Label fuer die umgerechnete Ausgabe, z.B. g oder N")
    parser.add_argument(
        "--pin-mode",
        choices=("bcm", "board"),
        default="bcm",
        help="GPIO numbering mode. Standard ist BCM.",
    )
    parser.add_argument(
        "--pulse-delay",
        type=float,
        default=0.00002,
        help="Kurze Pause pro HX711-Taktflanke in Sekunden. Standard: 20 us.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reader = Hx711Reader(
        data_pin=args.data_pin,
        clock_pin=args.clock_pin,
        gain=args.gain,
        stable_reads=args.samples,
        calibration_factor=args.calibration_factor,
        offset=args.offset,
        pulse_delay=args.pulse_delay,
        use_bcm=args.pin_mode == "bcm",
    )

    try:
        reader.setup()
        print(
            f"HX711 verbunden: DOUT={args.data_pin}, SCK={args.clock_pin}, gain={args.gain}, "
            f"mode={args.pin_mode.upper()}, median={args.samples}"
        )
        if args.tare:
            tare_offset = reader.tare(samples=args.tare_samples)
            print(f"Tare abgeschlossen. Neuer Offset: {tare_offset}")
        else:
            print(f"Starte ohne Tare. Aktueller Offset: {reader.offset:.3f}")

        print("Abbruch mit Ctrl+C")
        while True:
            sample = reader.read(unit_label=args.unit)
            print(
                f"raw={sample.raw_value:>9d} | value={sample.value:>10.3f} {sample.unit_label}"
            )
            time.sleep(max(0.01, float(args.interval)))
    except KeyboardInterrupt:
        print("\nTest beendet.")
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
