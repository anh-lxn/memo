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
    from smbus2 import SMBus, i2c_msg
except ImportError:  # pragma: no cover - depends on local environment
    SMBus = None
    i2c_msg = None

    try:
        from smbus import SMBus  # type: ignore[no-redef]
    except ImportError:
        SMBus = None


SET_GAIN_32 = 1
SET_GAIN_64 = 2
SET_GAIN_128 = 3
SET_SLEEP_ON = 4
SET_SLEEP_OFF = 5


@dataclass
class Hx711I2CSample:
    raw_value: int
    value: float
    unit_label: str
    timestamp_monotonic: float


class Hx711QwiicReader:
    """Reader for the Soldered HX711 easyC/Qwiic I2C variant."""

    def __init__(
        self,
        bus_number: int = 1,
        address: int = 0x30,
        calibration_factor: float = 1.0,
        offset: float = 0.0,
        stable_reads: int = 1,
    ):
        if SMBus is None:
            raise RuntimeError(
                "Kein SMBus-Modul gefunden. Installiere z. B. python3-smbus oder smbus2."
            )
        if calibration_factor == 0:
            raise ValueError("calibration_factor darf nicht 0 sein.")
        if stable_reads < 1:
            raise ValueError("stable_reads muss mindestens 1 sein.")

        self.bus_number = int(bus_number)
        self.address = int(address)
        self.calibration_factor = float(calibration_factor)
        self.offset = float(offset)
        self.stable_reads = int(stable_reads)
        self._bus = None

    def open(self):
        if self._bus is None:
            self._bus = SMBus(self.bus_number)

    def close(self):
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def _ensure_bus(self):
        self.open()
        return self._bus

    def ping(self):
        bus = self._ensure_bus()
        if i2c_msg is not None:
            write = i2c_msg.write(self.address, [])
            bus.i2c_rdwr(write)
            return
        bus.read_byte(self.address)

    def _write_command(self, command: int):
        bus = self._ensure_bus()
        if i2c_msg is not None:
            write = i2c_msg.write(self.address, [int(command) & 0xFF])
            bus.i2c_rdwr(write)
            return
        bus.write_byte(self.address, int(command) & 0xFF)

    def _read_exact(self, num_bytes: int) -> bytes:
        bus = self._ensure_bus()
        if i2c_msg is not None:
            read = i2c_msg.read(self.address, num_bytes)
            bus.i2c_rdwr(read)
            return bytes(read)

        data = bus.read_i2c_block_data(self.address, 0, num_bytes)
        return bytes(data)

    def set_gain(self, gain: int):
        gain_map = {
            128: SET_GAIN_128,
            64: SET_GAIN_64,
            32: SET_GAIN_32,
        }
        if gain not in gain_map:
            raise ValueError("gain muss 128, 64 oder 32 sein.")
        self._write_command(gain_map[gain])

    def set_sleep(self, enabled: bool):
        self._write_command(SET_SLEEP_ON if enabled else SET_SLEEP_OFF)

    def read_raw(self) -> int:
        data = self._read_exact(4)
        return int.from_bytes(data, byteorder="big", signed=True)

    def read_raw_average(self) -> int:
        values = [self.read_raw() for _ in range(self.stable_reads)]
        return int(round(statistics.median(values)))

    def read(self, unit_label: str = "counts") -> Hx711I2CSample:
        raw_value = self.read_raw_average()
        converted_value = (raw_value - self.offset) / self.calibration_factor
        return Hx711I2CSample(
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


def auto_detect_address(bus_number: int, candidates: list[int]) -> int | None:
    reader = Hx711QwiicReader(bus_number=bus_number, address=candidates[0])
    try:
        for address in candidates:
            reader.address = address
            try:
                reader.ping()
                return address
            except OSError:
                continue
    finally:
        reader.close()
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="I2C live test fuer den Soldered HX711 easyC/Qwiic."
    )
    parser.add_argument("--bus", type=int, default=1, help="I2C bus number. Standard ist 1.")
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=0x30,
        help="I2C-Adresse des Boards. Standard ist 0x30.",
    )
    parser.add_argument(
        "--auto-address",
        action="store_true",
        help="Scannt 0x30 bis 0x37 und verwendet die erste gefundene Adresse.",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=128,
        choices=(128, 64, 32),
        help="Gain fuer Kanal A/B wie von Soldered dokumentiert.",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="Abstand zwischen Ausgaben in Sekunden")
    parser.add_argument("--samples", type=int, default=1, help="Anzahl Rohmessungen pro Ausgabe")
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
    parser.add_argument("--unit", default="counts", help="Label fuer die umgerechnete Ausgabe, z. B. g oder N")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    address = args.address
    if args.auto_address:
        detected = auto_detect_address(args.bus, list(range(0x30, 0x38)))
        if detected is None:
            print("Kein HX711 easyC auf 0x30 bis 0x37 gefunden.")
            return 1
        address = detected

    reader = Hx711QwiicReader(
        bus_number=args.bus,
        address=address,
        calibration_factor=args.calibration_factor,
        offset=args.offset,
        stable_reads=args.samples,
    )

    try:
        reader.open()
        reader.ping()
        reader.set_sleep(False)
        time.sleep(0.05)
        reader.set_gain(args.gain)
        time.sleep(0.2)

        print(
            f"HX711 easyC verbunden: bus={args.bus}, address=0x{address:02x}, "
            f"gain={args.gain}, median={args.samples}"
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
                f"raw={sample.raw_value:>12d} | value={sample.value:>12.3f} {sample.unit_label}"
            )
            time.sleep(max(0.01, float(args.interval)))
    except KeyboardInterrupt:
        print("\nTest beendet.")
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
