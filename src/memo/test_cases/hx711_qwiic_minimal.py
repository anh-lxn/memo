from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:  # pragma: no cover - depends on local environment
    from smbus import SMBus  # type: ignore[no-redef]
    i2c_msg = None


SET_GAIN_32 = 1
SET_GAIN_64 = 2
SET_GAIN_128 = 3
SET_SLEEP_OFF = 5


def set_gain(bus: SMBus, address: int, gain: int):
    gain_map = {
        32: SET_GAIN_32,
        64: SET_GAIN_64,
        128: SET_GAIN_128,
    }
    command = gain_map[gain]
    if i2c_msg is not None:
        bus.i2c_rdwr(i2c_msg.write(address, [command]))
    else:
        bus.write_byte(address, command)


def wake(bus: SMBus, address: int):
    if i2c_msg is not None:
        bus.i2c_rdwr(i2c_msg.write(address, [SET_SLEEP_OFF]))
    else:
        bus.write_byte(address, SET_SLEEP_OFF)


def read_raw(bus: SMBus, address: int) -> int:
    if i2c_msg is not None:
        read = i2c_msg.read(address, 4)
        bus.i2c_rdwr(read)
        data = bytes(read)
    else:
        data = bytes(bus.read_i2c_block_data(address, 0, 4))
    return int.from_bytes(data, byteorder="big", signed=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Minimaler I2C-Test fuer Soldered HX711 easyC/Qwiic")
    parser.add_argument("--bus", type=int, default=1, help="I2C bus, Standard 1")
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x30, help="I2C-Adresse, Standard 0x30")
    parser.add_argument("--gain", type=int, choices=(128, 64, 32), default=128, help="HX711 gain")
    parser.add_argument("--interval", type=float, default=0.2, help="Messintervall in Sekunden")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    with SMBus(args.bus) as bus:
        wake(bus, args.address)
        time.sleep(0.05)
        set_gain(bus, args.address, args.gain)
        time.sleep(0.2)

        print(f"HX711 easyC aktiv auf 0x{args.address:02x}, gain={args.gain}")
        print("Abbruch mit Ctrl+C")

        try:
            while True:
                raw_value = read_raw(bus, args.address)
                print(raw_value)
                time.sleep(max(0.01, args.interval))
        except KeyboardInterrupt:
            print("\nTest beendet.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
