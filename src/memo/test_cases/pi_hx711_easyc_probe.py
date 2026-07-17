from __future__ import annotations

import argparse
import time

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:  # pragma: no cover - depends on local environment
    try:
        from smbus import SMBus  # type: ignore[no-redef]
        i2c_msg = None
    except ImportError:
        SMBus = None
        i2c_msg = None


SET_GAIN_32 = 1
SET_GAIN_64 = 2
SET_GAIN_128 = 3
SET_SLEEP_OFF = 5


def i2c_ping(bus: SMBus, address: int) -> bool:
    try:
        if i2c_msg is not None:
            bus.i2c_rdwr(i2c_msg.write(address, []))
        else:
            bus.write_quick(address)
        return True
    except OSError:
        return False


def write_command(bus: SMBus, address: int, command: int):
    if i2c_msg is not None:
        bus.i2c_rdwr(i2c_msg.write(address, [command & 0xFF]))
    else:
        bus.write_byte(address, command & 0xFF)


def read_raw(bus: SMBus, address: int) -> int:
    if i2c_msg is not None:
        read = i2c_msg.read(address, 4)
        bus.i2c_rdwr(read)
        data = bytes(read)
    else:
        data = bytes(bus.read_i2c_block_data(address, 0, 4))

    value = int.from_bytes(data, byteorder="big", signed=True)
    raw24 = value & 0x00FFFFFF
    if raw24 & 0x00800000:
        raw24 -= 1 << 24
    return raw24


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direkter Raspberry-Pi-Test fuer HX711 easyC/Qwiic.")
    parser.add_argument("--bus", type=int, default=1, help="I2C-Bus. Raspberry Pi GPIO-Pins sind meist Bus 1.")
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x31)
    parser.add_argument("--scan", action="store_true", help="Scannt 0x30 bis 0x37 vor dem Lesen.")
    parser.add_argument("--gain", type=int, choices=(128, 64, 32), default=64)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if SMBus is None:
        print("Kein SMBus-Modul gefunden. Installiere smbus2 oder python3-smbus.")
        return 1

    gain_command = {32: SET_GAIN_32, 64: SET_GAIN_64, 128: SET_GAIN_128}[args.gain]

    with SMBus(args.bus) as bus:
        if args.scan:
            found = [address for address in range(0x30, 0x38) if i2c_ping(bus, address)]
            if found:
                print("Gefunden:", ", ".join(f"0x{address:02X}" for address in found))
            else:
                print(f"Keine HX711-easyC-Adresse auf Bus {args.bus} bei 0x30-0x37 gefunden.")

        if not i2c_ping(bus, args.address):
            print(f"Adresse 0x{args.address:02X} antwortet nicht auf Bus {args.bus}.")
            return 2

        print(f"Adresse 0x{args.address:02X} antwortet auf Bus {args.bus}.")
        write_command(bus, args.address, SET_SLEEP_OFF)
        time.sleep(0.05)
        write_command(bus, args.address, gain_command)
        time.sleep(0.2)

        previous_time = None
        for _ in range(max(1, args.samples)):
            start = time.monotonic()
            try:
                raw_value = read_raw(bus, args.address)
            except OSError as exc:
                print(f"read_raw fehlgeschlagen: {exc}")
                return 3
            elapsed_ms = (time.monotonic() - start) * 1000.0
            delta_ms = "-" if previous_time is None else f"{(time.monotonic() - previous_time) * 1000.0:.1f}"
            previous_time = time.monotonic()
            print(f"raw={raw_value:>10d} read_ms={elapsed_ms:>7.1f} dt_ms={delta_ms}")
            time.sleep(max(0.0, args.interval))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
