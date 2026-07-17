from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - depends on local environment
    serial = None
    list_ports = None


ADDRESS_PATTERN = re.compile(r"(?:0x)?([0-7][0-9a-fA-F])\b")


def parse_address_tokens(line: str) -> list[int]:
    addresses: list[int] = []
    for match in ADDRESS_PATTERN.finditer(line):
        value = int(match.group(1), 16)
        if 0x03 <= value <= 0x77:
            addresses.append(value)
    return addresses


def auto_detect_arduino_port() -> str | None:
    if list_ports is None:
        return None

    candidates = []
    preferred_terms = ("arduino", "ch340", "usb serial", "usb-serial", "wchusbserial")
    for port in list_ports.comports():
        text = f"{port.device} {port.description} {port.manufacturer}".lower()
        if any(term in text for term in preferred_terms):
            candidates.append(port.device)

    if len(candidates) == 1:
        return candidates[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Oeffnet eine serielle Verbindung zum Arduino und zeigt die vom Arduino "
            "gemeldeten I2C-Adressen an."
        )
    )
    parser.add_argument("--port", help="Arduino-Port, z. B. COM3. Ohne Angabe wird automatisch gesucht.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate. Standard ist 115200.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serieller Read-Timeout in Sekunden.")
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Wartezeit nach dem Verbinden, weil viele Arduinos dabei resetten.",
    )
    parser.add_argument(
        "--command",
        default="SCAN",
        help=(
            "Befehl, der nach dem Verbinden an den Arduino gesendet wird. "
            "Mit leerem String wird nichts gesendet."
        ),
    )
    parser.add_argument(
        "--listen-seconds",
        type=float,
        default=5.0,
        help="Wie lange Antworten vom Arduino gelesen werden.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if serial is None:
        print("pyserial ist nicht installiert. Installiere es z. B. mit: pip install pyserial")
        return 1

    port = args.port or auto_detect_arduino_port()
    if not port:
        print("Kein eindeutiger Arduino-Port gefunden. Bitte mit --port COMx angeben.")
        return 1

    found_addresses: set[int] = set()
    print(f"Verbinde mit Arduino auf {port} @ {args.baudrate} baud ...")

    try:
        with serial.Serial(port=port, baudrate=args.baudrate, timeout=args.timeout) as connection:
            time.sleep(max(0.0, args.startup_delay))
            connection.reset_input_buffer()

            if args.command:
                command = args.command.strip()
                print(f"Sende Scan-Befehl: {command!r}")
                connection.write((command + "\n").encode("ascii"))
                connection.flush()

            print(f"Lese Arduino-Ausgabe fuer {args.listen_seconds:.1f} s ...")
            end_time = time.monotonic() + max(0.1, args.listen_seconds)
            while time.monotonic() < end_time:
                raw_line = connection.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                print(f"Arduino: {line}")
                found_addresses.update(parse_address_tokens(line))

    except serial.SerialException as exc:
        print(f"Serielle Verbindung fehlgeschlagen: {exc}")
        return 1
    except OSError as exc:
        print(f"Verbindung oder I/O fehlgeschlagen: {exc}")
        return 1

    if found_addresses:
        formatted = ", ".join(f"0x{address:02X}" for address in sorted(found_addresses))
        print(f"Gefundene I2C-Adressen: {formatted}")
        return 0

    print("Keine I2C-Adressen in der Arduino-Ausgabe erkannt.")
    print("Erwartet werden z. B. Zeilen wie: I2C device found at 0x30")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
