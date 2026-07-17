from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - depends on local environment
    list_ports = None


DEFAULT_FQBN = "arduino:avr:uno"
DEFAULT_BAUDRATE = 230400
DEFAULT_ADDRESSES = "0x31"

# Direkt hier anpassen, wenn du das Skript aus VS Code ohne Argumente startest.
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_SKIP_FLASH = True

def auto_detect_arduino_port() -> str | None:
    if list_ports is None:
        return None

    preferred_terms = (
        "arduino",
        "ch340",
        "usb serial",
        "usb-serial",
        "wchusbserial",
        "ttyacm",
        "ttyusb",
    )
    candidates = []
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
            "Flasht den Arduino-HX711-Bridge-Sketch und startet danach den "
            "HX711-Live-Plot."
        )
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"Arduino-Port, z. B. /dev/ttyACM0 oder /dev/ttyUSB0. Standard: {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--fqbn",
        default=DEFAULT_FQBN,
        help=f"Arduino fully qualified board name. Standard: {DEFAULT_FQBN}",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"Serial-Baudrate fuer Live-Plot. Standard: {DEFAULT_BAUDRATE}",
    )
    parser.add_argument(
        "--addresses",
        default=DEFAULT_ADDRESSES,
        help=f"Kommagetrennte HX711-Adressen fuer den Plot. Standard: {DEFAULT_ADDRESSES}",
    )
    parser.add_argument(
        "--skip-flash",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_FLASH,
        help="Arduino nicht flashen, nur Live-Plot mit dem angegebenen Port starten.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Wartezeit nach Upload, weil der Arduino danach meist resettet.",
    )
    return parser


def run_checked(command: list[str], cwd: Path):
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    sketch_dir = Path(__file__).resolve().parent / "arduino_hx711_qwiic_serial_bridge"
    live_plot_script = Path(__file__).resolve().parent / "hx711_qwiic_live_plot.py"

    port = args.port or auto_detect_arduino_port()
    if not port:
        print("Kein eindeutiger Arduino-Port gefunden.")
        print("Bitte Port explizit angeben, z. B.: --port /dev/ttyACM0")
        return 1

    if not args.skip_flash:
        arduino_cli = shutil.which("arduino-cli")
        if arduino_cli is None:
            print("arduino-cli wurde nicht gefunden.")
            print("Installiere arduino-cli oder starte nur den Plot mit --skip-flash.")
            print("Beispiel:")
            print(f"  python {Path(__file__).name} --port {port} --skip-flash")
            return 1

        try:
            run_checked([arduino_cli, "compile", "--fqbn", args.fqbn, str(sketch_dir)], cwd=repo_root)
            run_checked(
                [arduino_cli, "upload", "-p", port, "--fqbn", args.fqbn, str(sketch_dir)],
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Arduino compile/upload fehlgeschlagen mit Exit-Code {exc.returncode}.")
            return exc.returncode or 1

        time.sleep(max(0.0, args.startup_delay))

    env = os.environ.copy()
    env["MEMO_HX711_SOURCE"] = "serial"
    env["MEMO_HX711_SERIAL_PORT"] = port
    env["MEMO_HX711_SERIAL_BAUDRATE"] = str(args.baudrate)
    env["MEMO_HX711_SERIAL_DEBUG"] = "0"
    env["MEMO_HX711_ADDRESSES"] = args.addresses

    print(f"Starte Live-Plot auf {port} @ {args.baudrate} ...")
    return subprocess.run([sys.executable, str(live_plot_script)], cwd=repo_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
