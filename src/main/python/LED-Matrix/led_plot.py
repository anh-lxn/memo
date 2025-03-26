# ----------------- CSV-EMPFANG (unverändert) -----------------
import socket
import threading
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time
import math
import csv
import os
import errno

# CSV-Datei
FILE_NAME = "live_data.csv"
#FILE_PATH = f"../../resources/live_sync/{FILE_NAME}"
FILE_PATH = f"/home/memo/Documents/remote_rsync/src/main/resources/live_sync/{FILE_NAME}"
# GLOBAL
file_lock = threading.Lock()

# Ethernet
HOST = "0.0.0.0"
PORT = 5001

def start_csv_receiver():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        print(f"📡 Warten auf Verbindung auf Port {PORT}...")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"✅ Verbunden mit {addr}")
                tmp_path = FILE_PATH + ".tmp"

                with file_lock:
                    with open(tmp_path, "wb") as f:
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            f.write(data)

                    # Atomarer Austausch
                    os.replace(tmp_path, FILE_PATH)
                    time.sleep(0.2)  # 🛠 Kurze Pause nach dem Ersetzen, um OS-Zugriffsprobleme zu vermeiden

                print("💾 Datei aktualisiert!")

# ----------------- LED-MATRIX -----------------

def heatmap_color(value):
    r = int(255 * value)
    g = int(255 * (1 - abs(value - 0.5) * 2))
    b = int(255 * (1 - value))
    return (r, g, b)

def gaussian(x, y, x0, y0, sigma):
    return math.exp(-0.5*((x - x0)**2 + (y - y0)**2) / (2*sigma**2))

# Koordinatensystem von Membranfeld auf LED-Board transformieren (inkl. 180° Drehung)
def map_value(val, from_min, from_max, to_min, to_max):
    return (val - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

def transform_coords(x2, y2):
    # Skalierung von [-290, 290] → [0, 255]
    x1 = map_value(x2, -290, 290, 0, 255)
    y1 = map_value(y2, -290, 290, 0, 255)
    # 180° Drehung (Spiegelung an X- und Y-Achse)
    x1 = 255 - x1
    y1 = 255 - y1
    return int(x1), int(y1)

# LED - Matrix
def start_led_matrix():
    # Starte Receiver im Hintergrund
    csv_thread = threading.Thread(target=start_csv_receiver, daemon=True)
    csv_thread.start()

    # Matrix Einstellungen
    options = RGBMatrixOptions()
    options.rows = 64
    options.cols = 1024
    options.chain_length = 1
    options.pwm_bits = 4
    options.gpio_slowdown = 5
    options.show_refresh_rate = True
    options.pixel_mapper_config = "grid4x4123"
    options.hardware_mapping = 'regular'

    matrix = RGBMatrix(options=options)

    sigma = 50

    try:
        while True:
            try:
                with file_lock:
                    for i in range(5):  # max. 5 Versuche
                        try:
                            with open(FILE_PATH, mode='r') as csv_file:
                                reader = csv.reader(csv_file)
                                for spalte in reader:
                                    x_coordinate_original = float(spalte[0])
                                    y_coordinate_original = float(spalte[1])
                                    load_force = float(spalte[2])
                            break  # Erfolg → raus aus Retry-Schleife
                        except PermissionError as e:
                            if i < 4:
                                print(f"🔁 CSV nicht lesbar, neuer Versuch in 0.2s... ({i+1}/5)")
                                time.sleep(0.2)
                            else:
                                print("❌ CSV konnte nicht gelesen werden, überspringe...")
                                raise e
            except Exception as e:
                print(f"⚠️ Fehler beim Lesen der CSV-Datei: {e}")
                time.sleep(0.5)
                continue

            x_coordinate_led, y_coordinate_led = transform_coords(x_coordinate_original, y_coordinate_original)
            canvas = matrix.CreateFrameCanvas()
            for y in range(256):
                for x in range(256):
                    value = gaussian(x, y, x_coordinate_led, y_coordinate_led, sigma)
                    color = heatmap_color(value)
                    canvas.SetPixel(x, y, *color)

            canvas = matrix.SwapOnVSync(canvas)

    except KeyboardInterrupt:
        print("Beendet.")

# ----------------- START -----------------

if __name__ == "__main__":
    # Starte CSV-Empfang in einem eigenen Thread
    #csv_thread = threading.Thread(target=start_csv_receiver, daemon=True)
    #csv_thread.start()

    # Starte LED-Animation (läuft im Hauptthread)
    start_led_matrix()
