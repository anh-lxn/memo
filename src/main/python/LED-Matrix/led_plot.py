import socket
import threading
from queue import Queue
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import math
import time

HOST = "0.0.0.0"
PORT = 5001

def heatmap_color(value):
    r = int(255 * value)
    g = int(255 * (1 - abs(value - 0.5) * 2))
    b = int(255 * (1 - value))
    return (r, g, b)

def gaussian(y, x, y0, x0, sigma):
    return math.exp(-0.5*((x - x0)**2 + (y - y0)**2) / (2*sigma**2))

def radial_falloff(y, x, y0, x0, max_radius=140):
    dx = x - x0
    dy = y - y0
    r = math.sqrt(dx**2 + dy**2)
    value = max(0.0, 1.0 - (r / max_radius))
    return value


def map_value(val, from_min, from_max, to_min, to_max):
    return (val - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

def transform_coords(x2, y2):
    x1 = map_value(x2, -290, 290, 0, 255)
    y1 = map_value(y2, -290, 290, 0, 255)
    x1 = 255 - x1
    y1 = 255 - y1
    return int(x1), int(y1)

def start_led_matrix():
    # LED-Matrix Setup
    options = RGBMatrixOptions()
    options.rows = 64
    options.cols = 1024
    options.chain_length = 1
    options.pwm_bits = 4
    options.gpio_slowdown = 5
    options.show_refresh_rate = False
    options.pixel_mapper_config = "grid4x4123"
    options.hardware_mapping = 'regular'

    matrix = RGBMatrix(options=options)
    matrix.brightness = 70

    # Datenpuffer (thread-sicher)
    data_queue = Queue()

    def render_loop():
        sigma = 30 

        while True:
            if not data_queue.empty():
                while not data_queue.empty():
                    x, y, f = data_queue.get()

                x_led, y_led = transform_coords(x, y)
                canvas = matrix.CreateFrameCanvas()

                for xi in range(256):
                    for yi in range(256):
                        #value = gaussian(yi, xi, y_led, x_led, sigma)
                        value = radial_falloff(yi, xi, y_led, x_led, max_radius=150)
                        #value = value**0.01
                        value = min(1.0, (value**0.9))
                        color = heatmap_color(value)
                        canvas.SetPixel(yi, xi, *color)

                matrix.SwapOnVSync(canvas)

    def network_loop():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, PORT))
            server_socket.listen()
            print(f"📡 Warten auf Verbindung auf Port {PORT}...")

            conn, addr = server_socket.accept()
            print(f"✅ Verbunden mit {addr}")
            buffer = ""

            with conn:
                while True:
                    data = conn.recv(1024).decode('utf-8')
                    if not data:
                        break
                    buffer += data
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        try:
                            x_str, y_str, f_str = line.strip().split(",")
                            x = float(x_str)
                            y = float(y_str)
                            f = float(f_str)
                            data_queue.put((x, y, f))
                        except ValueError:
                            print(f"⚠️ Ungültige Daten empfangen: {line}")

    # Starte beide Threads
    threading.Thread(target=network_loop, daemon=True).start()
    threading.Thread(target=render_loop, daemon=True).start()

    # Hauptthread wartet, um das Programm am Leben zu halten
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🔴 Beendet.")

# ----------------- START -----------------

if __name__ == "__main__":
    # Starte LED-Animation (läuft im Hauptthread)
    start_led_matrix()
