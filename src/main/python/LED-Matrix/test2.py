from rgbmatrix import RGBMatrix, RGBMatrixOptions, graphics
import math
import time
import random
from PIL import Image
import socket
import threading
from queue import Queue

# ---------------- Connection Setup ----------------
HOST = "0.0.0.0"
PORT = 5001


# ---------------- Bereichsdefinition ----------------
"""
 0
0--------> x
 |
 |
 |
 y
"""
# (x_start, y_start, x_ende, y_ende)
AREA_TEXT_HEADER = (0, 0, 256, 63)
AREA_HEATMAP = (0, 64, 195, 256)
AREA_BALKEN = (210, 64, 230, 256)
AREA_LOGO = (155, 0, 256, 63)

def in_area(x, y, area):
    x0, y0, x1, y1 = area
    return x0 <= x <= x1 and y0 <= y <= y1

# ---------------- Hilfsfunktionen ----------------
def gaussian(x, y, x0, y0, sigma):
    return math.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def heatmap_color(value):
    """
    Mappt einen Wert zwischen 0 und 1 auf eine Heatmap-Farbe,
    ähnlich der Farbskala im Bild (cyan → grün → gelb → violett).
    """
    value = max(0.0, min(1.0, value))  # Clamp
    if value < 0.25:
        # cyan (0,255,255) → green (0,255,0)
        t = value / 0.5
        r = 50
        g = 99
        b = int(255 * (1 - t))
    elif value < 0.5:
        # green (0,255,0) → yellow (255,255,0)
        t = (value - 0.25) / 0.25
        r = int(255 * t)
        g = 150
        b = 120
    elif value < 0.75:
        # yellow (255,255,0) → pinkish (255,0,255)
        t = (value - 0.5) / 0.25
        r = 255
        g = int(255 * (1 - t))
        b = int(255 * t)
    else:
        # pinkish (255,0,255) → dark violet (100, 0, 200)
        t = (value - 0.75) / 0.25
        r = int(255 * (1 - t) + 100 * t)
        g = 0
        b = int(255 * (1 - t) + 200 * t)

    return (r, g, b)

def is_logo_pixel(r, g, b): # zur Erkennung von Blau im Logo
    return b > 50 and r < 100 and g < 100

def draw_rotated_rectangle_polygon(canvas, cx, cy, width, height, center_x, center_y, color=(0, 255, 0)):
    angle = math.atan2(center_y - cy, center_x - cx)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    hw = width / 2
    hh = height / 2

    # Eckpunkte vor Rotation
    corners = [
        (-hw, -hh),
        ( hw, -hh),
        ( hw,  hh),
        (-hw,  hh)
    ]

    # Eckpunkte nach Rotation und Verschiebung
    rotated = []
    for dx, dy in corners:
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        rotated.append((rx, ry))

    # Rasterisiert und füllt das Polygon (Scanline fill)
    fill_polygon(canvas, rotated, color)

def fill_polygon(canvas, points, color):
    # Y-Grenzen ermitteln
    min_y = int(min(p[1] for p in points))
    max_y = int(max(p[1] for p in points))

    for y in range(min_y, max_y + 1):
        # Schnittpunkte mit horizontaler Linie
        intersections = []
        for i in range(len(points)):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % len(points)]
            if y0 == y1:
                continue
            if (y >= min(y0, y1)) and (y <= max(y0, y1)):
                # Linear interpoliert
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append(x_int)

        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = int(intersections[i])
                x_end = int(intersections[i + 1])
                for x in range(x_start, x_end + 1):
                    if 0 <= x < 256 and 0 <= y < 256:
                        canvas.SetPixel(x, y, *color)
def draw_rectangle(canvas, cx, cy, width, height, color=(0, 255, 0), allign="right"):
    if allign == "right":
        for y in range(cy, cy + height+1):
            for x in range(cx, cx + width+1):
                if 0 <= x < 256 and 0 <= y < 256:
                    canvas.SetPixel(x, y, *color)
    elif allign == "left":
        for y in range(cy, cy - height-1, -1):
            for x in range(cx, cx - width-1, -1):
                if 0 <= x < 256 and 0 <= y < 256:
                    canvas.SetPixel(x, y, *color)

def calculate_xy_respect_to_width(width):
    if width == 10:
        x_offset, y_offset = 0, 0
    elif width == 9 or width == 8:
        x_offset, y_offset = 1, 1
    elif width == 7 or width == 6 or width == 5:
        x_offset, y_offset = 2, 2
    elif width == 4 or width == 3 or width == 2:
        x_offset, y_offset = 3, 3
    elif width == 1 or width == 0:
        x_offset, y_offset = 4, 4
    
    return x_offset, y_offset

def map_value(val, from_min, from_max, to_min, to_max):
    return (val - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

def transform_coords(x2, y2):
    x1 = map_value(x2, -400, 400, AREA_HEATMAP[0], AREA_HEATMAP[2])
    y1 = map_value(y2, -400, 400, AREA_HEATMAP[1], AREA_HEATMAP[3])
    # Y-Achse invertieren:
    y1 = AREA_HEATMAP[3] - (y1 - AREA_HEATMAP[1])
    return int(x1), int(y1)

def radial_falloff(x, y, x0, y0, max_radius=140):
    dx = x - x0
    dy = y - y0
    r = math.sqrt(dx**2 + dy**2)
    value = max(0.0, 1.0 - (r / max_radius))
    return value

# ---------------- Text Setup ----------------
font_header = graphics.Font()
font_header.LoadFont("../../../lib/rpi-rgb-led-matrix/fonts/7x14B.bdf")  # ggf. Pfad anpassen
text_color = graphics.Color(255, 255, 255)

font_balken = graphics.Font()
font_balken.LoadFont("../../../lib/rpi-rgb-led-matrix/fonts/6x9.bdf")

# ---------------- Balken Setup ----------------
bar_value = 0
bar_direction = 1

# ---------------- Logo Setup ----------------
image = Image.open("ITM_Logo.png").convert('RGB')
image = image.resize((100+10, 64+10)) # 64x64 Pixel

# ---------------- Sensoren Setup ----------------
sensors = {
    "S1": (16, 64+64+64+48),
    "S2": (16, 64+64+32),
    "S3": (16, 64+16),
    "S4": (64+32, 64+16),
    "S5": (64+64+48, 64+16),
    "S6": (64+64+48, 64+64+32),
    "S7": (64+64+48, 64+64+64+48),
    "S8": (64+32, 64+64+64+48),
}

# ---------------- LED-Matrix ----------------
def start_led_matrix():
    # ---------------- Matrix Setup ----------------
    options = RGBMatrixOptions()
    options.rows = 64
    options.cols = 1024
    options.chain_length = 1
    options.pwm_bits = 4
    options.gpio_slowdown = 5
    options.show_refresh_rate = True
    options.pixel_mapper_config = "grid4x4123;Rotate:270"
    options.hardware_mapping = 'regular'

    matrix = RGBMatrix(options=options)
    matrix.brightness = 60

    # Datenpuffer (thread-sicher)
    data_queue = Queue()

    sensor_max_values = {
    "S1": 3.65,
    "S2": 3.19,
    "S3": 3.37,
    "S4": 3.43,
    "S5": 3.29,
    "S6": 3.29,
    "S7": 3.62,
    "S8": 3.55}

    def render_loop():
        bar_value = 0
        while True:
            if not data_queue.empty():
                while not data_queue.empty():
                    x, y, f, *sensor_values = data_queue.get()
                canvas = matrix.CreateFrameCanvas()
                x_led, y_led = transform_coords(x, y)

                print(f"X: {round(x, 1)}, Y: {round(y, 1)}")
                print(f"X_NEU: {x_led}, Y_NEU: {y_led}")

                # Heatmap
                for yi in range(AREA_HEATMAP[1], AREA_HEATMAP[3]):
                    for xi in range(AREA_HEATMAP[0], AREA_HEATMAP[2]):                        
                        value = radial_falloff(xi, yi, x_led, y_led, max_radius=150)
                        value = min(1.0, (value**0.9))
                        color = heatmap_color(value)
                        canvas.SetPixel(xi, yi, *color)

                # Sensoren
                color_sensoren = (13, 205, 216)
                
                for name, (x, y) in sensors.items():
                    draw_rotated_rectangle_polygon(canvas, x, y, width=10, height=5, center_x=64+32, center_y=64+64+32, color=(1, 36, 49))

                    idx = int(name[1]) - 1  # "S1" → 0, "S2" → 1, ...
                    max_val = sensor_max_values[name]
                    level = 10 - round((sensor_values[idx] / max_val) * 10)
                    level = max(0, min(level, 10))  # Clamp to 0..10
                    x_offset, y_offset = calculate_xy_respect_to_width(level)

                    if name == "S1":
                        draw_rotated_rectangle_polygon(canvas, x - x_offset, y + y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
                    elif name == "S2":
                        draw_rectangle(canvas, x-5, y-2, width=level, height=4, color=color_sensoren, allign="right")
                    elif name == "S3":
                        draw_rotated_rectangle_polygon(canvas, x - x_offset, y - y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
                    elif name == "S4":
                        draw_rectangle(canvas, x-3, y-5, width=5, height=level, color=color_sensoren, allign="right")
                    elif name == "S5":
                        draw_rotated_rectangle_polygon(canvas, x + x_offset, y - y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
                    elif name == "S6":
                        draw_rectangle(canvas, x+5, y+2, width=level, height=4, color=color_sensoren, allign="left")
                    elif name == "S7":
                        draw_rotated_rectangle_polygon(canvas, x + x_offset, y + y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
                    elif name == "S8":
                        draw_rectangle(canvas, x+2, y+5, width=5, height=level, color=color_sensoren, allign="left")

                # Text Header
                graphics.DrawText(canvas, font_header, 5, 25, text_color, "KI-basierte Lokali-")
                graphics.DrawText(canvas, font_header, 5, 40, text_color, "sierung von Punktlasten")
                graphics.DrawText(canvas, font_header, 5, 55, text_color, "auf Textilmembran")
                
                # Text Balken
                graphics.DrawText(canvas, font_balken, 235, 165, text_color, "10N")
                graphics.DrawText(canvas, font_balken, 200, 165, text_color, "-")
                graphics.DrawText(canvas, font_balken, 235, 75, text_color, "20N")
                graphics.DrawText(canvas, font_balken, 200, 75, text_color, "-")

                # Balken
                bar_value = f/20.0 # 0.00 < bar_value < 1.00 
                bar_value = 1.0 if bar_value > 1.0 else bar_value
                bar_height = int((AREA_BALKEN[3] - AREA_BALKEN[1]) * bar_value)
                for y in range(AREA_BALKEN[3] - bar_height, AREA_BALKEN[3]):
                    for x in range(AREA_BALKEN[0], AREA_BALKEN[2]):
                        canvas.SetPixel(x, y, 13, 205, 216) 


                # ITM-Logo
                for y in range(image.height):
                    for x in range(image.width):
                        r, g, b = image.getpixel((x, y))
                        if is_logo_pixel(r, g, b):
                            canvas.SetPixel(x + AREA_LOGO[0], y + AREA_LOGO[1] - 5, 255, 255, 255)
                        else: # Kein Logo Pixel
                            pass

                # Refresh Bild
                canvas = matrix.SwapOnVSync(canvas)
        
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
                            parts = line.strip().split(",")
                            if len(parts) == 11:
                                x, y, f = map(float, parts[:3])
                                sensors = list(map(float, parts[3:]))
                                data_queue.put((x, y, f, *sensors))
                            else:
                                print(f"⚠️ Ungültiges Format (brauche 11 Werte): {line}")
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
        
        
        
        

        
        

