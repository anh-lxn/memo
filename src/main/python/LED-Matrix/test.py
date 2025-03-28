from rgbmatrix import RGBMatrix, RGBMatrixOptions, graphics
import math
import time
import random
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
canvas = matrix.CreateFrameCanvas()
matrix.brightness = 60

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


# ---------------- Main Loop ----------------
x0 = 60 # Mittelpunkt X-Koordinate
y0 = 200 # Mittelpunkt Y-Koordinate
sigma = 50
direction = -1

try:
    while True:
        canvas.Clear()

        
        # Heatmap
        for y in range(AREA_HEATMAP[1], AREA_HEATMAP[3]):
            for x in range(AREA_HEATMAP[0], AREA_HEATMAP[2]):  # Nur rechter Bereich
                value = gaussian(x, y, x0, y0, sigma)
                color = heatmap_color(min(value, 1.0))
                canvas.SetPixel(x, y, *color)
        
        
        # Sensoren
        color_sensoren = (13, 205, 216)
        for name, (x, y) in sensors.items():
            draw_rotated_rectangle_polygon(canvas, x, y, width=10, height=5, center_x=64+32, center_y=64+64+32, color=(1, 36, 49))
            #Sensorwert simulieren
            level = random.randint(0, 10)  # oder dein echter Wert
            #print(name, x, y)
            if name == "S1":
                x_offset, y_offset = calculate_xy_respect_to_width(level)
                draw_rotated_rectangle_polygon(canvas, x-x_offset, y+y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
            elif name == "S2":
                draw_rectangle(canvas, x-5, y-2, width=level, height=4, color=color_sensoren, allign="right")
            elif name == "S3":
                x_offset, y_offset = calculate_xy_respect_to_width(level)
                draw_rotated_rectangle_polygon(canvas, x-x_offset, y-y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
            elif name == "S4":
                draw_rectangle(canvas, x-3, y-5, width=5, height=level, color=color_sensoren, allign="right")
            elif name == "S5":
                x_offset, y_offset = calculate_xy_respect_to_width(level)
                draw_rotated_rectangle_polygon(canvas, x+x_offset, y-y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
            elif name == "S6":
                draw_rectangle(canvas, x+5, y+2, width=level, height=4, color=color_sensoren, allign="left")
            elif name == "S7":
                x_offset, y_offset = calculate_xy_respect_to_width(level)
                draw_rotated_rectangle_polygon(canvas, x+x_offset, y+y_offset, width=level, height=5, center_x=64+32, center_y=64+64+32, color=color_sensoren)
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

        # Balken: Wert in Prozent (0..1 → Y-Höhe berechnen)
        bar_height = int((AREA_BALKEN[3] - AREA_BALKEN[1]) * bar_value)
        for y in range(AREA_BALKEN[3] - bar_height, AREA_BALKEN[3]):
            for x in range(AREA_BALKEN[0], AREA_BALKEN[2]):
                canvas.SetPixel(x, y, 13, 205, 216) 

        # Beispiel rechnung für Balken
        bar_value += bar_direction * 0.2
        if bar_value >= 1.0:
            bar_value = 1.0
            bar_direction = -1
        elif bar_value <= 0.0:
            bar_value = 0.0
            bar_direction = 1

        # ITM-Logo
        for y in range(image.height):
            for x in range(image.width):
                r, g, b = image.getpixel((x, y))
                if is_logo_pixel(r, g, b):
                    canvas.SetPixel(x + AREA_LOGO[0], y + AREA_LOGO[1] - 5, 255, 255, 255)
                else: # Kein Logo Pixel
                    pass

    
        # Refresh BIld
        canvas = matrix.SwapOnVSync(canvas)
        
        #time.sleep(0.3)


except KeyboardInterrupt:
    print("🔴 Abbruch durch Benutzer.")
