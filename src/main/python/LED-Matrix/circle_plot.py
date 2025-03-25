from rgbmatrix import RGBMatrix, RGBMatrixOptions, graphics
import time
import math

# LED Matrix Konfiguration
options = RGBMatrixOptions()
options.rows = 64
options.cols = 64
options.chain_length = 16
options.parallel = 1
options.pixel_mapper_config = "Python:pixelmapper.py"  # Dein Mapper
options.hardware_mapping = 'regular'

matrix = RGBMatrix(options=options)

# Farbe für den Kreis
color = graphics.Color(255, 0, 0)  # Rot

# Anfangspunkt (Mittelpunkt des Kreises)
x_center = 128
y_center = 128
radius = 20

while True:
    matrix.Clear()
    
    # Kreis zeichnen
    for angle in range(0, 360, 10):  # alle 10°
        rad = math.radians(angle)
        x = int(x_center + radius * math.cos(rad))
        y = int(y_center + radius * math.sin(rad))
        
        if 0 <= x < 256 and 0 <= y < 256:
            matrix.SetPixel(x, y, color.red, color.green, color.blue)

    # Beispiel: Mittelpunkt bewegen (einfacher Ping-Pong-Effekt)
    x_center += 1
    if x_center > 200:
        x_center = 56  # zurückspringen

    time.sleep(0.05)
