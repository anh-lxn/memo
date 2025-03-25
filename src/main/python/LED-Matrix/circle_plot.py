from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time
import math

# LED Matrix Konfiguration
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

def heatmap_color(value):
    r = int(255 * value)
    g = int(255 * (1 - abs(value - 0.5) * 2))
    b = int(255 * (1 - value))
    return (r, g, b)

def gaussian(x, y, x0, y0, sigma):
    return math.exp(-0.5*((x - x0)**2 + (y - y0)**2) / (2*sigma**2))

# Startposition + Richtung
center_x = 255
center_y = 128
sigma = 50
direction = -1  # nach links

try:
    while True:
        canvas = matrix.CreateFrameCanvas()
        for y in range(256):
            for x in range(256):
                value = gaussian(x, y, center_x, center_y, sigma)
                color = heatmap_color(value)
                canvas.SetPixel(x, y, *color)

        canvas = matrix.SwapOnVSync(canvas)

        # Bewegung nach links
        center_x += direction
        if center_x < 0 or center_x > 255:
            direction *= -1  # Richtungswechsel (Ping-Pong)


except KeyboardInterrupt:
    print("Beendet.")
