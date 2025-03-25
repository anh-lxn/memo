# pixelmapper.py

def map(x, y):
    panel_width = 64
    panel_height = 64
    grid_width = 4  # 4 panels pro Zeile
    grid_height = 4 # 4 panels pro Spalte

    # Bestimme Panel innerhalb des 4x4 Grids
    panel_x = x // panel_width
    panel_y = y // panel_height
    panel_index = panel_y * grid_width + panel_x

    # Pixel-Koordinaten innerhalb des Panels
    inner_x = x % panel_width
    inner_y = y % panel_height

    # Jetzt umrechnen auf physikalische Koordinaten im 64x1024 Streifen
    phys_x = inner_x
    phys_y = panel_index * panel_height + inner_y

    return (phys_x, phys_y)