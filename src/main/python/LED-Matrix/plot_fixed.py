from rgbmatrix import RGBMatrix, RGBMatrixOptions

# Matrix-Konfiguration
options = RGBMatrixOptions()
options.rows = 64  # Maximale Höhe (Zeilen)
options.cols = 1024  # Maximale Breite (Spalten)
options.chain_length = 1
options.parallel = 1
options.pwm_bits = 2
options.gpio_slowdown = 4
options.show_refresh_rate = True

# Matrix-Objekt initialisieren
matrix = RGBMatrix(options=options)

# Startpunkt für das erste Rechteck
start_col, start_row = 1024, 10

# Erstes Rechteck (klein)
rect_width_1, rect_height_1 = 100, 100  

# Zweites Rechteck (größer)
rect_width_2, rect_height_2 = 100, 40  

# Farben
color_1 = (0, 255, 0)  # Grün für das erste Rechteck
color_2 = (255, 0, 0)  # Rot für das zweite Rechteck

# sonstige Variablen
prev_section = None  


# Funktion zur Überprüfung ob ein Abschnitt überschritten wurde
def set_pixel(row, col):
    global prev_section  # Greife auf die globale Variable zu
    
    # Detect Section Edges (Ein Abschnitt von 0 bis 63)
    new_row = row % 64 
    if new_row != row and row > 0:
        new_col = col + 256
    elif new_row != row and row < 0:
        new_col = col - 256
    else:
        new_col = col
    # Bestimme den aktuellen Abschnitt
    current_section = detect_section(new_row, new_col)
    
    # Falls der vorherige Abschnitt existiert und sich ändert, ignoriere das Pixel
    if prev_section is not None and current_section != prev_section:
        return  # Überspringe das Zeichnen
        
    # Zeichne das Pixel
    matrix.SetPixel(new_col, new_row, *color_1)
    
    # Speichere den aktuellen Abschnitt für den nächsten Pixel
    prev_section = current_section
            
# Funktion zur Überprüfung, in welchem Abschnitt sich der Pixel befindet
def detect_section(row, col):
    global current_section
    
    # Detect if Pixel is out of entire Matrix
    if 0 <= col <= 255: # Abschnitt 1       
        current_section = 1
    elif 256 <= col <= 511: # Abschnitt 2       
        current_section = 2
    elif 512 <= col <= 767: # Abschnitt 3       
        current_section = 3
    elif 768 <= col <= 1023: # Abschnitt 4      
        current_section = 4
    else:
        current_section = 0 # Kann keinem Abschnitt zugeordnet werden
    
    return current_section
    
""" 
# **Erstes (kleineres) Rechteck zeichnen**
for col in range(start_col - rect_width_1//2, start_col + rect_width_1//2 + 1): # vom Cols-Startpunkt bis zum Rand des Rechtecks
    for row in range(start_row - rect_height_1//2, start_row + rect_height_1//2 + 1):
        set_pixel(row, col)
matrix.SetPixel(start_col, start_row, *color_2)

"""

"""

# **Zweites (äußeres) Rechteck mit Wrap-Around**
for cols in range(start_cols - 10, start_cols - 10 + rect_width_2):
    for rows in range(start_rows - 10, start_rows - 10 + rect_height_2):
        new_row, new_col = adjust_position(rows, cols)
        if 0 <= new_col < options.cols:
            # Stelle sicher, dass das innere Rechteck sichtbar bleibt
            if not (start_cols <= cols < start_cols + rect_width_1 and start_rows <= rows < start_rows + rect_height_1):
                matrix.SetPixel(new_col, new_row, *color_2)
"""

set_pixel(0, 767)
set_pixel(0, 768)

# Endlos laufen lassen
try:
    while True:
        pass  # Das Bild bleibt erhalten
except KeyboardInterrupt:
    print("Beendet.")
