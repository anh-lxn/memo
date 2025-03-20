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
start_col, start_row = 300, 10

# Abmaße Rechtecke
rect_width_0, rect_height_0 = 40, 40 # Erstes Rechteck (klein)
rect_width_1, rect_height_1 = 70, 70 # Erstes Rechteck (klein)
rect_width_2, rect_height_2 = 100, 40  # Zweites Rechteck (größer)

# Farben
color_0 = (150, 0, 0)  # Rot für das nullte Rechteck
color_1 = (0, 150, 0)  # Grün für das erste Rechteck

# Variablen
prev_col = 0

# Funktion zur Überprüfung, in welchem Abschnitt sich der Pixel befindet
"""
def detect_section(row, col):
    global current_section
    
    # Prüfen, in welchem Bereich sich die Spalte befindet
    if 0 <= col <= 255:  # Abschnitt 1        
        current_section = 1
    elif 256 <= col <= 511:  # Abschnitt 2        
        current_section = 2
    elif 512 <= col <= 767:  # Abschnitt 3        
        current_section = 3
    elif 768 <= col <= 1023:  # Abschnitt 4        
        current_section = 4
    else:
        current_section = 0  # Kann keinem Abschnitt zugeordnet werden
    
    return current_section
"""

def col_passed_lower_border(col):
	global prev_col
	
	if prev_col == 255 and col == 256:  # Fall 1  
		return True
	elif prev_col == 511 and col == 512:  # Fall 2  
		return True
	elif prev_col == 767 and col == 768:  # Fall 3  
		return True
	elif prev_col == 1023 and col == 1024:  # Fall 3  
		return True
	else:
		return False

def calculate_col_start(col):

	# Prüfen, in welchem Bereich sich die Spalte befindet
    if 0 <= col <= 255 and start_col > 255:  # Abschnitt 1        
        new_col = 256
        return new_col
    elif  256 <= col <= 511 and start_col > 511:  # Abschnitt 2        
        new_col = 512
        return new_col
    elif 512 <= col <= 767 and start_col > 767:  # Abschnitt 3        
        new_col = 786
        return new_col
    elif 768 <= col <= 1023:  # Abschnitt 4        
        pass
	

# Funktion zur Überprüfung, ob ein Abschnitt überschritten wurde
def set_pixel(row, col, *color):
    global prev_section  # Greife auf die globale Variable zu
    global current_section
    
    # Detect Section Edges (Ein Abschnitt von 0 bis 63)
    new_row = row % 64 
    if new_row != row and row > 0:
        new_col = col + 256
    elif new_row != row and row < 0:
        new_col = col - 256
    else: # liegt zwischen 0 und 63
        new_col = col
   
    # Zeichne das Pixel
    matrix.SetPixel(new_col, new_row, *color)


# **Erstes Rechteck zeichnen**
new_col_start = calculate_col_start(start_col - rect_width_1 // 2)
for col in range(new_col_start, start_col + rect_width_1 // 2 + 1): 
    if not col_passed_lower_border(col):  # Falls die Spalte innerhalb des gültigen Bereichs liegt
        for row in range(start_row - rect_height_1 // 2, start_row + rect_height_1 // 2 + 1):
            set_pixel(row, col, *color_1)
        prev_col = col  # Speichere die letzte gültige Spalte
    else: # Falls außerhalb der Gesamtmatrix
        break
        
# **Nulltes Rechteck zeichnen**
new_col_start = calculate_col_start(start_col - rect_width_0//2)
for col in range(new_col_start, start_col + rect_width_0//2 + 1): # vom Cols-Startpunkt bis zum Rand des Rechtecks
	if not col_passed_lower_border(col):  # Falls die Spalte innerhalb des gültigen Bereichs liegt
		for row in range(start_row - rect_height_0//2, start_row + rect_height_0//2 + 1):
			set_pixel(row, col, *color_0)
		prev_col = col  # Speichere die letzte gültige Spalte
	else: # Falls außerhalb der Gesamtmatrix
		break

# Endlos laufen lassen
try:
    while True:
        pass  # Das Bild bleibt erhalten
except KeyboardInterrupt:
    print("Beendet.")
