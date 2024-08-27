# Skript zum Auslesen der Simulationsdaten

import csv

# Schleife über die verschiedenen Dateinamen
for i in range(1, 10):  # Für belastungspunkt1 bis belastungspunkt9
    file_path = f'belastungspunkt{i}_f50_Dehnungen.txt'

    # Listen für die aktuellen Datei-Daten
    id, point_x, point_y, strain_xx, strain_yy = [], [], [], [], []

    # Textdatei lesen und Werte extrahieren
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Überspringe die Header-Zeile, falls vorhanden
        for row in reader:
            id.append(int(float(row[0])))  # ID als Ganzzahl speichern
            point_x.append(float(row[1]))
            point_y.append(float(row[2]))
            strain_xx.append(float(row[3]))
            strain_yy.append(float(row[4]))

    # Optional: Überprüfen, ob die Werte korrekt gelesen wurden
    print(f'Daten aus {file_path}:')
    print(f'IDs: {id[0:10]}')
    print(f'Point X: {point_x[0:10]}')
    print(f'Point Y: {point_y[0:10]}')
    print(f'Strain XX: {strain_xx[0:10]}')
    print(f'Strain YY: {strain_yy[0:10]}')

