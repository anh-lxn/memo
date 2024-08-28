# Skript zum Auslesen der Simulationsdaten
import csv

# Schleife über die verschiedenen Dateinamen
for i in range(1, 999):  # Für Lastpunkt 1 bis 999
    file_path = f'sim{i}_SensoranordnungA.txt'
    file_path2 = f'sim{i}_Modellparameter.txt'
    # Listen zum Speichern der Werte
    id_list, load_pos_x, load_pos_y,load_value, point_x, point_y, strain = [], [], [], [], [], [], []

    # Textdatei lesen und Werte extrahieren
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Überspringe die Header-Zeile, falls vorhanden
        for row in reader:
            id_list.append(float(row[0]))  # ID als Float speichern
            point_x.append(float(row[1]))  # Punkt x in mm speichern
            point_y.append(float(row[2]))  # Punkt y in mm speichern
            strain.append(float(row[3]))  # Dehnung speichern

    with open(file_path2,'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for row in reader:
            load_pos_x.append(float(row[0]))
            load_pos_y.append(float(row[1]))
            load_value.append(float(row[3]))

    # Optional: Überprüfen, ob die Werte korrekt gelesen wurden
    print('ID:', id_list[0:10])
    print('LoadPos X:', load_pos_x[0:10])
    print('LoadPos Y:', load_pos_y[0:10])
    print('Load Value:', load_value[0:10])
    print('Point X:', point_x[0:10])
    print('Point Y:', point_y[0:10])
    print('Strain:', strain[0:10])

