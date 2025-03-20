import csv

# Öffne die Ausgabe-CSV-Datei einmal und schreibe die Header-Zeile
with open('#simulations_auswertung.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Schreibe die Header-Zeile
    writer.writerow(['ID', 'X', 'Y','Load_Value', 'Strain_1', 'Strain_2', 'Strain_3', 'Strain_4', 'Strain_5', 'Strain_6', 'Strain_7',
                     'Strain_8'])

    # Schleife über die verschiedenen Dateinamen (1 bis 999)
    for i in range(1, 1000):  # Für Lastpunkt 1 bis 999
        file_path = f'sim{i}_SensoranordnungA.txt'
        file_path2 = f'sim{i}_Modellparameter.txt'

        # Listen zum Speichern der Werte
        id_list, load_pos_x, load_pos_y, load_value, point_x, point_y, strain = [], [], [], [], [], [], []

        # Textdatei lesen und Werte extrahieren (Sensoranordnung)
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=',')
                next(reader)  # Überspringe die Header-Zeile, falls vorhanden
                for row in reader:
                    id_list.append(float(row[0]))  # ID als Float speichern
                    point_x.append(float(row[1]))  # Punkt x in mm speichern
                    point_y.append(float(row[2]))  # Punkt y in mm speichern
                    strain.append(float(row[3]))  # Dehnung speichern
        except FileNotFoundError:
            print(f"Warnung: Datei {file_path} für Simulation {i} nicht gefunden.")
            continue

        # Modellparameter lesen und Werte extrahieren
        try:
            with open(file_path2, 'r') as file:
                reader = csv.reader(file, delimiter=',')
                next(reader)  # Überspringe die Header-Zeile, falls vorhanden
                for row in reader:
                    load_pos_x.append(float(row[0]))
                    load_pos_y.append(float(row[1]))
                    load_value.append(float(row[3]))  # Nur den Load-Wert speichern (angenommen, es ist die 4. Spalte)
        except FileNotFoundError:
            print(f"Warnung: Datei {file_path2} für Simulation {i} nicht gefunden.")
            continue

        # Eine Zeile mit den gewünschten Werten erstellen
        row = [i, load_pos_x[0], load_pos_y[0],load_value[0],  strain[0],strain[1],strain[2],strain[3],strain[4],strain[5],strain[6],strain[7]]

        # Zeile in die CSV-Datei schreiben
        writer.writerow(row)

        # Optional: Ausgabe zur Überprüfung
        print(f"Die Daten für Simulation {i} wurden in die simulations_auswertung.csv gespeichert.")
