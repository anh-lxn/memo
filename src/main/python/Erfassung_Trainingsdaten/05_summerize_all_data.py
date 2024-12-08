import pandas as pd
import csv
import os
import re

# Ordner mit CSV-Dateien angeben
ordner_pfad = "../../resources/interpolation"

# Listen zum Speichern der Werte
headers = ['Datei_ID','X','Y','F','Sensor R1', 'Sensor R2', 'Sensor R3', 'Sensor R4', 'Sensor R5', 'Sensor R6', 'Sensor R7', 'Sensor R8']
alle_zeilen = []
alle_zeilen.append(headers)
forces = []

# Werte auslesen
def read_data_from_csv(ordner_pfad):
    ### Finde alle CSV-Dateien im angegebenen Ordner
    alle_dateien = os.listdir(ordner_pfad) # Liste aller Dateien im angegebenen Ordner
    
    # Filtere nur die Dateien mit der Endung '.csv'
    csv_dateien = [] # Leere Liste zum Speichern der CSV-Dateien
    for datei in alle_dateien:
        if datei.endswith('.csv') and ('interpoliert' in datei or 'original' in datei):
            csv_dateien.append(datei)

        # Extrahiere die Zahl vor dem 'N' (die Kraft)
        if ('interpoliert' in datei or 'original' in datei):
            match = re.search(r'(\d+)N', datei)
            if match:
                # Wenn ein Treffer gefunden wurde, die Zahl extrahieren und zur Liste hinzufügen
                n = int(match.group(1))  # Zahl vor dem 'N'
                forces.append(n)
    
    # Sortiert Kräfte
    forces.sort()
    # Sortiere die csv_dateien basierend auf der extrahierten Zahl (forces)
    csv_dateien = sorted(csv_dateien, key=lambda datei: int(re.search(r'\d+', datei).group()))

    ### Daten auslesen
    id = 0
    # Durch jede CSV-Datei iterieren
    for datei_name in csv_dateien:
        # Prüfen, ob der Dateiname den gewünschten Aufbau hat
        if not datei_name.startswith("auswertung_gesamt_") or not re.search(r'\d+N', datei_name):
            print(f"Überspringe Datei {datei_name}, da sie nicht den gewünschten Aufbau hat.")
            continue

        datei_pfad = os.path.join(ordner_pfad, datei_name)

        # CSV-Datei einlesen
        df = pd.read_csv(datei_pfad)

        # Erste Spalte löschen (Datei-ID)
        df = df.drop(df.columns[0], axis=1)

        # Lese Zeile für Zeile durch den DataFrame 
        for index, zeile in df.iterrows():
            # Konvertiere die Zeile in eine Liste und füge sie zu alle_zeilen hinzu
            zeilen_liste = zeile.tolist() # Konvertiert die Serie in eine Liste
            zeilen_liste.insert(0, id) # Fügt die Datei-ID als erste Spalte hinzu
            id += 1
            alle_zeilen.append(zeilen_liste)  # Fügt die Liste zu alle_zeilen hinzu

# Erzeuge die Daten für die CSV-Datei
def create_summarized_csv_data():
    read_data_from_csv(ordner_pfad)

    # Legt den Pfad für die CSV-Datei fest
    output_csv_path = f'../../resources/interpolation/auswertung_gesamt'
    for i in range(len(forces)):
        output_csv_path += f'_{forces[i]}N'
    output_csv_path += '.csv'
        
    # Schreibe die Daten in die CSV-Datei
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(alle_zeilen)

    print(f"Data saved to {output_csv_path}")

create_summarized_csv_data()