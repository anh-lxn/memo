"""
Dateiname: ki_demonstratordaten.py
Autor: Florian Schmidt
Datum: 30.05.2024

Beschreibung:
Diese Datei führt die Verarbeitung von Dehnungsdaten durch. Es werden Daten aus einer Textdatei eingelesen,
gefiltert und visualisiert. Die Benennung der Ordner muss nach einem definierten Schema vorliegen, damit
das Auslesen der Daten funktioniert. Die ausgelesenen Daten werden in einer .csv Datei abgelegt.
"""

import helper_fns_data as h_fns  # Import der Hilfsfunktionen für die Datenverarbeitung

# Definieren von ID, x- und y-Position, Kraft, Zeitintervall und Zeitpunkten
id = 33
x = 290
y = 174
F = 50
dt = 0.02
timepoints = [5, 17]  # Beispielsweise 4s und 12s

# Erstellen des Dateipfads basierend auf den oben definierten Parametern
filepath = f'../resources/messdaten/0{id}_{x}_{y}_{F}/Brücke_(V_V).txt'

# Einlesen der Dehnungsdaten aus der Textdatei
strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = h_fns.read_data_from_txt(filepath)

# Daten in eine Liste packen für eine einfachere Verarbeitung
strain_data = [strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7]

# Ausgabe der ersten paar Werte zur Überprüfung
for i, strain in enumerate(strain_data, start=1):
    print(f"strain_{i}:", strain[:20])

# Parameter für den Tiefpassfilter definieren
cutoff = 0.8  # Grenzfrequenz in Hz
fs = 15       # Abtastrate in Hz (1/0.05s = 20Hz)
order = 5     # Filterordnung

# Anwenden des Tiefpassfilters auf alle Dehnungsdaten
filtered_strain_lists = [h_fns.butter_lowpass_filter(strain, cutoff, fs, order) for strain in strain_data]

# Plotten der ungefilterten und gefilterten Dehnungsdaten
h_fns.plot_strain_data_filtered(strain_data, filtered_strain_lists, dt)

# Speichern der gefilterten Dehnungsdaten in einer CSV-Datei
output_csv_path = f'0{id}_strain_values_{x}_{y}_{F}.csv'
h_fns.save_strain_values_to_csv(timepoints, filtered_strain_lists, output_csv_path, dt, x, y, F)
