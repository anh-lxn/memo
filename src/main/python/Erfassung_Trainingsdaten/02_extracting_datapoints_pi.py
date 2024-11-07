"""
Dateiname: 02_extracting_datapoints_pi.py
Autor: Florian Schmidt
Datum: 07.11.2024

Beschreibung:
Diese Datei führt die Verarbeitung von Dehnungsdaten durch. Es werden Daten aus einer Textdatei eingelesen,
gefiltert und visualisiert. Die Benennung der Ordner muss nach einem definierten Schema vorliegen, damit
das Auslesen der Daten funktioniert. Die ausgelesenen Daten werden in einer .csv Datei abgelegt.
"""

from src.main.python import helper_fns_data as h_fns  # Import der Hilfsfunktionen für die Datenverarbeitung
import matplotlib as plt

# Definieren von ID, x- und y-Position, Kraft, Zeitintervall und Zeitpunkten -> definiert, welche Datei ausgelesen wird!
id = 33
x = 290
y = 174
F = 50


# Erstellen des Dateipfads basierend auf den oben definierten Parametern
filepath = f'../../resources/messungen/messung_pc/0{id}_strain_values_{x}_{y}_{F}.csv'

# Einlesen der Dehnungsdaten aus der Textdatei
dtlist, strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = h_fns.read_data_from_txt_adafruit(filepath)

# Daten in eine Liste packen für eine einfachere Verarbeitung
strain_data = [strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7]
dt = dtlist[0]
# Parameter für den Tiefpassfilter definieren
cutoff = 0.6  # Grenzfrequenz in Hz
fs = 1/dt       # Abtastrate in Hz (1/0.25s = 4Hz)
order = 3     # Filterordnung

# Anwenden des Tiefpassfilters auf alle Dehnungsdaten
filtered_strain_lists = [h_fns.butter_lowpass_filter(strain, cutoff, fs, order) for strain in strain_data]
timepoints = [1, 4]  # Beispielsweise 4s und 12s

# Plotten der ungefilterten und gefilterten Dehnungsdaten
h_fns.plot_strain_data_filtered(strain_data, filtered_strain_lists, dt)

# Speichern der gefilterten Dehnungsdaten in einer CSV-Datei
output_csv_path = f'../../resources/messungen/auswertung/0{id}_strain_values_{x}_{y}_{F}_extracted.csv'
h_fns.save_strain_values_to_csv(timepoints,filtered_strain_lists, output_csv_path, dt, x, y, F)
