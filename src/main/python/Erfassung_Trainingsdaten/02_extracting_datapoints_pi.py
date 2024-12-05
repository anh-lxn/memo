"""
Dateiname: 02_extracting_datapoints_pi.py
Autor: Florian Schmidt
Datum: 07.11.2024

Beschreibung:
Diese Datei führt die Verarbeitung von Dehnungsdaten durch. Es werden Daten aus einer Textdatei eingelesen,
gefiltert und visualisiert. Die Benennung der Ordner muss nach einem definierten Schema vorliegen, damit
das Auslesen der Daten funktioniert. Die ausgelesenen Daten werden in einer .csv Datei abgelegt.
"""
import sys
import os
import matplotlib as plt

# Füge den Pfad von python (eine Ebene höher) zum sys.path hinzu, damit helper_fns_data importiert werden kann
sys.path.append(os.path.abspath('../'))
import helper_fns_data as h_fns  # Import der Hilfsfunktionen für die Datenverarbeitung

# Funktion, die Benutzereingaben für die Start-ID, Y-Position, Kraft und Anzahl der Dateien erhält
def get_user_inputs():
    # Listen zur Speicherung der ID und der X-Werte
    ids_list = []  # Liste zur Speicherung der IDs, die mit führenden Nullen formatiert werden
    y_value_list = []  # Liste zur Speicherung der X-Positionen
    y_start = 290  # Startwert für die X-Positionen
    anz_dateien = 11  # Anzahl der Dateien, die verarbeitet werden sollen
    F = 15  # Konstante Kraft, die für alle Dateien gilt
    abstand = -58 # Abstand in y-Richtung zwischen zwei Punkten mit x=const.

    # Benutzerabfrage zur Eingabe der Start-ID
    id_start = int(input("Geben Sie die Start-ID ein: "))  # Start-ID wird als Integer eingegeben
    # Benutzerabfrage zur Eingabe der Y-Position
    x= int(input("Geben Sie die X-Position ein: "))  # X-Position wird als Integer eingegeben
    
    # Schleife zur Berechnung und Speicherung der IDs und X-Werte für jede Datei
    for i in range(anz_dateien):
        # Berechne die aktuelle ID und formatiere sie mit führenden Nullen (z.B. "001", "002", ...)
        formatted_id = str(id_start + i).zfill(3)  # zfill(3) sorgt dafür, dass die ID immer 3-stellig ist
        ids_list.append(formatted_id)  # Füge die formatierte ID zur IDs-Liste hinzu

        # Berechne die X-Position für diese Datei und füge sie zur X-Wert-Liste hinzu
        y_value = y_start + i * abstand  # Y-Wert wird basierend auf der Schleifenvariable berechnet
        y_value_list.append(y_value)  # Füge den berechneten X-Wert zur Liste hinzu

        # Ausgabe der aktuellen ID (dient der Anzeige des aktuellen Fortschritts)
        print(formatted_id)  # Zeigt die formatierte ID im Terminal an

    # Rückgabe der Listen und der anderen Parameter (y, F und die Anzahl der Dateien)
    return ids_list, x, y_value_list, F, anz_dateien  # Gibt die Listen und Werte zurück



# Hauptprogramm Ausführung
if __name__ == "__main__":
	ids_list, x, y_value_list, F, anz_dateien = get_user_inputs()  # Benutzereingaben abrufen
	for i in range(anz_dateien):
		# Erstellen des Dateipfads basierend auf den oben definierten Parametern
		filepath = f'../../resources/messungen/messung_pi_05_12_15N/{ids_list[i]:3}_strain_values_{x}_{y_value_list[i]}_{F}.csv'

		# Einlesen der Dehnungsdaten aus der Textdatei
		dtlist, strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = h_fns.read_data_from_txt_adafruit(filepath)

		# Daten in eine Liste packen für eine einfachere Verarbeitung
		strain_data = [strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7]
		dt = dtlist[0]

		# Plotten der ungefilterten Dehnungsdaten
		timepoints = h_fns.plot_strain_data(strain_data, dt)  # Ungefilterte Daten plotten [0, eingebener_wert]

		# Speichern der ungefilterten Dehnungsdaten in einer CSV-Datei
		output_csv_path = f'../../resources/messungen/auswertung/05_12_15N/{ids_list[i]:3}_strain_values_{x}_{y_value_list[i]}_{F}_extracted.csv'
		h_fns.save_strain_values_to_csv(timepoints, strain_data, output_csv_path, dt, x, y_value_list[i], F)
