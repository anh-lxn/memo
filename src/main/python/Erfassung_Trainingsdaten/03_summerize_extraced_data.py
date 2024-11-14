import os
import pandas as pd
from datetime import datetime
import re

# Ordner mit CSV-Dateien angeben
ordner_pfad = "../../resources/messungen/auswertung/14_11"  # <-- Passe diesen Pfad an

# Aktuelles Datum und Uhrzeit (ohne Sekunden) als Teil des Dateinamens
datum_zeit = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_datei = os.path.join(ordner_pfad, f"auswertung_gesamt_{datum_zeit}.csv")

# Erstelle eine Liste, um die extrahierten Zeilen zu speichern
alle_zeilen = []
kopfzeile_gesetzt = False

# Durch alle Dateien im angegebenen Ordner iterieren
for datei_name in os.listdir(ordner_pfad):
    if datei_name.endswith('.csv'):
        # Prüfen, ob der Dateiname mit einer dreistelligen Zahl beginnt
        if not re.match(r'^\d{3}_', datei_name):
            print(f"Überspringe Datei {datei_name}, da sie nicht mit einer dreistelligen Zahl beginnt.")
            continue

        datei_pfad = os.path.join(ordner_pfad, datei_name)

        # CSV-Datei einlesen
        df = pd.read_csv(datei_pfad)

        # Prüfen, ob die letzte Zeile mit '1-2' beginnt
        if df.iloc[-1, 0].startswith("1-2"):
            # Die ersten drei Ziffern des Dateinamens extrahieren
            erste_drei_ziffern = datei_name[:3]

            # Kopfzeile einmalig setzen
            if not kopfzeile_gesetzt:
                # Füge eine neue Spalte "Datei_ID" zur Kopfzeile hinzu
                neue_kopfzeile = ['Datei_ID'] + df.columns.tolist()[1:]
                alle_zeilen.append(neue_kopfzeile)
                kopfzeile_gesetzt = True

            # Die letzte Zeile als Liste umwandeln und die ersten drei Ziffern hinzufügen
            zeile = [erste_drei_ziffern] + df.iloc[-1, 1:].tolist()
            alle_zeilen.append(zeile)

# Alle Zeilen in eine neue CSV-Datei schreiben
output_df = pd.DataFrame(alle_zeilen[1:], columns=alle_zeilen[0])

# Sortiere den DataFrame nach der 'Datei_ID' (als numerische Werte)
output_df['Datei_ID'] = pd.to_numeric(output_df['Datei_ID'])
output_df = output_df.sort_values(by='Datei_ID')

output_df.to_csv(output_datei, index=False)

print(f"Gesamtdatei gespeichert als: {output_datei}")
