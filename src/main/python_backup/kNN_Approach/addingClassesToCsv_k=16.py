import pandas as pd

name_file = 'gesamt_simulations_auswertung'
# CSV-Datei laden
df = pd.read_csv(name_file+'.csv')

# Finde den maximalen und minimalen Wert der 'X'-Spalte
x_max = df['X'].max()
x_min = df['X'].min()

print(x_min)
print(x_max)

range = 700


# Funktion zur Kategorisierung für 16 Felder
def kategorisieren(x, y):
    # Bestimme die Intervalle für die 16 Felder
    x_interval = range / 4
    y_interval = range / 4

    # Kategorisierungen anhand der Intervalle
    if x < -x_interval and y > y_interval:
        return '1'
    elif x >= -x_interval and x <= 0 and y > y_interval:
        return '2'
    elif x > 0 and x <= x_interval and y > y_interval:
        return '3'
    elif x > x_interval and y > y_interval:
        return '4'
    elif x < -x_interval and y > 0 and y <= y_interval:
        return '5'
    elif x >= -x_interval and x <= 0 and y > 0 and y <= y_interval:
        return '6'
    elif x > 0 and x <= x_interval and y > 0 and y <= y_interval:
        return '7'
    elif x > x_interval and y > 0 and y <= y_interval:
        return '8'
    elif x < -x_interval and y <= 0 and y > -y_interval:
        return '9'
    elif x >= -x_interval and x <= 0 and y <= 0 and y > -y_interval:
        return '10'
    elif x > 0 and x <= x_interval and y <= 0 and y > -y_interval:
        return '11'
    elif x > x_interval and y <= 0 and y > -y_interval:
        return '12'
    elif x < -x_interval and y <= -y_interval:
        return '13'
    elif x >= -x_interval and x <= 0 and y <= -y_interval:
        return '14'
    elif x > 0 and x <= x_interval and y <= -y_interval:
        return '15'
    else:
        return '16'

# Schleife über alle Zeilen, um die Kategorien zu berechnen
kategorien = []
for index, row in df.iterrows():
    kategorie = kategorisieren(row['X'], row['Y'])
    kategorien.append(kategorie)

# Die berechneten Kategorien in eine neue Spalte einfügen
df['Class'] = kategorien

# Die geänderte CSV-Datei speichern
df.to_csv(name_file+'_edit_k=16.csv', index=False)

# Ausgabe der ersten paar Zeilen zur Überprüfung
print(df.head())
