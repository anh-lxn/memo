import pandas as pd

name_file = 'gesamt_simulations_auswertung'
# CSV-Datei laden
df = pd.read_csv(name_file+'.csv')

# Finde den maximalen und minimalen Wert der 'X'-Spalte
x_max = df['X'].max()
x_min = df['X'].min()

print(x_min)
print(x_max)

split = 117

# Funktion zur Kategorisierung
def kategorisieren(x, y):
    if x < -split and y > split:
        return '1'
    elif x >= -split and x <= split  and y > split:
        return '2'
    elif x > split and y > split:
        return '3'
    elif x < -split and y > -split and y <= split:
        return '4'
    elif x >= -split and x <= split and y > -split and y <= split:
        return '5'
    elif x > split and y > -split and y <= split:
        return '6'
    elif x < -split and y <= -split:
        return '7'
    elif x >= -split and x <= split and y <= -split:
        return '8'
    else:
        return '9'

# Schleife über alle Zeilen, um die Kategorien zu berechnen
kategorien = []
for index, row in df.iterrows():
    kategorie = kategorisieren(row['X'], row['Y'])
    kategorien.append(kategorie)

# Die berechneten Kategorien in eine neue Spalte einfügen
df['Class'] = kategorien

# Die geänderte CSV-Datei speichern
df.to_csv(name_file+'_edit.csv', index=False)

# Ausgabe der ersten paar Zeilen zur Überprüfung
print(df.head())
