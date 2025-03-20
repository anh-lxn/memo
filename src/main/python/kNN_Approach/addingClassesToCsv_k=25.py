import pandas as pd

name_file = 'gesamt_simulations_auswertung'
# CSV-Datei laden
df = pd.read_csv(name_file+'.csv')

# Finde den maximalen und minimalen Wert der 'X'-Spalte
x_max = df['X'].max()
x_min = df['X'].min()

print(x_min)
print(x_max)

dimension_x = 700
dimension_y = 700


# Funktion zur Kategorisierung für 25 Felder
def kategorisieren(x, y):
    # Bestimme die Intervalle für die 25 Felder
    x_interval = dimension_x / 5
    y_interval = dimension_y / 5

    # Kategorisierungen anhand der Intervalle
    if x < -x_interval*1.5 and y > 1.5 * y_interval:
        return '1'
    elif x >= -x_interval*1.5 and x < -0.5 and y > 1.5 * y_interval:
        return '2'
    elif x >= -0.5*x_interval and x < 0.5 * x_interval and y > 1.5 * y_interval:
        return '3'
    elif x >= 0.5 * x_interval and x < 1.5 * x_interval and y > 1.5 * y_interval:
        return '4'
    elif x >= 1.5 * x_interval and y > 1.5 * y_interval:
        return '5'
    elif x < -x_interval*1.5 and y >= 0.5 * y_interval and y <= 1.5 * y_interval:
        return '6'
    elif x >= -x_interval*1.5 and x < -0.5*x_interval and y >= 0.5 * y_interval and y <= 1.5 * y_interval:
        return '7'
    elif x >= -0.5*x_interval and x < 0.5 * x_interval and y >= 0.5 * y_interval and y <= 1.5 * y_interval:
        return '8'
    elif x >= 0.5 * x_interval and x < 1.5 * x_interval and y >= 0.5 * y_interval and y <= 1.5 * y_interval:
        return '9'
    elif x >= 1.5* x_interval and y >= 0.5 * y_interval and y <= 1.5 * y_interval:
        return '10'
    elif x < -1.5*x_interval and y > -0.5 * y_interval and y <= 0.5 * y_interval: # hier weiter
        return '11'
    elif x >= -1.5*x_interval and x < -0.5*x_interval and y > -0.5 * y_interval and y <= 0.5 * y_interval:
        return '12'
    elif x >= -0.5*x_interval and x < 0.5 * x_interval and y > -0.5 * y_interval and y <= 0.5 * y_interval:
        return '13'
    elif x >= 0.5 * x_interval and x < 1.5 * x_interval and y > -0.5 * y_interval and y <= 0.5 * y_interval:
        return '14'
    elif x >= 1.5 * x_interval and y > -0.5 * y_interval and y <= 0.5 * y_interval:
        return '15'
    elif x < -1.5*x_interval and y >= -1.5*y_interval and y <= -0.5 * y_interval:
        return '16'
    elif x >= -1.5*x_interval and x < -0.5*x_interval and y >= -1.5*y_interval and y <= -0.5 * y_interval:
        return '17'
    elif x >= -0.5*x_interval and x < 0.5 * x_interval and y >= -1.5*y_interval and y <= -0.5 * y_interval:
        return '18'
    elif x >= 0.5 * x_interval and x < 1.5 * x_interval and y >= -1.5*y_interval and y <= -0.5 * y_interval:
        return '19'
    elif x >= 1.5 * x_interval and y >= -1.5*y_interval and y <= -0.5 * y_interval:
        return '20'
    elif x < -1.5*x_interval and y < -1.5*y_interval:
        return '21'
    elif x >= -1.5*x_interval and x < -0.5*x_interval and y < -1.5*y_interval:
        return '22'
    elif x >= -0.5 * x_interval and x < 0.5 * x_interval and y < -1.5*y_interval:
        return '23'
    elif x >= 0.5 * x_interval and x <  1.5* x_interval and y < -1.5*y_interval:
        return '24'
    elif x >= 1.5 * x_interval and y < -1.5*y_interval:
        return '25'

# Schleife über alle Zeilen, um die Kategorien zu berechnen
kategorien = []
for index, row in df.iterrows():
    kategorie = kategorisieren(row['X'], row['Y'])
    kategorien.append(kategorie)

# Die berechneten Kategorien in eine neue Spalte einfügen
df['Class'] = kategorien

# Die geänderte CSV-Datei speichern
df.to_csv(name_file+'_edit_k=25.csv', index=False)

# Ausgabe der ersten paar Zeilen zur Überprüfung
print(df.head())
