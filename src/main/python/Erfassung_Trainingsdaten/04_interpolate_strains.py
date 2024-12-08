import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
from scipy.interpolate import interp1d
import csv

# Design der Plots
plt.style.use("cyberpunk")

# Hintergrundfarbe und Textfarben für den Cyberpunk-Stil
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # blaugrauer Hintergrund

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # sehr helles Grau für Text und Achsen


# Datei und Daten laden
datei_pfade = [ "../../resources/interpolation/auswertung_gesamt_10N_original.csv", 
                "../../resources/interpolation/auswertung_gesamt_15N_original.csv",
                "../../resources/interpolation/auswertung_gesamt_20N_original.csv" ]

# Listen zum Speichern der Werte
load_positions = []
strain_sensors_10N = [[] for _ in range(8)]  # Liste für die 8 Sensoren mit 10 Newton
strain_sensors_15N = [[] for _ in range(8)]  # Liste für die 8 Sensoren mit 15 Newton
strain_sensors_20N = [[] for _ in range(8)]  # Liste für die 8 Sensoren mit 20 Newton

# Kräfte (10N, 15N, 20N) und Positionen (x, y)
forces = [10, 15, 20]  # Kräfte in Newton

# Werte zum ändern
sensor = 1
anz_messwerte = 11
interpolated_force = 8  # Interpolationskraft in Newton

def read_data_from_csv(datei_pfade):
    # Die Datei zeilenweise einlesen
    for datei in range(len(datei_pfade)):
        with open(datei_pfade[datei], mode='r', newline='') as csv_datei:
            reader = csv.reader(csv_datei)
            # Kopfzeile extrahieren
            kopfzeile = next(reader)
            #print("Kopfzeile:", kopfzeile)

            # Jede weitere Zeile lesen und verarbeiten
            for zeile in reader:
                if len(load_positions) < 121:
                    load_positions.append((int(zeile[1]), int(zeile[2]))) # Koordinaten der Lastpunkte mit load_positions[id]
                    # print(load_positions[0]) -> (x, y) = (-290, 290)
                for i in range(8):
                    if datei == 0:
                        strain_sensors_10N[i].append(float(zeile[4 + i])) # Dehnungen der Sensoren mit strain_sensors[R{i + 1}]
                    elif datei == 1:
                        strain_sensors_15N[i].append(float(zeile[4 + i])) # Dehnungen der Sensoren mit strain_sensors[R{i + 1}]
                    else:
                        strain_sensors_20N[i].append(float(zeile[4 + i])) # Dehnungen der Sensoren mit strain_sensors[R{i + 1}]
                        # print(strain_sensors[2] -> R3: [0.680625, 1.070625, 1.462125, 1.855875, 2.36075, 3.014125, 2.74375, 2.599625, 2.561375, 2.588625] 

    #print(f"Load positions: {load_positions}") 
    #print(f"Strain sensors für 10N: {strain_sensors_10N[2]}") 
    #print(f"Strain sensors für 15N: {strain_sensors_15N[2]}") 
    #print(f"Strain sensors für 20N: {strain_sensors_20N[2]}")

    return load_positions, strain_sensors_10N, strain_sensors_15N, strain_sensors_20N

def create_interpolated_strain_plot(sensor, anz_messwerte):

    load_positions, strain_sensors_10N, strain_sensors_15N, strain_sensors_20N = read_data_from_csv(datei_pfade)
    x = np.array([position[0] for position in load_positions[:anz_messwerte]])  # x-Punkt
    #y = np.array([position[1] for position in load_positions[:anz_messwerte]])  # y-Punkte
    y = np.linspace(-290, 290, anz_messwerte)

    # Interpolation: Zwischenwerte berechnen für eine beliebige Kraft
    stress_data = np.array([strain_sensors_10N[sensor], strain_sensors_15N[sensor], strain_sensors_20N[sensor]])
    interpolator = interp1d(forces, stress_data, kind='linear', axis=0, fill_value="extrapolate")

    # Spannungswerte für die interpolierte Kraft berechnen
    stress_interp = interpolator(interpolated_force)
    print(stress_interp[:anz_messwerte])

    # Plot der Spannungswerte
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(y, strain_sensors_10N[sensor][:anz_messwerte], 'o-', label="10N")
    plt.plot(y, strain_sensors_15N[sensor][:anz_messwerte], 'o-', label="15N")
    plt.plot(y, strain_sensors_20N[sensor][:anz_messwerte], 'o-', label="20N")
    plt.plot(y, stress_interp[:anz_messwerte], 'o--', label=f"{interpolated_force}N (interpoliert)", color='red')

    plt.xlabel('y-Punkt')
    plt.ylabel('Spannung')
    plt.title(f'Interpolation der Spannungswerte für {interpolated_force}N des Sensors R{sensor + 1}')
    plt.ylim(0, 4)
    plt.legend()
    plt.grid()
    mplcyberpunk.make_lines_glow(ax)
    plt.show()

def save_interpolated_strain_values_to_csv():

    load_positions, strain_sensors_10N, strain_sensors_15N, strain_sensors_20N = read_data_from_csv(datei_pfade)

    output_csv_path = f'../../resources/interpolation/auswertung_gesamt_{interpolated_force}N_interpoliert.csv'

    # Interpolation: Zwischenwerte berechnen für eine beliebige Kraft
    strain_data_sensorR1 = np.array([strain_sensors_10N[0], strain_sensors_15N[0], strain_sensors_20N[0]])
    strain_data_sensorR2 = np.array([strain_sensors_10N[1], strain_sensors_15N[1], strain_sensors_20N[1]])
    strain_data_sensorR3 = np.array([strain_sensors_10N[2], strain_sensors_15N[2], strain_sensors_20N[2]])
    strain_data_sensorR4 = np.array([strain_sensors_10N[3], strain_sensors_15N[3], strain_sensors_20N[3]])
    strain_data_sensorR5 = np.array([strain_sensors_10N[4], strain_sensors_15N[4], strain_sensors_20N[4]])
    strain_data_sensorR6 = np.array([strain_sensors_10N[5], strain_sensors_15N[5], strain_sensors_20N[5]])
    strain_data_sensorR7 = np.array([strain_sensors_10N[6], strain_sensors_15N[6], strain_sensors_20N[6]])
    strain_data_sensorR8 = np.array([strain_sensors_10N[7], strain_sensors_15N[7], strain_sensors_20N[7]])

    # Interpolation der Sensoren R1 bis R8
    interpolator_sensorR1 = interp1d(forces, strain_data_sensorR1, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR2 = interp1d(forces, strain_data_sensorR2, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR3 = interp1d(forces, strain_data_sensorR3, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR4 = interp1d(forces, strain_data_sensorR4, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR5 = interp1d(forces, strain_data_sensorR5, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR6 = interp1d(forces, strain_data_sensorR6, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR7 = interp1d(forces, strain_data_sensorR7, kind='linear', axis=0, fill_value="extrapolate")
    interpolator_sensorR8 = interp1d(forces, strain_data_sensorR8, kind='linear', axis=0, fill_value="extrapolate")

    # Dehnungswerte für die interpolierte Kraft berechnen
    strain_interpolated_sensorR1 = interpolator_sensorR1(interpolated_force)
    strain_interpolated_sensorR2 = interpolator_sensorR2(interpolated_force)
    strain_interpolated_sensorR3 = interpolator_sensorR3(interpolated_force)
    strain_interpolated_sensorR4 = interpolator_sensorR4(interpolated_force)
    strain_interpolated_sensorR5 = interpolator_sensorR5(interpolated_force)
    strain_interpolated_sensorR6 = interpolator_sensorR6(interpolated_force)
    strain_interpolated_sensorR7 = interpolator_sensorR7(interpolated_force)
    strain_interpolated_sensorR8 = interpolator_sensorR8(interpolated_force)

    # Liste der interpolierten Werte
    interpolated_strain_lists = [
        strain_interpolated_sensorR1,
        strain_interpolated_sensorR2,
        strain_interpolated_sensorR3,
        strain_interpolated_sensorR4,
        strain_interpolated_sensorR5,
        strain_interpolated_sensorR6,
        strain_interpolated_sensorR7,
        strain_interpolated_sensorR8
    ]

    # Erzeuge die Daten für die CSV-Datei
    data_for_csv = []
    headers = ['Datei_ID','X','Y','F','Sensor R1', 'Sensor R2', 'Sensor R3', 'Sensor R4', 'Sensor R5', 'Sensor R6', 'Sensor R7', 'Sensor R8'] #'X','Y','F',
    data_for_csv.append(headers)

    # Füge die Zeilen der interpolierten Werte hinzu
    for i in range(len(load_positions)):
        row = [i, load_positions[i][0], load_positions[i][1], interpolated_force]  # Datei_ID, X, Y, F
        for sensor_values in interpolated_strain_lists:
            row.append(sensor_values[i])  # Dehnungswerte für Sensor R1 bis R8
        data_for_csv.append(row)

    # Schreibe die Daten in die CSV-Datei
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_for_csv)

    print(f"Data saved to {output_csv_path}")



#create_interpolated_strain_plot(sensor, anz_messwerte)

save_interpolated_strain_values_to_csv()