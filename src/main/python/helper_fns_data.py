import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
import csv

def read_data_from_txt_adafruit(filepath):
    # Listen initialisieren
    strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = [], [], [], [], [], [], [], []

    # CSV-Datei einlesen
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Jede Zeile enthält Timestamp und 8 Sensorwerte
            timestamp = row['Timestamp']
            # Zugriff auf die Sensorwerte
            strain_0.append(float(row['Voltage1']))
            strain_1.append(float(row['Voltage2']))
            strain_2.append(float(row['Voltage3']))
            strain_3.append(float(row['Voltage4']))
            strain_4.append(float(row['Voltage5']))
            strain_5.append(float(row['Voltage6']))
            strain_6.append(float(row['Voltage7']))
            strain_7.append(float(row['Voltage8']))

        print(strain_0)
    return strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7

def read_data_from_txt(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    data_section_found = False
    data_lines = []

    # Suche nach der "data:"-Sektion und extrahiere die folgenden Zeilen
    for line in lines:
        if data_section_found:
            data_lines.append(line.strip())
        elif line.strip().lower() == "data:":
            data_section_found = True

    # Listen initialisieren
    strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = [], [], [], [], [], [], [], []

    # Datenzeilen verarbeiten
    for line in data_lines:
        values = [float(value.replace(',', '.')) for value in line.split()]
        if len(values) == 8:
            strain_0.append(values[0])
            strain_1.append(values[1])
            strain_2.append(values[2])
            strain_3.append(values[3])
            strain_4.append(values[4])
            strain_5.append(values[5])
            strain_6.append(values[6])
            strain_7.append(values[7])

    return strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7

def plot_strain_data(strain_data, dt):
    """
    Plot the strain data with x-values starting from 0 and incrementing by 0.05.

    Parameters:
    strain_data (list of lists): A list containing 8 lists of strain values.
    """
    # Anzahl der Datenpunkte
    num_points = len(strain_data[0])

    # Generiere die x-Werte
    x_values = [i * dt for i in range(num_points)]
    # Erstelle den Plot
    plt.figure(figsize=(12, 8))

    # Plot für jeden strain
    for i, strain in enumerate(strain_data, start=0):
        plt.plot(x_values, strain, label=f'Strain {i}')

    # Diagrammeinstellungen
    plt.xlabel('Zeit (s)')
    plt.ylabel('Dehnung (Strain)')
    plt.title('Dehnungswerte der Sensoren über die Zeit')
    plt.legend()
    plt.grid(True)
    # Zeige den Plot an
    plt.show()

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def plot_strain_data_filtered(strain_data, filtered_strain_data, dt):
    num_points = len(strain_data[0])
    print("Number of datapoints: ", num_points)
    x_values = [i * dt for i in range(num_points)]
    plt.figure(figsize=(12, 8))
    for i, (strain, filtered_strain) in enumerate(zip(strain_data, filtered_strain_data), start=1):
        plt.plot(x_values, strain, label=f'Strain {i} Original')
        plt.plot(x_values, filtered_strain, label=f'Strain {i} Gefiltert', linestyle='--')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Dehnung (Strain)')
    plt.title('Dehnungswerte der Sensoren über die Zeit')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_strain_values_to_csv(timepoints, filtered_strain_lists, output_csv_path, dt, x, y, F):
    # Berechne die Indizes der gewünschten Zeitpunkte
    indices = [int(time / dt) for time in timepoints]
    print("indices: ", indices)
    # Erzeuge die Daten für die CSV-Datei
    data_for_csv = []
    headers = ['Messwert','X','Y','F','Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8'] #'X','Y','F',
    data_for_csv.append(headers)
    i = 1
    for idx in indices:
        row = [i,x,y, F]  # Zeitwert hinzufügen
        for strain_list in filtered_strain_lists:
            row.append(strain_list[idx])
        data_for_csv.append(row)
        i += 1

    result = [a - b for a, b in zip(data_for_csv[1], data_for_csv[2])]
    data_for_csv.append(result)
    data_for_csv[3][0] = '1-2'
    data_for_csv[3][1] = x
    data_for_csv[3][2] = y
    data_for_csv[3][3] = F
    # Schreibe die Daten in die CSV-Datei
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_for_csv)

    print(f"Data saved to {output_csv_path}")

def read_last_line(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Überspringe die Header-Zeile
        last_row = None
        for row in reader:
            last_row = row
            if last_row is not None:
                last_row = [value if value == '1-2' else float(value) for value in last_row]
        return last_row

def create_scatterplot(load_pos_x, load_pos_y, sensor_pos):
    sensor_x, sensor_y = zip(*sensor_pos)  # Entpacke die Tupel in separate Listen für x und y

    # Scatterplot erstellen
    plt.figure(figsize=(8, 8))  # Größere Figur für bessere Sichtbarkeit

    # Hintergrundfläche einfärben
    plt.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color='yellow', alpha=0.4)

    plt.scatter(load_pos_x, load_pos_y, c='blue', alpha=0.5)  # Scatterplot mit blauen Punkten und halber Transparenz
    plt.scatter(sensor_x, sensor_y, c='red', alpha=1.0, marker='x', label='Sensor Positions')  # Sensorpositionen in Rot

    # Sensorpositionen beschriften
    for i, (sx, sy) in enumerate(sensor_pos, start=1):
        plt.text(sx, sy, "Sensor "+str(i)+"  ", fontsize=7, fontweight='bold', ha='right')

    plt.xlim(-500, 500)  # Setzt die x-Achse auf den Bereich -500 bis 500
    plt.ylim(-500, 500)  # Setzt die y-Achse auf den Bereich -500 bis 500
    plt.xlabel('X-Achse [mm]')
    plt.ylabel('Y-Achse [mm]')
    plt.title('XY-2D-Plot - Lastpunkte (Blau), Sensorpunkte (Rot)')
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.show()

# Funktion zum Erstellen des Scatterplots
def create_scatterplot_testing(load_pos_x, load_pos_x_pred, load_pos_y, load_pos_y_pred, sensor_pos):
    sensor_x, sensor_y = zip(*sensor_pos)  # Entpacke die Tupel in separate Listen für x und y

    # Scatterplot erstellen
    plt.figure(figsize=(8, 8))  # Größere Figur für bessere Sichtbarkeit

    # Hintergrundfläche einfärben
    plt.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color='yellow', alpha=0.4)

    # Scatterplots mit Labels hinzufügen
    for i, (x, y) in enumerate(zip(load_pos_x, load_pos_y), start=1):
        plt.scatter(x, y, c='blue', alpha=0.8)
        plt.text(x, y, str(i)+"  ", fontsize=7, ha='right', va='bottom')

    for i, (x, y) in enumerate(zip(load_pos_x_pred, load_pos_y_pred), start=1):
        plt.scatter(x, y, c='green', alpha=0.5)
        plt.text(x, y, str(i)+"  ", fontsize=7, ha='right', va='bottom')

    plt.scatter(sensor_x, sensor_y, c='red', alpha=1.0, marker='x', label='Sensor Positions')  # Sensorpositionen in Rot

    # Sensorpositionen beschriften
    for i, (sx, sy) in enumerate(sensor_pos, start=1):
        plt.text(sx, sy, "Sensor "+str(i)+"  ", fontsize=7, fontweight='bold', ha='right')

    plt.xlim(-500, 500)  # Setzt die x-Achse auf den Bereich -500 bis 500
    plt.ylim(-500, 500)  # Setzt die y-Achse auf den Bereich -500 bis 500
    plt.xlabel('X-Achse [mm]')
    plt.ylabel('Y-Achse [mm]')
    plt.title('Memo-2D-Plot - Lastpunkte (Blau), Vorhergesagte Lastpunkte (Grün), Sensorpunkte (Rot)')
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.show()