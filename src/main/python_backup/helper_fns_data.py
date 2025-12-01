import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import matplotlib.patches as patches
import numpy as np
import csv
from matplotlib.transforms import Affine2D
from matplotlib.widgets import TextBox

def read_data_from_txt_adafruit(filepath):
    # Listen initialisieren
    strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = [], [], [], [], [], [], [], []
    dt = []
    # CSV-Datei einlesen
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Jede Zeile enthält Timestamp, dt und 8 Sensorwerte
            timestamp = row['Timestamp']
            dt.append(float(row['dt']))
            # Zugriff auf die Sensorwerte
            strain_0.append(float(row['Voltage1']))
            strain_1.append(float(row['Voltage2']))
            strain_2.append(float(row['Voltage3']))
            strain_3.append(float(row['Voltage4']))
            strain_4.append(float(row['Voltage5']))
            strain_5.append(float(row['Voltage6']))
            strain_6.append(float(row['Voltage7']))
            strain_7.append(float(row['Voltage8']))

    return dt,strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7

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
    Plot the strain data with x-values starting from 0 and incrementing by dt.
    Allow the user to click on the plot to select a time and return the corresponding strain values.

    Parameters:
    strain_data (list of lists): A list containing 8 lists of strain values.
    dt (float): Time step for the x-axis values.
    
    Returns:
    tuple: The selected time and the corresponding strain values at that time.
    """
    # Anzahl der Datenpunkte
    num_points = len(strain_data[0])

    # Generiere die x-Werte (Zeit)
    x_values = np.array([i * dt for i in range(num_points)])

    # Erstelle den Plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot für jeden Strain
    for i, strain in enumerate(strain_data, start=0):
        ax.plot(x_values, strain, label=f'Sensor R{i + 1}')

    # Diagrammeinstellungen
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Dehnung (Strain)')
    ax.set_title('Dehnungswerte der Sensoren über die Zeit')
    ax.legend()
    ax.grid(True)
    
    # Klick-Ereignis-Variable
    clicked_time = None
    clicked_strains = None

    # Klick-Funktion
    def on_click(event):
        nonlocal clicked_time, clicked_strains
        if event.inaxes:  # Überprüfen, ob der Klick innerhalb des Plots war
            clicked_time = event.xdata  # x-Koordinate (Zeit)
            
            # Index des nächsten Datenpunkts basierend auf der Zeit
            closest_idx = np.abs(x_values - clicked_time).argmin()
            clicked_time = x_values[closest_idx]  # Exakte Zeit aus den Daten
            clicked_strains = [strain[closest_idx] for strain in strain_data]  # Strain-Werte an der nächsten Zeit

            print(f"Gewählte Zeit: {clicked_time:.2f}s")
            print(f"Strain-Werte: {clicked_strains}")

            # Markieren des ausgewählten Punkts
            for i, strain in enumerate(clicked_strains):
                ax.plot(clicked_time, strain, 'ro')  # Punkt markieren
            fig.canvas.draw()  # Plot aktualisieren

            # Schließen des Plots, um den Wert zurückzugeben
            plt.close()

    # Verbindung der Klick-Ereignisfunktion mit dem Plot
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Make the plot fullscreen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    # Zeige das Plot an und halte das Fenster offen
    plt.show()

    # Gebe den geklickten Wert zurück
    return [0, float(clicked_time)] if clicked_time else [0, 0.0] 

def butter_lowpass_filter(data, cutoff, fs, order=5):
    print(data)
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
    #print("indices: ", indices)
    # Erzeuge die Daten für die CSV-Datei
    data_for_csv = []
    headers = ['Messwert','X','Y','F','Sensor R1', 'Sensor R2', 'Sensor R3', 'Sensor R4', 'Sensor R5', 'Sensor R6', 'Sensor R7', 'Sensor R8'] #'X','Y','F',
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

def create_scatterplot2(load_pos_x, load_pos_y, sensor_pos):
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



def create_scatterplot(load_pos_x, load_pos_y, sensor_pos, rectangle_width=50, rectangle_height=20, rotation_angles=None):
    sensor_x, sensor_y = zip(*sensor_pos)  # Entpacke die Tupel in separate Listen für x und y

    # Falls keine Rotationswinkel angegeben sind, setzen wir sie alle auf 0 Grad
    if rotation_angles is None:
        rotation_angles = [45, 0, -45, 90, 45, 0, -45, 90]

    # Scatterplot erstellen
    plt.figure(figsize=(8, 8))  # Größere Figur für bessere Sichtbarkeit

    # Hintergrundfläche einfärben
    plt.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color='yellow', alpha=0.4)

    # Lastpunkte (blaue Punkte) darstellen
    plt.scatter(load_pos_x, load_pos_y, c='blue', alpha=0.5, label='Load Points')

    # Sensorpositionen als rotierte Rechtecke darstellen
    for (sx, sy), angle in zip(sensor_pos, rotation_angles):
        # Affine2D-Transformation zur Rotation um den Mittelpunkt
        trans = Affine2D().rotate_deg_around(sx, sy, angle) + plt.gca().transData

        # Rechteck hinzufügen, wobei `angle` die Rotation um den Mittelpunkt des Rechtecks angibt
        rect = patches.Rectangle(
            (sx - rectangle_width / 2, sy - rectangle_height / 2),  # Linke untere Ecke berechnen
            rectangle_width,
            rectangle_height,
            edgecolor='red',
            facecolor='red',
            alpha=0.6,
            transform=trans  # Transformation anwenden
        )
        plt.gca().add_patch(rect)  # Rechteck zum aktuellen Plot hinzufügen
        plt.text(sx, sy, f"Sensor R{sensor_pos.index((sx, sy)) + 1}",
                 fontsize=11, fontweight='bold', ha='center', color='black',
                 transform=plt.gca().transData + Affine2D().translate(0, 30),
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9)
                 )

    # Achsenlimits und Beschriftungen
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.xlabel('X-Achse [mm]')
    plt.ylabel('Y-Achse [mm]')
    plt.title('Lastpunkte (Blau), Sensorpatches (Rot)')
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.6)
    plt.axvline(0, color='gray', linewidth=0.6)
    plt.legend()
    plt.show()

# Funktion zum Erstellen des Scatterplots
def create_scatterplot_testing2(load_pos_x, load_pos_x_pred, load_pos_y, load_pos_y_pred, sensor_pos):
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

# Funktion zum Erstellen des Scatterplots
def create_scatterplot_testing(load_pos_x, load_pos_x_pred, load_pos_y, load_pos_y_pred, sensor_pos, rectangle_width=50, rectangle_height=20, rotation_angles=None):
    sensor_x, sensor_y = zip(*sensor_pos)  # Entpacke die Tupel in separate Listen für x und y

    # Falls keine Rotationswinkel angegeben sind, setzen wir sie alle auf 0 Grad
    if rotation_angles is None:
        rotation_angles = [0] * len(sensor_pos)  # Setze Standardrotationen, falls keine angegeben sind

    # Scatterplot erstellen
    plt.figure(figsize=(8, 8))  # Größere Figur für bessere Sichtbarkeit

    # Hintergrundfläche einfärben
    plt.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color='yellow', alpha=0.4)

    # Scatterplots mit Labels hinzufügen für die Lastpunkte
    for i, (x, y) in enumerate(zip(load_pos_x, load_pos_y), start=1):
        plt.scatter(x, y, c='blue', alpha=0.8)
        plt.text(x, y, str(i) + "  ", fontsize=7, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9))

    # Scatterplots für die vorhergesagten Lastpunkte
    for i, (x, y) in enumerate(zip(load_pos_x_pred, load_pos_y_pred), start=1):
        plt.scatter(x, y, c='green', alpha=0.5)
        plt.text(x, y, str(i) + "  ", fontsize=7, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9))

    # Sensorpositionen als rotierte Rechtecke darstellen
    for (sx, sy), angle in zip(sensor_pos, rotation_angles):
        # Affine2D-Transformation zur Rotation um den Mittelpunkt
        trans = Affine2D().rotate_deg_around(sx, sy, angle) + plt.gca().transData

        # Rechteck hinzufügen
        rect = patches.Rectangle(
            (sx - rectangle_width / 2, sy - rectangle_height / 2),  # Linke untere Ecke berechnen
            rectangle_width,
            rectangle_height,
            edgecolor='red',
            facecolor='red',
            alpha=0.6,
            transform=trans  # Transformation anwenden
        )
        plt.gca().add_patch(rect)  # Rechteck zum aktuellen Plot hinzufügen

        # Sensorbeschriftungen 30 Pixel über der aktuellen Position anzeigen, mit weißem Hintergrund
        plt.text(sx, sy, f"Sensor {sensor_pos.index((sx, sy)) + 1}",
                 fontsize=11, fontweight='bold', ha='center', color='black',
                 transform=plt.gca().transData + Affine2D().translate(0, 30),
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9)
                 )

    # Achsenlimits und Beschriftungen
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.xlabel('X-Achse [mm]')
    plt.ylabel('Y-Achse [mm]')
    plt.title('Memo-2D-Plot - Lastpunkte (Blau), Vorhergesagte Lastpunkte (Grün), Sensorpatches (Rot)')
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.legend()
    plt.show()
