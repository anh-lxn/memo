import helper_fns_data_with_force as h_fn
import helper_fns_ki_model_with_force as h_fn_ki
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import interp1d

# Parameter für verringerte Trainingsdaten
PERCENTAGE = 1.0
DECREASE = 0.1

# Hyperparameter
LEARNING_RATE= 0.00205
EPOCHS = 49501

# Listen für Optimizer und Loss-Funktionen
optimizer_list = ['Adam', 'SGD', 'RMSProp']
loss_function_list = ['MSE', 'MAE']
OPTIMIZER = optimizer_list[0]
LOSS_FUNCTION = loss_function_list[0]

# Listen zum Speichern der Werte (load_pos_x, load_pos_y sind Positionen der Lasteinleitung, strain_1 bis strain_8
# sind die resultierenden Dehnungswerte der Sensoren 1-8 (Auf Sensoranordnung achten!))
load_pos_x, load_pos_y, load_value, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7, strain_8 = [], [], [], [], [], [], [], [], [], [], []

# Pfad zur Gesamtdatei angeben
datei_pfad = "../../resources/interpolation/auswertung_gesamt_8N_10N_12N_15N_17N_18N_20N.csv"

# Die Datei zeilenweise einlesen
with open(datei_pfad, mode='r', newline='') as csv_datei:
    reader = csv.reader(csv_datei)

    # Kopfzeile extrahieren (erste Zeile)
    kopfzeile = next(reader)
    #print("Kopfzeile:", kopfzeile)

    # Jede weitere Zeile lesen und verarbeiten
    for zeile in reader:
        #print("Zeile:", zeile)
        load_pos_x.append(float(zeile[1]))
        load_pos_y.append(float(zeile[2]))
        load_value.append(float(zeile[3]))
        strain_1.append(float(zeile[4]))
        strain_2.append(float(zeile[5]))
        strain_3.append(float(zeile[6]))
        strain_4.append(float(zeile[7]))
        strain_5.append(float(zeile[8]))
        strain_6.append(float(zeile[9]))
        strain_7.append(float(zeile[10]))
        strain_8.append(float(zeile[11]))


strains = [strain_1,strain_2,strain_3,strain_4,strain_5,strain_6,strain_7,strain_8]

#### Normalisierung der Strain Listen
# Umwandeln in ein numpy array für einfachere Handhabung
strains_array = np.array(strains)

# Normieren der Werte für jeden Index
min_vals = np.min(strains_array, axis=0)
max_vals = np.max(strains_array, axis=0)

# Vermeidung von Division durch Null
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1  # Falls min und max gleich sind, setze den Bereich auf 1 um Division durch Null zu vermeiden

# Normierte Werte berechnen
normalized_strains = (strains_array - min_vals) / range_vals

# In Listen zurück konvertieren
normalized_strains_list = normalized_strains.tolist()
strain_1_norm = normalized_strains_list[0]
strain_2_norm = normalized_strains_list[1]
strain_3_norm = normalized_strains_list[2]
strain_4_norm = normalized_strains_list[3]
strain_5_norm = normalized_strains_list[4]
strain_6_norm = normalized_strains_list[5]
strain_7_norm = normalized_strains_list[6]
strain_8_norm = normalized_strains_list[7]
# Ergebnis
#print("normalized_strain_list: ", normalized_strains_list[:][:5])


# Datei-Name erstellen
file_name = f"reduced_data_training_results_{EPOCHS}_{LOSS_FUNCTION}_{OPTIMIZER}.csv"
file_path = f"../../resources/reduced_training/{file_name}"

def save_hypertuning_parameters_to_csv(file_path):
    global PERCENTAGE

    # Interne Liste zur Speicherung der Ergebnisse
    data_for_csv = []
    headers = ["Anzahl aller Daten", "Anzahl ausgewaehlter Daten", "Prozentualer Anteil", "Epochs", "Optimizer", "Loss Function", "Learning Rate", "Avg Train Loss", "Avg Test Loss"]
    data_for_csv.append(headers)

    # Füge die Zeilen der interpolierten Werte hinzu
    while PERCENTAGE >= 0.1:
        # Test und Trainingsdaten erstellen
        X_train, X_val, X_test, y_train, y_val, y_test = h_fn_ki.prepare_data(
            [strain_1_norm, strain_2_norm, strain_3_norm, strain_4_norm, strain_5_norm, strain_6_norm, strain_7_norm, strain_8_norm],
            load_pos_x, load_pos_y, load_value, round(PERCENTAGE, 2)
            )
        
        # Anzahl der Daten
        num_all_data = len(strain_1_norm)
        num_reduced_data = len(X_train) + len(X_val) + len(X_test)

        # Training
        print(f"Aktuelle Anzahl an Daten: {num_reduced_data} ({PERCENTAGE*100}%)")
        avg_train_loss, avg_test_loss = h_fn_ki.train_model_hypertuning(X_train, X_val, y_train, y_val, EPOCHS, LEARNING_RATE, OPTIMIZER, LOSS_FUNCTION)
        row = [num_all_data, num_reduced_data, round(PERCENTAGE*100, 2), EPOCHS, OPTIMIZER, LOSS_FUNCTION, round(LEARNING_RATE, 6), avg_train_loss, avg_test_loss]
        data_for_csv.append(row)
        PERCENTAGE -= DECREASE
        print(f"data_for_csv: {data_for_csv}")

    ### CSV-Datei für Hypertuning Plot
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_for_csv)

    print(f"CSV-Datei '{file_path}' wurde erfolgreich erstellt.")

def create_plot_hypertuning_parameters():
    # Lese die CSV-Datei ein
    data = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    # Entferne die Kopfzeile
    data = data[1:]

    # Extrahiere die Werte
    num_data = [float(row[1]) for row in data]
    end_train_loss = [float(row[7]) for row in data]
    end_test_loss = [float(row[8]) for row in data]
    
    # Kubische Interpolation
    interpolator_train_loss = interp1d(num_data, end_train_loss, kind='linear', fill_value="extrapolate")
    interpolator_test_loss = interp1d(num_data, end_test_loss, kind='linear', fill_value="extrapolate")

    # Erstelle feinere Werte für die Lernrate für glattere Kurven
    fine_num_data = np.linspace(min(num_data), max(num_data), 500)
    smoothed_train_loss = interpolator_train_loss(fine_num_data)
    smoothed_test_loss = interpolator_test_loss(fine_num_data)

    # Setze negative Werte auf 0
    smoothed_train_loss = np.clip(smoothed_train_loss, 0, None)
    smoothed_test_loss = np.clip(smoothed_test_loss, 0, None)

    # Plotting training and test loss curves
    plt.plot(fine_num_data, smoothed_train_loss, label='Average Train Loss')
    #plt.plot(fine_num_data, smoothed_test_loss, label='Average Test Loss')
    plt.xlabel('Anzahl Daten')
    plt.ylabel('Loss')
    plt.title('Train-Loss und Anzahl der Daten')
    plt.legend()
    plt.grid(True)
    plt.show()

#save_hypertuning_parameters_to_csv(file_path)
create_plot_hypertuning_parameters()
   