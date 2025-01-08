import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os
import csv
import helper_fns_ki_model_with_force as h_fn_ki

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 3  # 3 Outputs: x, y, Force
DIR = os.getcwd()
EPOCHS = 15000  # Je nach Bedarf anpassen

### Experimentell ermittelte Daten
datei_pfad = "../../resources/interpolation/auswertung_gesamt_8N_10N_12N_15N_17N_18N_20N.csv"  # <-- Pfad auf Raspberry anpassen

# Listen zum Speichern der Werte
load_pos_x, load_pos_y, load_value, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7, strain_8 = [], [], [], [], [], [], [], [], [], [], []

# Die Datei zeilenweise einlesen
with open(datei_pfad, mode='r', newline='') as csv_datei:
    reader = csv.reader(csv_datei) # Die .csv Datei einlesen
    kopfzeile = next(reader) # Kopfzeile extrahieren (erste Zeile)
    for zeile in reader: # Jede weitere Zeile lesen und verarbeiten
        load_pos_x.append(float(zeile[1])) # X Position der Lasteinleitung speichern
        load_pos_y.append(float(zeile[2])) # Y Position der Lasteinleitung speichern
        load_value.append(float(zeile[3])) # Kraft der Lasteinleitung speichern
        strain_1.append(float(zeile[4])) # Sensor R1 Werte speichern
        strain_2.append(float(zeile[5])) # Sensor R2 Werte speichern
        strain_3.append(float(zeile[6])) # Sensor R3 Werte speichern
        strain_4.append(float(zeile[7])) # Sensor R4 Werte speichern
        strain_5.append(float(zeile[8])) # Sensor R5 Werte speichern
        strain_6.append(float(zeile[9])) # Sensor R6 Werte speichern
        strain_7.append(float(zeile[10])) # Sensor R7 Werte speichern
        strain_8.append(float(zeile[11])) # Sensor R8 Werte speichern

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


# Dein fixes Modell (identisch zu dem, was du vorher hattest)
class FixedModel(nn.Module):
    def __init__(self):
        super(FixedModel, self).__init__()
        self.layer1 = nn.Linear(8, 128)  # 8 Eingaben für die 8 Sensorwerte
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(512, 128)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(128, 128)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(128, 6)  # 6 Ausgabefeatures (z.B. x- und y-Koordinaten)
        self.relu6 = nn.ReLU()
        self.output_layer = nn.Linear(6, 3)  # 3 Outputs (z.B. x-, y-Koordinaten der Lastenposition)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = self.relu5(self.layer5(x))
        x = self.relu6(self.layer6(x))
        x = self.output_layer(x)
        return x
    

def define_model(trial):
    # Optuna schlägt die Anzahl der Schichten, Einheiten und Dropout-Rate vor
    n_layers = trial.suggest_int("n_layers", 5, 5)
    layers = []

    in_features = 8  # 8 Strain-Sensoren als Eingabe
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 128, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        # Optional: Dropout für jede Schicht
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)  # Dropout-Rate zwischen 0.2 und 0.5
        layers.append(nn.Dropout(p))

        in_features = out_features

    # Ausgabe-Schicht
    layers.append(nn.Linear(in_features, 6))  # 6 Ausgabefeatures (für 6 Werte z.B. x- und y-Koordinaten)
    layers.append(nn.LogSoftmax(dim=1))  # Softmax für die Klassifizierung (falls zutreffend)

    return nn.Sequential(*layers).to(DEVICE)

def objective(trial):
    # Modell definieren
    model = FixedModel().to(DEVICE)
    print("Device: ", DEVICE)

    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer_name = "Adam"
    lr = trial.suggest_float("lr", 0.0002, 0.0003, step=0.00002, log=False)
    lr = round(lr, 6)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    accuracy_list = []
    mittelwert_accuracy= 0

    # Get dataset
    X_train, X_val, X_test, y_train, y_val, y_test = h_fn_ki.prepare_data(
        [strain_1_norm, strain_2_norm, strain_3_norm, strain_4_norm, strain_5_norm, strain_6_norm, strain_7_norm, strain_8_norm],
        load_pos_x, load_pos_y, load_value
        )
    
    # Training des Modells
    # Definition von Loss- und Optimierungsfunktionen
    loss_fn = nn.MSELoss()  # mean square error

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X_train.to(DEVICE))
        loss = loss_fn(y_pred, y_train.to(DEVICE))
        # Backward pass
        loss.backward()
        optimizer.step()

        # Deaktiviert das Tracking von Gradienten, um den Speicherverbrauch zu reduzieren und die Berechnung zu beschleunigen,
        # da keine Rückwärtsausbreitung durchgeführt wird
        
        with torch.no_grad():
            y_test_pred = model(X_test.to(DEVICE))
            test_loss = loss_fn(y_test_pred, y_test.to(DEVICE))

           
        if epoch % 500 == 0:
            accuracy_list = []
            print(f"Epoch {epoch}: Loss = {round(loss.item(), 2)}; Test-Loss = {round(test_loss.item(), 2)}; Average-Test-Loss = {round(mittelwert_accuracy, 2)}")

        accuracy_list.append(test_loss.item())
        mittelwert_accuracy = np.mean(accuracy_list)
        trial.report(mittelwert_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mittelwert_accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, timeout=600)  # n_trials und timeout anpassen

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
