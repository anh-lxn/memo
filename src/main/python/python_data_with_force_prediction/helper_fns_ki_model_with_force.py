"""
Dateiname: helper_fns_ki_model.py
Autor: Florian Schmidt
Datum: 30.05.2024

Beschreibung:
Dieses Skript stellt Hilfsfunktionen für die Verarbeitung, Normalisierung und Vorhersage von Dehnungsdaten durch.
Die Daten werden mithilfe eines neuronalen Netzmodells trainiert und getestet.
Das Modell kann gespeichert und später für Vorhersagen wieder geladen werden.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import datetime

from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setze den Random Seed für NumPy
#np.random.seed(42)
# Setze den Random Seed für Python
#random.seed(42)

# Setze den Random Seed für PyTorch auf allen Geräten (CPU und GPU), wenn verfügbar
#torch.manual_seed(42)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(42)

# Funktion zum Testen des Modells
def test_model(model, X_test, y_test):
    # Setzt das Modell in den Evaluierungsmodus. In diesem Modus werden einige Operationen, wie Dropout, deaktiviert.
    model.eval()
    # Deaktiviert das Tracking von Gradienten, um den Speicherverbrauch zu reduzieren und die Berechnung zu beschleunigen,
    # da keine Rückwärtsausbreitung durchgeführt wird.
    with torch.no_grad():
        # Führt einen Vorwärtsdurchlauf durch, um die Vorhersagen des Modells für die Testdaten zu erhalten.
        y_pred = model(X_test.to(device))
        # Berechnet die mittlere absolute Abweichung für die x- und y-Werte.
        # Dies erfolgt durch die absolute Differenz zwischen den vorhergesagten und den tatsächlichen x-Werten,
        # gefolgt von der Mittelwertbildung.
        accuracy_x = torch.abs(y_pred[:, 0] - y_test[:, 0].to(device)).mean().item()
        accuracy_y = torch.abs(y_pred[:, 1] - y_test[:, 1].to(device)).mean().item()
        print(f"Mittlere absolute Abweichung für x-Werte in [mm]: {accuracy_x}")
        print(f"Mittlere absolute Abweichung für y-Werte in [mm]: {accuracy_y}")

def prepare_data(strains, load_pos_x, load_pos_y, load_value, scale_data=False):
    # Sicherstellen, dass die Längen übereinstimmen
    if not all(len(strain) == len(load_pos_x) for strain in strains):
        raise ValueError("Die Längen der Eingabewerte stimmen nicht überein.")

    # Erstellen des DataFrames
    data = {f'strain_{i+1}': strain for i, strain in enumerate(strains)}
    df = pd.DataFrame(data)

    # Aufteilen der Daten in Trainings-, Validierungs- und Testdaten
    X = df.values
    y_combined = np.column_stack((load_pos_x, load_pos_y, load_value))

    # Aufteilen in Train-, Val- und Testdaten
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_combined, train_size=0.8, shuffle=True)  # 80% für Training
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)  # 10% für Validation, 10% für Test

    # Daten skalieren
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Daten in Tensor umwandeln
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)

    print(f'Trainingsdaten: {X_train.shape}, Validierungsdaten: {X_val.shape}, Testdaten: {X_test.shape}')
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train,X_test,y_train,y_test):
    # Definieren des neuronalen Netzes
    model = nn.Sequential(
        nn.Linear(8, 128), # 8 Inputs für die 8 Sensorwerte
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 6),
        nn.ReLU(),
        nn.Linear(6, 3)  # drei Outputs für x- und y-Koordinaten des Lasteinleitungspunktes mit der Kraft
    ).to(device)  # Modell auf GPU verschieben

    # Definition von Loss- und Optimierungsfunktionen
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.00025)

    n_epochs = 30000  # number of epochs to run

    # Listen für Training- und Test-Plots
    train_losses = []
    test_losses = []
    epochs = []

    # Training loop without batches
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device))
        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: Test-Loss = {loss.item()}")
            # Append loss and epoch
            with torch.no_grad():
                y_test_pred = model(X_test.to(device))
                test_loss = loss_fn(y_test_pred, y_test.to(device))
                test_losses.append(test_loss.item())
            train_losses.append(loss.item())
            epochs.append(epoch)

    # Plotting training and test loss curves
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Aufruf der test_model Funktion, um die Genauigkeit der Vorhersage für die Lasteinleitung zu ermitteln (über den Testdaten)
    test_model(model,X_test,y_test)

    #joblib.dump(scaler, 'scaler_v4.pkl') scaler speichern
    save_model(model, normalized="normalized")
    return model

def save_model(model, path_prefix='../../resources/models/model_demonstrator', normalized='notnormalized'):
    # Aktuelles Datum auslesen
    current_date = datetime.datetime.now()
    # Formatierung des Datums & Uhrzeit
    formatted_datetime = current_date.strftime('%d_%m_%Y_%H-%M-%S')
    filename = f"{path_prefix}_{normalized}_{formatted_datetime}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Funktion zum Laden des Modells
def load_model(path='../resources/models/model_demonstrator_v02_normalized.pth'):
    # Definieren des neuronalen Netzes
    model = nn.Sequential(
        nn.Linear(8, 128), # 8 Inputs für die 8 Sensorwerte
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 6),
        nn.ReLU(),
        nn.Linear(6, 3)  # zwei Outputs für x- und y-Koordinaten des Lasteinleitungspunktes
    ).to(device)

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

def test_random_samples(model, X_test, y_test, num_samples=2):
    # Wähle 10 zufällige Indizes aus dem Test-Set aus
    random_indices = random.sample(range(len(X_test)), num_samples)
    #print(random_indices)
    #print("X_test",X_test)
    #print("y_test:",y_test)
    #print("X_test[random_indices]",X_test[random_indices])
    #print("y_test[random_indices]",y_test[random_indices])
    X_sample = X_test[random_indices]
    y_sample = y_test[random_indices]

    #print(X_sample)
    # Wende das Modell auf die Eingabedaten an, um die Vorhersage zu erhalten
    model.eval()  # Setze das Modell in den Auswertungsmodus (deaktiviere Dropout und BatchNorm)
    with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
        y_pred = model(X_sample.to(device))

    return X_sample, y_sample, y_pred

def normalize_strain_data(strains):
    #### Normalisierung der Strain Listen
    # Umwandeln in ein numpy array für einfachere Handhabung
    strains_array = np.array(strains)

    # Listen mit min- und max-Werten erstellen
    min_vals = np.min(strains_array, axis=0)
    max_vals = np.max(strains_array, axis=0)

    # Vermeidung von Division durch Null
    range_vals = max_vals - min_vals
    range_vals[
        range_vals == 0] = 1  # Falls min und max gleich sind, setze den Bereich auf 1 um Division durch Null zu vermeiden

    # Normierte Werte berechnen
    normalized_strains = (strains_array - min_vals) / range_vals

    # In Listen zurückkonvertieren
    normalized_strain_list = normalized_strains.tolist()
    return normalized_strain_list

def test_all_samples(model, X_test, y_test):
    model.eval()  # Setze das Modell in den Auswertungsmodus (deaktiviere Dropout und BatchNorm)
    with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
        y_pred = model(X_test.to(device))

    return X_test, y_test, y_pred