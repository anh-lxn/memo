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
        y_pred = model(X_test)
        # Berechnet die mittlere absolute Abweichung für die x- und y-Werte.
        # Dies erfolgt durch die absolute Differenz zwischen den vorhergesagten und den tatsächlichen x-Werten,
        # gefolgt von der Mittelwertbildung.
        accuracy_x = torch.abs(y_pred[:, 0] - y_test[:, 0]).mean().item()
        accuracy_y = torch.abs(y_pred[:, 1] - y_test[:, 1]).mean().item()
        print(f"Mittlere absolute Abweichung für x-Werte in [mm]: {accuracy_x}")
        print(f"Mittlere absolute Abweichung für y-Werte in [mm]: {accuracy_y}")

# Datenvorbereitung und Modelltraining
def prepare_data(strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7, strain_8,
                           load_pos_x, load_pos_y):

    print("test and train gestartet")
    # Read data
    data = {'strain_1':strain_1, 'strain_2':strain_2, 'strain_3':strain_3,
            'strain_4':strain_4, 'strain_5':strain_5, 'strain_6':strain_6, 'strain_7': strain_7,
            'strain_8':strain_8}

    df = pd.DataFrame(data)

    # Aufteilung in Trainings- und Testdaten
    X = df[['strain_1', 'strain_2','strain_3', 'strain_4', 'strain_5', 'strain_6',
            'strain_7', 'strain_8']].values

    # Erstelle numpy Array mit Kombination aus x- und y-Koordinaten
    y_combined = np.column_stack((load_pos_x, load_pos_y))

    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y_combined, train_size=0.8, shuffle=True)

    # Daten skalieren
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Daten in Tensor umwandeln
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)

    return X_train, X_test,y_train,y_test

def train_model(X_train,X_test,y_train,y_test):
    # Definieren des neuronalen Netzes
    model = nn.Sequential(
        nn.Linear(8, 128), # 8 Inputs für die 8 Sensorwerte
        nn.ReLU(),
        nn.Linear(128, 6),
        nn.ReLU(),
        nn.Linear(6, 2)  # zwei Outputs für x- und y-Koordinaten des Lasteinleitungspunktes
    )

    # Definition von Loss- und Optimierungsfunktionen
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 46000   # number of epochs to run

    # Listen für Training- und Test-Plots
    train_losses = []
    test_losses = []
    epochs = []

    # Training loop without batches
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: Test-Loss = {loss.item()}")
            # Append loss and epoch
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred, y_test)
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
    save_model(model)
    return model

def save_model(model, path_prefix='../resources/models/model_demonstrator'):
    # Aktuelles Datum auslesen
    current_date = datetime.datetime.now()
    # Formatierung des Datums & Uhrzeit
    formatted_datetime = current_date.strftime('%d_%m_%Y_%H-%M-%S')
    filename = f"{path_prefix}_{formatted_datetime}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Funktion zum Laden des Modells
def load_model(path='../resources/models/model_demonstrator_v02_normalized.pth'):
    # Definieren des neuronalen Netzes
    model = nn.Sequential(
        nn.Linear(8, 128), # 8 Inputs für die 8 Sensorwerte
        nn.ReLU(),
        nn.Linear(128, 6),
        nn.ReLU(),
        nn.Linear(6, 2)  # zwei Outputs für x- und y-Koordinaten des Lasteinleitungspunktes
    )

    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

def test_random_samples(model, X_test, y_test, num_samples=15):
    # Wähle 10 zufällige Indizes aus dem Test-Set aus
    random_indices = random.sample(range(len(X_test)), num_samples)
    print(random_indices)
    #print("X_test",X_test)
    #print("y_test:",y_test)
    #print("X_test[random_indices]",X_test[random_indices])
    #print("y_test[random_indices]",y_test[random_indices])
    X_sample = X_test[random_indices]
    y_sample = y_test[random_indices]

    print(X_sample)
    # Wende das Modell auf die Eingabedaten an, um die Vorhersage zu erhalten
    model.eval()  # Setze das Modell in den Auswertungsmodus (deaktiviere Dropout und BatchNorm)
    with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
        y_pred = model(X_sample)

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

    # In Listen zurück konvertieren
    normalized_strain_list = normalized_strains.tolist()
    return normalized_strain_list