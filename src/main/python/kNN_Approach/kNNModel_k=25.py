import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score  # Für die Berechnung der Genauigkeit
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time

# CSV-Datei laden (ersetze 'deine_datei.csv' durch den Pfad zu deiner Datei)
data = pd.read_csv('gesamt_simulations_auswertung_edit_k=25.csv')

centers = [(-280,280),(-140,280),(0,280),(140,280),(280,280),
           (-280,140),(-140,140),(0,140),(140,140),(280,140),
           (-280,0),(-140,0),(0,0),(140,0),(280,0),
           (-280,-140),(-140,-140),(0,-140),(140,-140),(280,-140),
           (-280,-280),(-140,-280),(0,-280),(140,-280),(280,-280)]

# Umwandlung in numpy Array
centers_arr = np.array((centers))

# Features (Sensorwerte) extrahieren: Spalten von 'Strain_1' bis 'Strain_8'
X = data[['Strain_1', 'Strain_2', 'Strain_3', 'Strain_4', 'Strain_5', 'Strain_6', 'Strain_7', 'Strain_8']].values

# Label (Datei_ID) extrahieren: 'Datei_ID' als Zielvariable
y = data['Class'].values

# Daten in Trainings- und Testset aufteilen (95% Training, 5% Test)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, data.index, test_size=0.1, random_state=40)

# Feature-Skalierung für bessere Leistung von kNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN Klassifikator mit k=9 erstellen
knn = KNeighborsClassifier(n_neighbors=25)

# Modell mit Trainingsdaten trainieren
knn.fit(X_train, y_train)

start = time.perf_counter()
# Vorhersage für das Testset treffen
predictions = knn.predict(X_test)
end = time.perf_counter()

print(f"Laufzeit für Prediction: {(end-start)/len(X_test):.6f} Sekunden pro Testdatenset")


# Berechnung der Genauigkeit (wie gut das Modell auf den Testdaten performt)
accuracy = accuracy_score(y_test, predictions)
print(f"Genauigkeit des Modells auf den Testdaten: {accuracy:.2f}")

# Falls du die Wahrscheinlichkeiten für jede Klasse sehen möchtest
prediction_prob = knn.predict_proba(X_test)


# Ausgabe der Vorhersagewahrscheinlichkeiten zusammen mit den zugehörigen Zeilen der CSV
print("\nVorhersagewahrscheinlichkeiten für das Testset zusammen mit den zugehörigen CSV-Zeilen:")
for idx, prob in zip(indices_test, prediction_prob):
    prob_array = np.array(prob)
    weighted_centers = centers_arr * prob_array.reshape(-1,1)
    predicted_coordinate = weighted_centers.sum(axis=0)
    print(f"Zeile (Original Index: {idx}):")
    print(f"Vorhersagewahrscheinlichkeiten: {prob}")
    print("Mittlere vorhergesagte Koordinate:", predicted_coordinate)
    print(f"Zugehörige CSV-Zeile: {data.iloc[idx].to_dict()}")
    print("-" * 80)


# Confusion Matrix berechnen
cm = confusion_matrix(y_test, predictions)

# Confusion Matrix als Heatmap plotten
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

