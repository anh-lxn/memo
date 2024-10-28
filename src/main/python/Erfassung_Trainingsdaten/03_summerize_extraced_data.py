import matplotlib.pyplot as plt
import seaborn as sns
import random

# Netzwerkarchitektur
layers = [8, 16, 2]  # Anzahl der Neuronen pro Schicht

# Farben und Stil
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Position der Neuronen auf der y-Achse
y_offsets = []
for layer_size in layers:
    y_offsets.append(range(layer_size))

# Zeichne die Verbindungen zwischen den Schichten
for i in range(len(layers) - 1):
    connections_drawn = 0
    for y1 in y_offsets[i]:
        if connections_drawn >= 5:  # Zeichne nur 5 vollständige Verbindungen pro Neuron
            break
        for y2 in random.sample(y_offsets[i + 1], min(5, layers[i + 1])):  # 5 Verbindungen zu zufälligen Neuronen
            plt.plot([i, i + 1], [y1, y2], 'k-', lw=0.3, alpha=0.4)  # dünne vollständige Linie
            connections_drawn += 1

    # Zeichne gestrichelte Linie nur, wenn genügend Neuronen vorhanden sind
    if len(y_offsets[i]) > 4 and len(y_offsets[i + 1]) > 4:
        plt.plot([i, i + 1], [y_offsets[i][4], y_offsets[i + 1][4]], 'k--', lw=0.3, alpha=0.4)

# Zeichne die Neuronen
for i, layer_size in enumerate(layers):
    plt.scatter([i] * layer_size, y_offsets[i], s=100, color='skyblue', edgecolor='k', zorder=5)

# Schicht-Labels hinzufügen
layer_labels = ["Input Layer"] + [f"Hidden Layer {i}" for i in range(1, len(layers) - 2)] + ["Output Layer"]
for i, label in enumerate(layer_labels):
    plt.text(i, max(layers) + 20, label, ha="center", va="center", fontweight="bold")

# Anpassungen für eine klare Visualisierung
plt.title("Vereinfachte Visualisierung des neuronalen Netzwerks", fontsize=14)
plt.xlabel("Layer", fontsize=12)
plt.xticks(range(len(layers)), [])
plt.yticks([])
plt.grid(False)
plt.show()
