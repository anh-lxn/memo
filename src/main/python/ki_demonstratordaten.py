import helper_fns_data as h_fns
# Definieren von id, x und y Position
id = 33
x = 290
y = 174
F = 50
dt = 0.02
timepoints = [5, 17]  # Beispielsweise 4s und 12s

filepath = f'messdaten/0{id}_{x}_{y}_{F}/Brücke_(V_V).txt'

# Beispielaufruf mit den extrahierten Daten
strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7 = h_fns.read_data_from_txt(filepath)

# Daten in eine Liste packen
strain_data = [strain_0, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7]

# Ausgabe der ersten paar Werte zur Überprüfung
for i in enumerate(strain_data, start=1):
    print(f"strain_{i}:", strain_1[:20])

# Plotten der Daten
#h_fns.plot_strain_data(strain_data)

# Parameter für den Tiefpassfilter
cutoff = 0.8  # Grenzfrequenz in Hz
fs = 15     # Abtastrate in Hz (1/0.05s = 20Hz)
order = 5     # Filterordnung

# Anwenden des Filters auf alle strain-Listen
filtered_strain_lists = [h_fns.butter_lowpass_filter(strain, cutoff, fs, order) for strain in strain_data]

# Plotten der Daten
h_fns.plot_strain_data_filtered(strain_data, filtered_strain_lists, dt)

# Speichern der Daten in einer CSV-Datei
output_csv_path = f'0{id}_strain_values_{x}_{y}_{F}.csv'
h_fns.save_strain_values_to_csv(timepoints, filtered_strain_lists, output_csv_path, dt, x, y ,F)


