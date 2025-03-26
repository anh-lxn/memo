import csv

# CSV-Datei
FILE_NAME = "live_data.csv"
#FILE_PATH = f"../../resources/live_sync/{FILE_NAME}"
FILE_PATH = f"/home/memo/Documents/remote_rsync/src/main/resources/live_sync/{FILE_NAME}"

# CSV-Datei auslesen
with open(FILE_PATH, mode='r') as csv_file:
    reader = csv.reader(csv_file)
        
    for spalte in reader:
        x_coordinate_original = float(spalte[0])
        y_coordinate_original = float(spalte[1])
        load_force = float(spalte[2])

# Koordinatensystem von Membranfeld auf LED-Board transformieren 
# Koordinatensystem von Membranfeld auf LED-Board transformieren (inkl. 180° Drehung)
def map_value(val, from_min, from_max, to_min, to_max):
    return (val - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

def transform_coords(x2, y2):
    # Skalierung von [-290, 290] → [0, 255]
    x1 = map_value(x2, -290, 290, 0, 255)
    y1 = map_value(y2, -290, 290, 0, 255)
    # 180° Drehung (Spiegelung an X- und Y-Achse)
    x1 = 255 - x1
    y1 = 255 - y1
    return int(x1), int(y1)

x_coordinate_led, y_coordinate_led = transform_coords(x_coordinate_original, y_coordinate_original)


print(f"X_original: {x_coordinate_original}")
print(f"Y_original: {y_coordinate_original}")
print(f"X: {x_coordinate_led}")
print(f"Y: {y_coordinate_led}")

    