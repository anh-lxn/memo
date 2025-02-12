import socket
import os
import time

DEST_IP = "192.200.2.2"  # IP-Adresse des Pi 4
PORT = 5001  # Gleicher Port wie auf Pi 4
FILE_PATH = "/home/anh.lxn/Documents/memo/src/main/resources/live_sync/test.csv"

last_mtime = None  # Speichert die letzte Änderungszeit

while True:
    if not os.path.exists(FILE_PATH):
        print(f"❌ Datei {FILE_PATH} nicht gefunden!")
        time.sleep(1)
        continue

    mtime = os.path.getmtime(FILE_PATH)
    if mtime != last_mtime:  # Falls die Datei geändert wurde
        print(f"⚡ Datei geändert → Senden an {DEST_IP}...")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((DEST_IP, PORT))
            with open(FILE_PATH, "rb") as f:
                client_socket.sendall(f.read())

        last_mtime = mtime

    time.sleep(0.1)  # Kurze Pause für CPU-Entlastung
