import socket

HOST = "0.0.0.0"  # Lauscht auf alle IP-Adressen
PORT = 5001  # Port für Verbindung

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print(f"📡 Warten auf Verbindung auf Port {PORT}...")

    while True:
        conn, addr = server_socket.accept()
        with conn:
            print(f"✅ Verbunden mit {addr}")

            with open("/home/anh.lxn/Documents/memo/src/main/resources/live_sync/live_data.csv", "wb") as f:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    f.write(data)

            print("💾 Datei aktualisiert!")
