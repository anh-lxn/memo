from PIL import Image, ImageDraw, ImageFont

def generate_grid_image(filename="grid.jpg"):
    width, height = 1028, 1028  # Bildgröße beibehalten
    img = Image.new("RGB", (width, height), "white")  # Weißes Bild erstellen
    draw = ImageDraw.Draw(img)
    
    # Font für die Koordinaten (falls verfügbar)
    try:
        font = ImageFont.truetype("arial.ttf", 8)
    except IOError:
        font = None
    
    # Abschnitte mit verschiedenen Farben
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i in range(4):
        x_offset = i * 64  # Startpunkt für jeden Abschnitt
        for x in range(64):
            for y in range(256):
                # Rasterlinien zeichnen
                if x % 8 == 0 or y % 8 == 0:
                    draw.point((x + x_offset, 255 - y), fill=(0, 0, 0))  # Schwarz für Raster
                else:
                    draw.point((x + x_offset, 255 - y), fill=colors[i])  # Färbung nach Abschnitt
                
                # Koordinaten als Text einfügen
                if x % 16 == 0 and y % 16 == 0:  # Alle 16 Pixel beschriften
                    coord_text = f"({x},{y + i * 256})"
                    if font:
                        draw.text((x + x_offset + 2, 255 - y - 6), coord_text, fill="black", font=font)
                    else:
                        draw.text((x + x_offset + 2, 255 - y - 6), coord_text, fill="black")
    
    img.save(filename, "JPEG")
    print(f"Bild gespeichert als {filename}")

# Funktion aufrufen
generate_grid_image()
