from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os

# 📁 Ordner vorbereiten
os.makedirs("masks_png", exist_ok=True)
os.makedirs("polygons", exist_ok=True)
os.makedirs("polygons_normed", exist_ok=True)

# 📷 Bildpfad & Modell laden
img_path = "images/features_knoten_0417.png"
model = YOLO("best.pt")

# 🔍 Inferenz durchführen
results = model(img_path, task="segment")
res = results[0]

# 🔄 Bild laden & Größe merken
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")
h_orig, w_orig = img.shape[:2]

# 1️⃣ Binärmasken speichern (.png, originalgröße)
for i, mask in enumerate(res.masks.data):
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # 0/255
    mask_resized = cv2.resize(mask_np, (w_orig, h_orig))   # Maske an Originalbild anpassen
    Image.fromarray(mask_resized).save(f"masks_png/mask_{i}.png")

# 2️⃣ Polygon (Pixel)
for i, poly in enumerate(res.masks.xy):
    np.savetxt(f"polygons/mask_{i}_xy.txt", poly, fmt="%.2f")
    print(f"Maske {i} Pixel-Polygon shape:", poly.shape)

# 3️⃣ Polygon (normalisiert)
for i, poly_n in enumerate(res.masks.xyn):
    np.savetxt(f"polygons_normed/mask_{i}_xyn.txt", poly_n, fmt="%.6f")
    print(f"Maske {i} Normed-Polygon shape:", poly_n.shape)

# 4️⃣ RGBA-Masken (transparente Überlagerung)
for i, mask in enumerate(res.masks.data):
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_np, (w_orig, h_orig))
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask_resized])
    cv2.imwrite(f"masks_png/mask_{i}_rgba.png", rgba)

print("✅ Alle Formate wurden erfolgreich exportiert.")
