import os
import cv2
import numpy as np

# 🔹 Pfade
img_path = "images/features_knoten_0417.png"
skeleton_dir = "skeletons"
output_path = "skeletons_combined_overlay.png"

# 🔸 Originalbild laden
original = cv2.imread(img_path, cv2.IMREAD_COLOR)
if original is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")
h, w = original.shape[:2]

# 🎨 Kopie des Originals zum Zeichnen
overlay = original.copy()

# 🔁 Alle Skeletons zeichnen
for filename in os.listdir(skeleton_dir):
    if filename.endswith("_skeleton.png"):
        skel_path = os.path.join(skeleton_dir, filename)
        skeleton = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            print(f"⚠️ Fehler beim Laden: {skel_path}")
            continue

        # Resize falls nötig
        skeleton = cv2.resize(skeleton, (w, h))

        # Skeleton-Koordinaten finden (Pixel > 0)
        yx = np.argwhere(skeleton > 0)

        # 🔴 Rote Punkte zeichnen (du kannst auch Linien oder Konturen verwenden)
        for y, x in yx:
            cv2.circle(overlay, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

print(f"✅ Gezeichnete Skeletons: {len(os.listdir(skeleton_dir))}")
cv2.imwrite(output_path, overlay)
print(f"📷 Ergebnis gespeichert: {output_path}")
