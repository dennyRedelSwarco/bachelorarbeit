import os
from skimage.io import imread, imsave
from skimage.morphology import skeletonize
import numpy as np

input_dir = "masks_png"
output_dir = "skeletons"
os.makedirs(output_dir, exist_ok=True)

# Alle Maske-Dateien durchgehen
for filename in os.listdir(input_dir):
    if filename.startswith("mask_") and filename.endswith(".png") and "rgba" not in filename:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".png", "_skeleton.png"))

        # Bild laden und in binär umwandeln
        img = imread(input_path, as_gray=True)
        binary = img > 127  # Schwelle setzen (Bild ist 8-bit, 0–255)

        # Skeletonisierung
        skeleton = skeletonize(binary)

        # Ergebnis speichern
        imsave(output_path, (skeleton * 255).astype(np.uint8))
        print(f"✅ Skeleton gespeichert: {output_path}")
