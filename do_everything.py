import os
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from skimage.io import imsave
from ultralytics import YOLO
from pyproj import Transformer
import pandas as pd
from pathlib import Path
from polygon_centerline import polygon_centerline


# 📁 Ordner vorbereiten
input_dir_gml = "gml_data"
output_dir_images = "images"
output_dir_masks = "masks_png"
output_dir_vectors = "vectors"
output_dir_final = "output_final"
os.makedirs(input_dir_gml, exist_ok=True)
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)
os.makedirs(output_dir_vectors, exist_ok=True)
os.makedirs(output_dir_final, exist_ok=True)

# 🎨 Bildgröße
target_size = (1024, 1024)

# 🌐 Koordinatenreferenzsysteme
source_crs = "EPSG:25832"  # Annahme: GML in UTM Zone 32N
target_crs = "EPSG:4326"   # WGS84 für Geokoordinaten
transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

def chaikin_smooth(coords, iterations=2):
    """Glättet eine Linie mit Chaikins Algorithmus für weichere Rundungen."""
    points = np.array(coords)
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_points.append(q)
            new_points.append(r)
        points = np.array(new_points)
    return points.tolist()

def render_gml_to_image(gml_file, output_file):
    """GML-Datei in PNG rendern, Geometrien mit Skalierungsinformationen zurückgeben und zwei Geokoordinaten einbetten."""
    try:
        gdf = gpd.read_file(gml_file)
        points = []
        for geom in gdf.geometry:
            if geom.geom_type in ["LineString", "MultiLineString"]:
                if geom.geom_type == "LineString":
                    points.extend(geom.coords)
                else:
                    for line in geom.geoms:
                        points.extend(line.coords)
        
        if not points:
            print(f"Keine Linien in {gml_file}. Überspringe.")
            return None, None, None, None
        
        points = np.array(points)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        scale = min(target_size[0] / (max_x - min_x + 1e-6), target_size[1] / (max_y - min_y + 1e-6))
        
        img = Image.new("RGB", target_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        for geom in gdf.geometry:
            if geom.geom_type == "LineString":
                scaled_points = [((p[0] - min_x) * scale, (p[1] - min_y) * scale) for p in geom.coords]
                draw.line(scaled_points, fill=(0, 0, 0), width=3)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    scaled_points = [((p[0] - min_x) * scale, (p[1] - min_y) * scale) for p in line.coords]
                    draw.line(scaled_points, fill=(0, 0, 0), width=3)
        
        bounds = gdf.total_bounds
        lower_corner = (bounds[0], bounds[1])
        upper_corner = (bounds[2], bounds[3])
        
        lower_corner_wgs84 = transformer.transform(lower_corner[0], lower_corner[1])
        upper_corner_wgs84 = transformer.transform(upper_corner[0], upper_corner[1])
        
        lower_corner_img = (0, target_size[1] - 1)
        upper_corner_img = (target_size[0] - 1, 0)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        lower_text = f"({lower_corner_wgs84[0]:.6f}, {lower_corner_wgs84[1]:.6f})"
        upper_text = f"({upper_corner_wgs84[0]:.6f}, {upper_corner_wgs84[1]:.6f})"
        
        draw.text((lower_corner_img[0] + 10, lower_corner_img[1] - 20), lower_text, fill=(255, 0, 0), font=font)
        draw.text((upper_corner_img[0] - 100, upper_corner_img[1] + 10), upper_text, fill=(255, 0, 0), font=font)
        
        img.save(output_file)
        print(f"✅ Gerendert mit Geokoordinaten: {output_file}")
        return points, min_x, min_y, scale
    
    except Exception as e:
        print(f"❌ Fehler bei {gml_file}: {str(e)}")
        return None, None, None, None

def apply_yolo_model(img_path, model_path="best.pt"):
    """YOLO-Modell auf Bild anwenden, Masken speichern und Pixelmasken zurückgeben."""
    try:
        model = YOLO(model_path)
        results = model(img_path, task="segment")
        res = results[0]
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")
        h_orig, w_orig = img.shape[:2]
        
        masks = []
        for i, mask in enumerate(res.masks.data):
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            mask_path = os.path.join(output_dir_masks, f"mask_{i}_{os.path.basename(img_path)}")
            Image.fromarray(mask_resized).save(mask_path)
            print(f"✅ Maske gespeichert: {mask_path}")
            masks.append(mask_resized)
        
        return masks, h_orig, w_orig
    
    except Exception as e:
        print(f"❌ Fehler bei YOLO-Inferenz: {str(e)}")
        return None, None, None

def vectorize_masks(masks, img_path, iterations=2, tolerance=3.0):
    """Pixelmasken in geschlossene, geglättete Polygone umwandeln und speichern."""
    vectorized_data = []
    
    for i, mask in enumerate(masks):
        # Konturen aus der Pixelmaske extrahieren
        mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Keine Konturen in Maske {i} gefunden. Überspringe.")
            continue
        
        # Alle Konturen zu Polygonen vereinen
        polygons = []
        for cnt in contours:
            if len(cnt) >= 3:  # Mindestens 3 Punkte für ein Polygon
                points = cnt.squeeze()
                if points.ndim == 1:  # Einzelpunkt verhindern
                    continue
                # Konturpunkte glätten mit Chaikins Algorithmus
                rounded_coords = chaikin_smooth(points, iterations=iterations)
                # Polygon erstellen
                try:
                    polygon = Polygon(rounded_coords)
                    if not polygon.is_valid:
                        print(f"Ungültiges Polygon in Maske {i}. Überspringe.")
                        continue
                    # Glättung mit simplify
                    smoothed_polygon = polygon.simplify(tolerance, preserve_topology=True)
                    if not smoothed_polygon.is_valid:
                        print(f"Ungültiges geglättetes Polygon in Maske {i}. Überspringe.")
                        continue
                    polygons.append(smoothed_polygon)
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von Kontur in Maske {i}: {str(e)}")
                    continue
        
        if not polygons:
            print(f"Keine gültigen Polygone in Maske {i}. Überspringe.")
            continue
        
        # Polygone vereinen
        union_polygon = unary_union(polygons)
        if not union_polygon.is_valid:
            print(f"Ungültiges vereintes Polygon in Maske {i}. Überspringe.")
            continue
        
        # Polygon als Bild speichern
        polygon_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        if union_polygon.geom_type == "Polygon":
            x, y = union_polygon.exterior.xy
            points = np.array(list(zip(x, y)), dtype=np.int32)
            cv2.fillPoly(polygon_img, [points], color=255)
        elif union_polygon.geom_type == "MultiPolygon":
            for poly in union_polygon.geoms:
                x, y = poly.exterior.xy
                points = np.array(list(zip(x, y)), dtype=np.int32)
                cv2.fillPoly(polygon_img, [points], color=255)
        
        polygon_path = os.path.join(output_dir_vectors, f"vectorized_polygon_{i}_{os.path.basename(img_path)}")
        imsave(polygon_path, polygon_img)
        print(f"✅ Vektorisiertes Polygon gespeichert: {polygon_path}")
        
        vectorized_data.append(union_polygon)
    
    return vectorized_data

def plot_skeletons_on_image(img_path, vectorized_data, output_path, scale_info):
    """Geglättete Polygone auf Originalbild überlagern und Vektoren speichern."""
    try:
        original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")
        overlay = original.copy()
        h, w = original.shape[:2]
        
        vectors = []
        min_x, min_y, scale = scale_info
        
        for i, polygon in enumerate(vectorized_data):
            # Polygon zeichnen (rot, geschlossen)
            if polygon.geom_type == "Polygon":
                x, y = polygon.exterior.xy
                points = np.array(list(zip(x, y)))
                points = [(int(x), int(y)) for x, y in points if 0 <= x < w and 0 <= y < h]
                if len(points) >= 3:
                    cv2.polylines(overlay, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=1)
                
                # Koordinaten für CSV
                yx = np.array(list(zip(x, y)))
                if len(yx) == 0:
                    print(f"Keine gültigen Polygonpunkte für Polygon {i}. Überspringe.")
                    continue
                original_points = [(x / scale + min_x, y / scale + min_y) for x, y in yx]
                geo_points = [transformer.transform(x, y) for x, y in original_points]
                vectors.append({
                    "polygon_id": i,
                    "image_coordinates": yx.tolist(),
                    "original_coordinates": original_points,
                    "geocoordinates": geo_points
                })
            elif polygon.geom_type == "MultiPolygon":
                for j, poly in enumerate(polygon.geoms):
                    x, y = poly.exterior.xy
                    points = np.array(list(zip(x, y)))
                    points = [(int(x), int(y)) for x, y in points if 0 <= x < w and 0 <= y < h]
                    if len(points) >= 3:
                        cv2.polylines(overlay, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=1)
                    
                    # Koordinaten für CSV
                    yx = np.array(list(zip(x, y)))
                    if len(yx) == 0:
                        print(f"Keine gültigen Polygonpunkte für Polygon {i}.{j}. Überspringe.")
                        continue
                    original_points = [(x / scale + min_x, y / scale + min_y) for x, y in yx]
                    geo_points = [transformer.transform(x, y) for x, y in original_points]
                    vectors.append({
                        "polygon_id": f"{i}.{j}",
                        "image_coordinates": yx.tolist(),
                        "original_coordinates": original_points,
                        "geocoordinates": geo_points
                    })
        
        cv2.imwrite(output_path, overlay)
        print(f"✅ Überlagertes Bild gespeichert: {output_path}")
        
        vector_path = os.path.join(output_dir_vectors, os.path.basename(img_path).replace(".png", "_vectors.csv"))
        df_vectors = pd.DataFrame(vectors)
        df_vectors.to_csv(vector_path, index=False)
        print(f"✅ Vektoren gespeichert: {vector_path}")
        
        return vectors
    
    except Exception as e:
        print(f"❌ Fehler beim Plotten: {str(e)}")
        return None

def main():
    gml_files = [f for f in os.listdir(input_dir_gml) if f.lower().endswith(".gml")]
    if not gml_files:
        print("❌ Keine GML-Dateien in gml_data gefunden.")
        return
    
    for gml_file in gml_files:
        print(f"📂 Verarbeite GML-Datei: {gml_file}")
        gml_path = os.path.join(input_dir_gml, gml_file)
        
        img_path = os.path.join(output_dir_images, gml_file.replace(".gml", ".png"))
        points, min_x, min_y, scale = render_gml_to_image(gml_path, img_path)
        if points is None:
            continue
        
        masks, h_orig, w_orig = apply_yolo_model(img_path)
        if masks is None:
            continue
        
        vectorized_data = vectorize_masks(masks, img_path, iterations=10, tolerance=10.0)
        if not vectorized_data:
            print("❌ Keine vektorisierten Polygone generiert.")
            continue
        
        output_path = os.path.join(output_dir_final, gml_file.replace(".gml", "_vectorized_overlay.png"))
        vectors = plot_skeletons_on_image(img_path, vectorized_data, output_path, (min_x, min_y, scale))
        
                # ➕ Schritt: Mittellinien berechnen & als gemeinsames Overlay speichern
        original = cv2.imread(img_path)
        if original is None:
            print("⚠️ Konnte Bild für Mittellinien-Overlay nicht laden.")
            continue
        overlay = original.copy()

        for i, polygon in enumerate(vectorized_data):
            if polygon.geom_type != "Polygon":
                continue
            try:
                print(f"➕ Berechne Mittellinie für Polygon {i}")
                centerline = polygon_centerline(polygon, dx=10.0, smooth_window=21, smooth_order=2, show_plots=False)

                # Mittellinie auf das gemeinsame Overlay zeichnen
                x_line, y_line = centerline.xy
                for j in range(len(x_line) - 1):
                    pt1 = (int(x_line[j]), int(y_line[j]))
                    pt2 = (int(x_line[j + 1]), int(y_line[j + 1]))
                    cv2.line(overlay, pt1, pt2, color=(0, 255, 0), thickness=2)

            except Exception as e:
                print(f"⚠️ Fehler bei Mittellinie für Polygon {i}: {e}")

        # Gemeinsames Mittellinienbild speichern
        centerlines_path = os.path.join(output_dir_final, f"centerlines_{os.path.basename(img_path)}")
        cv2.imwrite(centerlines_path, overlay)
        print(f"✅ Gemeinsames Mittellinienbild gespeichert: {centerlines_path}")

    
    print("🎉 Verarbeitung aller Dateien abgeschlossen!")

if __name__ == "__main__":
    main()