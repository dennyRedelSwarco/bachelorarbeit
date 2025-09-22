# do_everything_app.py

import os
import geopandas as gpd
from shapely.geometry import LineString, Polygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid, explain_validity
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from skimage.io import imsave
from ultralytics import YOLO
from pyproj import Transformer
import pandas as pd
from pathlib import Path
from polygon_centerline_2 import polygon_centerline_polynomial_only
import xml.etree.ElementTree as ET
import math
import logging
from generate_xml import generate_map_its_xml
from flask import Flask, request, send_file, render_template_string, redirect, url_for
import tempfile
import shutil

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

# Logging einrichten
logging.basicConfig(
    filename=os.path.join(output_dir_final, 'invalid_polygons.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def chaikin_smooth(coords, iterations=0):  # Keine Glättung
    """Glättet eine Linie mit Chaikins Algorithmus für weichere Rundungen."""
    points = np.array(coords)
    if iterations == 0:
        return points.tolist()
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
            return None, None, None, None, None, None
        
        points = np.array(points)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        scale = min(target_size[0] / (max_x - min_x + 1e-6), target_size[1] / (max_y - min_y + 1e-6))
        
        img = Image.new("RGB", target_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        for geom in gdf.geometry:
            if geom.geom_type == "LineString":
                scaled_points = [((p[0] - min_x) * scale, target_size[1] - (p[1] - min_y) * scale) for p in geom.coords]
                draw.line(scaled_points, fill=(0, 0, 0), width=3)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    scaled_points = [((p[0] - min_x) * scale, target_size[1] - (p[1] - min_y) * scale) for p in line.coords]
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
        logging.info(f"Gerendert mit Geokoordinaten: {output_file}")
        return points, min_x, min_y, scale, lower_corner_wgs84, upper_corner_wgs84
    
    except Exception as e:
        print(f"❌ Fehler bei {gml_file}: {str(e)}")
        logging.info(f"Fehler bei {gml_file}: {str(e)}")
        return None, None, None, None, None, None

def apply_yolo_model(img_path, model_path="best.pt"):
    """YOLO-Modell auf Bild anwenden, Masken zurückgeben (ohne Speichern)."""
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
            masks.append(mask_resized)
        
        return masks, h_orig, w_orig
    
    except Exception as e:
        print(f"❌ Fehler bei YOLO-Inferenz: {str(e)}")
        logging.info(f"Fehler bei YOLO-Inferenz für {img_path}: {str(e)}")
        return None, None, None

def vectorize_masks(masks, img_path, iterations=0, tolerance=2.0):
    """Pixelmasken in Polygone umwandeln, ungültige Konturen sammeln."""
    vectorized_data = []
    invalid_contours = []
    
    for i, mask in enumerate(masks):
        mask_bin = (mask > 127).astype(np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ Keine Konturen in Maske {i} gefunden.")
            logging.info(f"Keine Konturen in Maske {i} für {img_path}")
            continue
        
        polygons = []
        for j, cnt in enumerate(contours):
            if len(cnt) < 3:
                print(f"⚠️ Kontur {j} in Maske {i} hat zu wenige Punkte: {len(cnt)}")
                logging.info(f"Kontur {j} in Maske {i} für {img_path} hat zu wenige Punkte: {len(cnt)}")
                continue
            points = cnt.squeeze()
            if points.ndim == 1:
                print(f"⚠️ Kontur {j} in Maske {i} ist ungültig (ndim=1).")
                logging.info(f"Kontur {j} in Maske {i} für {img_path} ist ungültig (ndim=1)")
                continue
            try:
                if not np.array_equal(points[0], points[-1]):
                    points = np.vstack([points, points[0]])
                print(f"ℹ️ Kontur {j} in Maske {i} hat {len(points)} Punkte: {points.tolist()[:5]}...")
                logging.info(f"Kontur {j} in Maske {i} für {img_path} hat {len(points)} Punkte: {points.tolist()[:5]}...")
                
                rounded_coords = chaikin_smooth(points, iterations=iterations)
                polygon = Polygon(rounded_coords)
                if not polygon.is_valid:
                    invalid_reason = explain_validity(polygon)
                    print(f"⚠️ Polygon {j} in Maske {i} ungültig: {invalid_reason}")
                    logging.info(f"Polygon {j} in Maske {i} für {img_path} ungültig: {invalid_reason}")
                    invalid_contours.append((i, cnt))
                    polygon = make_valid(polygon)
                    if not polygon.is_valid:
                        print(f"⚠️ Polygon {j} in Maske {i} konnte nicht repariert werden.")
                        logging.info(f"Polygon {j} in Maske {i} für {img_path} konnte nicht repariert werden")
                        continue
                    if polygon.geom_type != "Polygon":
                        print(f"⚠️ Repariertes Objekt {j} in Maske {i} ist kein Polygon, sondern {polygon.geom_type}.")
                        logging.info(f"Repariertes Objekt {j} in Maske {i} für {img_path} ist kein Polygon, sondern {polygon.geom_type}")
                        if isinstance(polygon, GeometryCollection):
                            print(f"ℹ️ GeometryCollection enthält: {[geom.geom_type for geom in polygon.geoms]}")
                            logging.info(f"GeometryCollection für Polygon {j} in Maske {i} enthält: {[geom.geom_type for geom in polygon.geoms]}")
                            valid_polygons = [geom for geom in polygon.geoms if geom.geom_type == "Polygon" and not geom.is_empty]
                            if valid_polygons:
                                polygon = max(valid_polygons, key=lambda x: x.area)
                                print(f"ℹ️ Größtes Polygon aus GeometryCollection ausgewählt mit Fläche: {polygon.area}")
                                logging.info(f"Größtes Polygon aus GeometryCollection für {img_path} ausgewählt mit Fläche: {polygon.area}")
                            else:
                                coords = []
                                for geom in polygon.geoms:
                                    if geom.geom_type == "LineString":
                                        coords.extend(geom.coords[:-1])
                                if len(coords) >= 3:
                                    if not np.array_equal(coords[0], coords[-1]):
                                        coords.append(coords[0])
                                    polygon = Polygon(coords)
                                    if not polygon.is_valid:
                                        polygon = make_valid(polygon)
                                        if not polygon.is_valid or polygon.geom_type != "Polygon":
                                            print(f"⚠️ Hard-Cast für Polygon {j} in Maske {i} fehlgeschlagen: {polygon.geom_type}")
                                            logging.info(f"Hard-Cast für Polygon {j} in Maske {i} für {img_path} fehlgeschlagen: {polygon.geom_type}")
                                            continue
                                else:
                                    print(f"⚠️ Zu wenige Punkte für Hard-Cast in Maske {i}, Kontur {j}: {len(coords)}")
                                    logging.info(f"Zu wenige Punkte für Hard-Cast in Maske {i}, Kontur {j} für {img_path}: {len(coords)}")
                                    continue
                smoothed_polygon = polygon if i == 7 else polygon.simplify(tolerance, preserve_topology=True)
                if not smoothed_polygon.is_valid:
                    invalid_reason = explain_validity(smoothed_polygon)
                    print(f"⚠️ Vereinfachtes Polygon {j} in Maske {i} ungültig: {invalid_reason}")
                    logging.info(f"Vereinfachtes Polygon {j} in Maske {i} für {img_path} ungültig: {invalid_reason}")
                    invalid_contours.append((i, cnt))
                    continue
                if smoothed_polygon.geom_type != "Polygon":
                    print(f"⚠️ Vereinfachtes Objekt {j} in Maske {i} ist kein Polygon, sondern {smoothed_polygon.geom_type}.")
                    logging.info(f"Vereinfachtes Objekt {j} in Maske {i} für {img_path} ist kein Polygon, sondern {smoothed_polygon.geom_type}")
                    continue
                polygons.append(smoothed_polygon)
            except Exception as e:
                print(f"⚠️ Fehler bei Kontur {j} in Maske {i}: {str(e)}")
                logging.info(f"Fehler bei Kontur {j} in Maske {i} für {img_path}: {str(e)}")
                continue
        
        if polygons:
            union_polygon = unary_union(polygons)
            if not union_polygon.is_valid:
                invalid_reason = explain_validity(union_polygon)
                print(f"⚠️ Vereintes Polygon in Maske {i} ungültig: {invalid_reason}")
                logging.info(f"Vereintes Polygon in Maske {i} für {img_path} ungültig: {invalid_reason}")
                union_polygon = make_valid(union_polygon)
            if union_polygon.is_valid and union_polygon.geom_type in ["Polygon", "MultiPolygon"]:
                vectorized_data.append(union_polygon)
            else:
                print(f"⚠️ Vereintes Polygon in Maske {i} konnte nicht repariert werden oder ist kein Polygon: {union_polygon.geom_type}")
                logging.info(f"Vereintes Polygon in Maske {i} für {img_path} konnte nicht repariert werden oder ist kein Polygon: {union_polygon.geom_type}")
    
    return vectorized_data, invalid_contours

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
            print(f"ℹ️ Verarbeite Polygon {i} mit Geometrietyp: {polygon.geom_type}")
            logging.info(f"Verarbeite Polygon {i} mit Geometrietyp: {polygon.geom_type} für {img_path}")
            if polygon.geom_type == "Polygon":
                x, y = polygon.exterior.xy
                points = np.array(list(zip(x, y)))
                points = [(int(x), int(y)) for x, y in points if 0 <= x < w and 0 <= y < h]
                if len(points) >= 3:
                    cv2.polylines(overlay, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=1)
                    cv2.fillPoly(overlay, [np.array(points)], color=(200, 200, 255))
                
                yx = np.array(list(zip(x, y)))
                if len(yx) == 0:
                    print(f"⚠️ Keine gültigen Polygonpunkte für Polygon {i}. Überspringe.")
                    logging.info(f"Keine gültigen Polygonpunkte für Polygon {i} für {img_path}")
                    continue
                original_points = [(x / scale + min_x, (target_size[1] - y) / scale + min_y) for x, y in yx]
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
                        cv2.fillPoly(overlay, [np.array(points)], color=(200, 200, 255))
                    
                    yx = np.array(list(zip(x, y)))
                    if len(yx) == 0:
                        print(f"⚠️ Keine gültigen Polygonpunkte für Polygon {i}.{j}. Überspringe.")
                        logging.info(f"Keine gültigen Polygonpunkte für Polygon {i}.{j} für {img_path}")
                        continue
                    original_points = [(x / scale + min_x, (target_size[1] - y) / scale + min_y) for x, y in yx]
                    geo_points = [transformer.transform(x, y) for x, y in original_points]
                    vectors.append({
                        "polygon_id": f"{i}.{j}",
                        "image_coordinates": yx.tolist(),
                        "original_coordinates": original_points,
                        "geocoordinates": geo_points
                    })
            else:
                print(f"⚠️ Polygon {i} hat ungültigen Geometrietyp: {polygon.geom_type}. Überspringe.")
                logging.info(f"Polygon {i} hat ungültigen Geometrietyp: {polygon.geom_type} für {img_path}")
                continue
        
        cv2.imwrite(output_path, overlay)
        print(f"✅ Überlagertes Bild gespeichert: {output_path}")
        logging.info(f"Überlagertes Bild gespeichert: {output_path}")
        
        # CSV zusätzlich speichern
        vector_path = os.path.join(output_dir_vectors, os.path.basename(img_path).replace(".png", "_vectors.csv"))
        df_vectors = pd.DataFrame(vectors)
        df_vectors.to_csv(vector_path, index=False)
        print(f"✅ Vektoren gespeichert: {vector_path}")
        logging.info(f"Vektoren gespeichert: {vector_path}")
        
        return vectors
    
    except Exception as e:
        print(f"❌ Fehler beim Plotten: {str(e)}")
        logging.info(f"Fehler beim Plotten für {img_path}: {str(e)}")
        return None

def get_distinct_color(i, total=30):
    """Erzeugt deutlich unterscheidbare RGB-Farben auf Basis von HSL."""
    import colorsys
    hue = (i * 360 / total) % 360
    saturation = 0.6 + 0.4 * ((i % 5) / 4)
    lightness = 0.4 + 0.3 * ((i % 3) / 2)
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    return (int(r * 255), int(g * 255), int(b * 255))

def print_centerline_geocoords(centerline: LineString, min_x: float, min_y: float, scale: float, num_points: int = 10):
    if centerline.length == 0 or len(centerline.coords) < 2:
        print("❌ Mittellinie ist leer oder zu kurz.")
        logging.info("Mittellinie ist leer oder zu kurz.")
        return

    distances = np.linspace(0, centerline.length, num_points)
    sampled_points = [centerline.interpolate(d) for d in distances]
    print("🧭 Geokoordinaten entlang der Mittellinie:")
    logging.info("Geokoordinaten entlang der Mittellinie:")
    for i, pt in enumerate(sampled_points):
        x_img, y_img = pt.x, pt.y
        x_orig = x_img / scale + min_x
        y_orig = (target_size[1] - y_img) / scale + min_y
        lon, lat = transformer.transform(x_orig, y_orig)
        print(f"  Punkt {i+1}: ({lat:.6f}, {lon:.6f})")
        logging.info(f"  Punkt {i+1}: ({lat:.6f}, {lon:.6f})")

def process_single_gml(gml_path, gml_file_name):
    print(f"📂 Verarbeite GML-Datei: {gml_file_name}")
    logging.info(f"Verarbeite GML-Datei: {gml_file_name}")

    # Render GML to image
    img_path = os.path.join(output_dir_images, gml_file_name.replace(".gml", ".png"))
    points, min_x, min_y, scale, lower_corner_wgs84, upper_corner_wgs84 = render_gml_to_image(gml_path, img_path)
    if points is None:
        print(f"❌ Keine gültigen Linien in {gml_file_name}. Überspringe.")
        logging.info(f"Keine gültigen Linien in {gml_file_name}. Überspringe.")
        return None

    # Apply YOLO model to get masks
    masks, h_orig, w_orig = apply_yolo_model(img_path)
    if masks is None or not masks:
        print(f"❌ Keine gültigen Masken für {gml_file_name}. Überspringe.")
        logging.info(f"Keine gültigen Masken für {gml_file_name}. Überspringe.")
        return None

    # Debugging: Save individual masks to inspect them
    for i, mask in enumerate(masks):
        mask_path = os.path.join(output_dir_masks, f"mask_{i}_{os.path.basename(img_path)}")
        cv2.imwrite(mask_path, mask)
        print(f"✅ Maske {i} gespeichert: {mask_path} (Max-Wert: {mask.max()})")
        logging.info(f"Maske {i} gespeichert: {mask_path} (Max-Wert: {mask.max()})")

    # Plot combined masks on original image
    original = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if original is None:
        print(f"❌ Konnte Originalbild nicht laden: {img_path}")
        logging.info(f"Konnte Originalbild nicht laden: {img_path}")
        return None
    combined_mask = original.copy()
    for i, mask in enumerate(masks):
        if mask.max() == 0:
            print(f"⚠️ Maske {i} für {gml_file_name} ist leer (Max-Wert: {mask.max()}).")
            logging.info(f"Maske {i} für {gml_file_name} ist leer (Max-Wert: {mask.max()}).")
            continue
        color = get_distinct_color(i)
        mask_colored = np.zeros_like(combined_mask)
        mask_colored[mask == 255] = color
        combined_mask = cv2.addWeighted(combined_mask, 0.7, mask_colored, 0.3, 0)
    combined_mask_path = os.path.join(output_dir_final, f"masks_combined_{os.path.basename(img_path)}")
    cv2.imwrite(combined_mask_path, combined_mask)
    print(f"✅ Kombinierte Pixelmasken gespeichert: {combined_mask_path}")
    logging.info(f"Kombinierte Pixelmasken gespeichert: {combined_mask_path}")

    # Plot raw contours before validation
    contour_overlay = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if contour_overlay is None:
        print(f"❌ Konnte Originalbild nicht laden: {img_path}")
        logging.info(f"Konnte Originalbild nicht laden: {img_path}")
        return None
    for i, mask in enumerate(masks):
        mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ Keine Konturen in Maske {i} für {gml_file_name} gefunden.")
            logging.info(f"Keine Konturen in Maske {i} für {gml_file_name} gefunden.")
            continue
        color = get_distinct_color(i)
        cv2.drawContours(contour_overlay, contours, -1, color, thickness=2)
    contour_overlay_path = os.path.join(output_dir_final, f"raw_contours_{os.path.basename(img_path)}")
    cv2.imwrite(contour_overlay_path, contour_overlay)
    print(f"✅ Rohe Konturen gespeichert: {contour_overlay_path}")
    logging.info(f"Rohe Konturen gespeichert: {contour_overlay_path}")

    # Vectorize masks and collect invalid contours
    vectorized_data, invalid_contours = vectorize_masks(masks, img_path, iterations=0, tolerance=2.0)
    if not vectorized_data:
        print(f"⚠️ Keine gültigen Polygone für {gml_file_name}. Überspringe Polygon-Plotting.")
        logging.info(f"Keine gültigen Polygone für {gml_file_name}. Überspringe Polygon-Plotting.")
        if invalid_contours:
            invalid_contour_overlay = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if invalid_contour_overlay is None:
                print(f"❌ Konnte Originalbild nicht laden: {img_path}")
                logging.info(f"Konnte Originalbild nicht laden: {img_path}")
                return None
            for mask_idx, contour in invalid_contours:
                cv2.drawContours(invalid_contour_overlay, [contour], -1, (255, 0, 0), thickness=2)
            invalid_contour_path = os.path.join(output_dir_final, f"invalid_contours_{os.path.basename(img_path)}")
            cv2.imwrite(invalid_contour_path, invalid_contour_overlay)
            print(f"✅ Ungültige Konturen gespeichert: {invalid_contour_path}")
            logging.info(f"Ungültige Konturen gespeichert: {invalid_contour_path}")
        return None

    # Plot validated polygons
    polygon_overlay = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if polygon_overlay is None:
        print(f"❌ Konnte Originalbild nicht laden: {img_path}")
        logging.info(f"Konnte Originalbild nicht laden: {img_path}")
        return None

    polygon_records = []
    for i, polygon in enumerate(vectorized_data):
        if polygon.is_empty:
            print(f"⚠️ Polygon {i} ist leer.")
            logging.info(f"Polygon {i} ist leer für {gml_file_name}")
            continue
        polygons_to_draw = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
        for j, poly in enumerate(polygons_to_draw):
            if poly.geom_type != "Polygon":
                print(f"⚠️ Objekt {i}.{j} ist kein Polygon, sondern {poly.geom_type}. Überspringe.")
                logging.info(f"Objekt {i}.{j} ist kein Polygon, sondern {poly.geom_type} für {gml_file_name}")
                continue
            x, y = poly.exterior.xy
            pts = [(int(px), int(py)) for px, py in zip(x, y) if 0 <= px < target_size[0] and 0 <= py < target_size[1]]
            if len(pts) < 3:
                print(f"⚠️ Polygon {i}.{j} hat zu wenige gültige Punkte: {len(pts)}")
                logging.info(f"Polygon {i}.{j} hat zu wenige gültige Punkte: {len(pts)} für {gml_file_name}")
                continue
            pts_np = np.array(pts, np.int32)
            cv2.polylines(polygon_overlay, [pts_np], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.fillPoly(polygon_overlay, [pts_np], color=(200, 200, 255))

            points_str = " ".join([f"{pt[0]},{pt[1]}" for pt in pts])
            polygon_id = f"{i}" if len(polygons_to_draw) == 1 else f"{i}.{j}"
            polygon_records.append({
                "polygon_id": polygon_id,
                "points": points_str
            })

    polygon_overlay_path = os.path.join(output_dir_final, f"polygons_overlay_{os.path.basename(img_path)}")
    cv2.imwrite(polygon_overlay_path, polygon_overlay)
    print(f"✅ Polygonüberlagerung gespeichert: {polygon_overlay_path}")
    logging.info(f"Polygonüberlagerung gespeichert: {polygon_overlay_path}")

    polygon_csv_path = os.path.join(output_dir_final, f"polygons_{os.path.basename(img_path).replace('.png', '.csv')}")
    df_polygons = pd.DataFrame(polygon_records)
    df_polygons.to_csv(polygon_csv_path, index=False)
    print(f"✅ Polygon-CSV gespeichert: {polygon_csv_path}")
    logging.info(f"Polygon-CSV gespeichert: {polygon_csv_path}")

    # Calculate centerlines and generate XML
    centerline_coords = []
    overlay = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if overlay is None:
        print(f"❌ Konnte Originalbild nicht laden: {img_path}")
        logging.info(f"Konnte Originalbild nicht laden: {img_path}")
        return None
    for i, polygon in enumerate(vectorized_data):
        if polygon.geom_type != "Polygon":
            print(f"⚠️ Polygon {i} ist kein einfacher Polygon (Typ: {polygon.geom_type}). Überspringe.")
            logging.info(f"Polygon {i} ist kein einfacher Polygon (Typ: {polygon.geom_type}) für {gml_file_name}")
            continue
        try:
            print(f"➕ Berechne Mittellinie für Polygon {i}")
            logging.info(f"Berechne Mittellinie für Polygon {i} für {gml_file_name}")
            centerline, avg_width, avg_xy, poly_centerline_coords = polygon_centerline_polynomial_only(
                polygon,
                dx=5.0,
                degrees=[3],
                smooth_window=51,
                smooth_order=3,
                min_x=min_x,
                min_y=min_y,
                scale=scale,
                lower_corner_wgs84=lower_corner_wgs84,  # Hinzufügen
                upper_corner_wgs84=upper_corner_wgs84,  # Hinzufügen
                show_plots=False
            )
            if not isinstance(centerline, LineString):
                print(f"⚠️ Keine gültige Mittellinie für Polygon {i}.")
                logging.info(f"Keine gültige Mittellinie für Polygon {i} für {gml_file_name}")
                continue
            print(f"ℹ️ Breite: {avg_width:.2f} cm | Abstand AVG(x+y): {avg_xy:.2f}")
            logging.info(f"Breite: {avg_width:.2f} cm | Abstand AVG(x+y): {avg_xy:.2f} für Polygon {i} in {gml_file_name}")

            x_line, y_line = centerline.xy
            for j in range(len(x_line) - 1):
                pt1 = (int(x_line[j]), int(y_line[j]))
                pt2 = (int(x_line[j + 1]), int(y_line[j + 1]))
                cv2.line(overlay, pt1, pt2, color=(0, 255, 0), thickness=2)

            for coord in poly_centerline_coords:
                coord["polygon_id"] = i
                centerline_coords.append(coord)

        except Exception as e:
            print(f"⚠️ Fehler bei Polygon {i}: {e}")
            logging.info(f"Fehler bei Polygon {i} für {gml_file_name}: {str(e)}")
            continue

    centerlines_path = os.path.join(output_dir_final, f"centerlines_{os.path.basename(img_path)}")
    cv2.imwrite(centerlines_path, overlay)
    print(f"✅ Bild mit Mittellinien gespeichert: {centerlines_path}")
    logging.info(f"Bild mit Mittellinien gespeichert: {centerlines_path}")

    centerline_csv_path = os.path.join(output_dir_final, f"centerlines_{os.path.basename(img_path).replace('.png', '.csv')}")
    df_centerline = pd.DataFrame(centerline_coords)
    df_centerline.to_csv(centerline_csv_path, index=False)
    print(f"✅ GPS-Koordinaten der Mittellinien gespeichert: {centerline_csv_path}")
    logging.info(f"GPS-Koordinaten der Mittellinien gespeichert: {centerline_csv_path}")

    # Generate MAP ITS XML
    xml_path = os.path.join(output_dir_final, f"map_its_{os.path.basename(img_path).replace('.png', '.xml')}")
    generate_map_its_xml(centerline_coords, xml_path, lower_corner_wgs84)

    print(f"🎉 Verarbeitung von {gml_file_name} abgeschlossen.")
    logging.info(f"Verarbeitung von {gml_file_name} abgeschlossen.")

    return xml_path

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GML Processing App</title>
</head>
<body>
    <h1>Upload GML File</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
        <input type="file" name="file" accept=".gml">
        <input type="submit" value="Upload and Process">
    </form>
    {% if xml_path %}
    <h2>Processing Complete!</h2>
    <p>Download the generated XML: <a href="/download/{{ xml_filename }}">Download XML</a></p>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, xml_path=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and file.filename.endswith('.gml'):
        # Create a temporary directory for this upload
        temp_dir = tempfile.mkdtemp()
        gml_path = os.path.join(temp_dir, file.filename)
        file.save(gml_path)
        
        # Process the file
        xml_path = process_single_gml(gml_path, file.filename)
        
        # Clean up temp dir after processing
        shutil.rmtree(temp_dir)
        
        if xml_path:
            xml_filename = os.path.basename(xml_path)
            return render_template_string(HTML_TEMPLATE, xml_path=xml_path, xml_filename=xml_filename)
        else:
            return "Processing failed."
    return "Invalid file type. Please upload a .gml file."

@app.route('/download/<xml_filename>', methods=['GET'])
def download_xml(xml_filename):
    xml_path = os.path.join(output_dir_final, xml_filename)
    if os.path.exists(xml_path):
        return send_file(xml_path, as_attachment=True)
    return "File not found."

if __name__ == "__main__":
    app.run(debug=True)