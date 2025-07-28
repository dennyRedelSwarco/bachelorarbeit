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
import xml.etree.ElementTree as ET
import math

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
                # Korrigiere Y-Achse für PNG-Rendering
                scaled_points = [((p[0] - min_x) * scale, target_size[1] - (p[1] - min_y) * scale) for p in geom.coords]
                draw.line(scaled_points, fill=(0, 0, 0), width=3)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    # Korrigiere Y-Achse für PNG-Rendering
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
        return points, min_x, min_y, scale
    
    except Exception as e:
        print(f"❌ Fehler bei {gml_file}: {str(e)}")
        return None, None, None, None

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
        return None, None, None
    
def vectorize_masks(masks, img_path, iterations=2, tolerance=3.0):
    """Pixelmasken in Polygone umwandeln, keine Bilddateien speichern."""
    vectorized_data = []
    
    for i, mask in enumerate(masks):
        mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Keine Konturen in Maske {i} gefunden. Überspringe.")
            continue
        
        polygons = []
        for cnt in contours:
            if len(cnt) >= 3:
                points = cnt.squeeze()
                if points.ndim == 1:
                    continue
                rounded_coords = chaikin_smooth(points, iterations=iterations)
                try:
                    polygon = Polygon(rounded_coords)
                    if not polygon.is_valid:
                        continue
                    smoothed_polygon = polygon.simplify(tolerance, preserve_topology=True)
                    if not smoothed_polygon.is_valid:
                        continue
                    polygons.append(smoothed_polygon)
                except Exception as e:
                    print(f"Fehler bei Kontur in Maske {i}: {str(e)}")
                    continue
        
        if not polygons:
            continue
        
        union_polygon = unary_union(polygons)
        if not union_polygon.is_valid:
            continue
        
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
            if polygon.geom_type == "Polygon":
                x, y = polygon.exterior.xy
                points = np.array(list(zip(x, y)))
                points = [(int(x), int(y)) for x, y in points if 0 <= x < w and 0 <= y < h]
                if len(points) >= 3:
                    cv2.polylines(overlay, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=1)
                
                yx = np.array(list(zip(x, y)))
                if len(yx) == 0:
                    print(f"Keine gültigen Polygonpunkte für Polygon {i}. Überspringe.")
                    continue
                # Korrigiere Y-Achse für Rücktransformation
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
                    
                    yx = np.array(list(zip(x, y)))
                    if len(yx) == 0:
                        print(f"Keine gültigen Polygonpunkte für Polygon {i}.{j}. Überspringe.")
                        continue
                    # Korrigiere Y-Achse für Rücktransformation
                    original_points = [(x / scale + min_x, (target_size[1] - y) / scale + min_y) for x, y in yx]
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


def generate_map_its_xml(centerline_coords, output_path):
    """Generiert eine Utopia-konforme MAP ITS XML-Datei basierend auf centerline_coords."""
    try:
        # Wurzelelement
        map_data_elem = ET.Element("MAP_Data")
        
        # Map-Element mit Metadaten
        map_elem = ET.SubElement(map_data_elem, "Map")
        map_elem.set("Version", "2")
        map_elem.set("IntersectionId", "3")
        map_elem.set("MapID", "19531")  # Beispielwert, anpassen falls nötig
        
        # GenericIntersection
        intersection_elem = ET.SubElement(map_elem, "GenericIntersection")
        ET.SubElement(intersection_elem, "Name").text = "MAP_ITS_17_1729_2.1"  # Beispielname
        ET.SubElement(intersection_elem, "ID").text = "1"
        
        # Referenzpunkt
        if not centerline_coords:
            print("❌ Keine Mittelliniendaten vorhanden. XML-Generierung abgebrochen.")
            return
        
        first_coord = centerline_coords[0]["geocoordinate"]
        lon, lat = map(float, first_coord.strip("()").split(","))
        ref_point = ET.SubElement(intersection_elem, "RefPoint")
        ET.SubElement(ref_point, "lat").text = str(int(lat * 10000000))  # 1/10 Mikrograd
        ET.SubElement(ref_point, "long").text = str(int(lon * 10000000))
        
        # Projektive Transformation für präzise Delta-Werte
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        ref_x, ref_y = transformer.transform(lon, lat)  # Referenzpunkt in UTM
        
        # LaneSet
        lane_set_elem = ET.SubElement(intersection_elem, "LaneSet")
        
        # Gruppieren der Koordinaten nach polygon_id
        lane_groups = {}
        for coord in centerline_coords:
            polygon_id = coord["polygon_id"]
            if polygon_id not in lane_groups:
                lane_groups[polygon_id] = []
            lane_groups[polygon_id].append(coord)
        
        for lane_id, (polygon_id, coords) in enumerate(lane_groups.items(), 1):
            lane_elem = ET.SubElement(lane_set_elem, "GenericLane")
            ET.SubElement(lane_elem, "LaneID").text = str(lane_id)
            
            # LaneAttributes
            lane_attrs_elem = ET.SubElement(lane_elem, "LaneAttributes")
            directional_use = ET.SubElement(lane_attrs_elem, "directionalUse")
            directional_use.text = "01"  # Einfahrspur
            shared_with = ET.SubElement(lane_attrs_elem, "sharedWith")
            shared_with.text = "00000000"  # Beispielwert
            lane_type = ET.SubElement(lane_attrs_elem, "laneType")
            lane_type_elem = ET.SubElement(lane_type, "vehicle")
            lane_type_elem.text = "0001000000"  # Fahrzeugspur
            
            # NodeList
            node_list_elem = ET.SubElement(lane_elem, "NodeList")
            nodes_elem = ET.SubElement(node_list_elem, "Nodes")
            
            for coord in coords:
                lon_p, lat_p = map(float, coord["geocoordinate"].strip("()").split(","))
                x_p, y_p = transformer.transform(lon_p, lat_p)  # Punkt in UTM
                x_cm = int((x_p - ref_x) * 100)  # Delta in cm
                y_cm = int((y_p - ref_y) * 100)  # Delta in cm
                
                node_elem = ET.SubElement(nodes_elem, "Node")
                node_xy_elem = ET.SubElement(node_elem, "Node-XY")
                delta_elem = ET.SubElement(node_xy_elem, "delta")
                ET.SubElement(delta_elem, "x").text = str(x_cm)
                ET.SubElement(delta_elem, "y").text = str(y_cm)
            
            # ConnectsTo (Beispiel, anpassen falls Verbindungen bekannt)
            connects_to_elem = ET.SubElement(lane_elem, "ConnectsTo")
            connecting_lane_elem = ET.SubElement(connects_to_elem, "ConnectingLane")
            ET.SubElement(connecting_lane_elem, "lane").text = str(lane_id + 1 if lane_id < len(lane_groups) else 1)
            ET.SubElement(connecting_lane_elem, "maneuver").text = "100000000000"  # Geradeausfahren
            
            # Attributes (z. B. Spurbreite)
            attrs_elem = ET.SubElement(lane_elem, "Attributes")
            ET.SubElement(attrs_elem, "laneWidth").text = "350"  # Beispiel: 3,5 m
            
            # Maneuvers (Beispiel)
            maneuvers_elem = ET.SubElement(lane_elem, "Maneuvers")
            maneuvers_elem.text = "100000000000"  # Geradeausfahren
        
        # Schöne Formatierung
        tree = ET.ElementTree(map_data_elem)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ MAP ITS XML gespeichert: {output_path}")
    
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der XML-Datei: {str(e)}")


def get_distinct_color(i, total=30):
    """Erzeugt deutlich unterscheidbare RGB-Farben auf Basis von HSL."""
    import colorsys
    hue = (i * 360 / total) % 360  # gleichmäßig auf dem Farbkreis verteilt
    saturation = 0.6 + 0.4 * ((i % 5) / 4)  # variiert zwischen 0.6 und 1.0
    lightness = 0.4 + 0.3 * ((i % 3) / 2)  # variiert zwischen 0.4 und 0.7
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    return (int(r * 255), int(g * 255), int(b * 255))



def main():
    gml_files = [f for f in os.listdir(input_dir_gml) if f.lower().endswith(".gml")]
    if not gml_files:
        print("❌ Keine GML-Dateien in gml_data gefunden.")
        return

    for gml_file in gml_files:
        print(f"📂 Verarbeite GML-Datei: {gml_file}")
        gml_path = os.path.join(input_dir_gml, gml_file)

        # Render GML to image
        img_path = os.path.join(output_dir_images, gml_file.replace(".gml", ".png"))
        points, min_x, min_y, scale = render_gml_to_image(gml_path, img_path)
        if points is None:
            print(f"❌ Keine gültigen Linien in {gml_file}. Überspringe.")
            continue

        # Apply YOLO model to get masks
        masks, h_orig, w_orig = apply_yolo_model(img_path)
        if masks is None or not masks:
            print(f"❌ Keine gültigen Masken für {gml_file}. Überspringe.")
            continue

        # Plot combined masks on original image
        original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if original is None:
            print(f"❌ Konnte Originalbild nicht laden: {img_path}")
            continue
        combined_mask = original.copy()
        for i, mask in enumerate(masks):
            if mask.max() == 0:
                print(f"⚠️ Maske {i} für {gml_file} ist leer (max Wert: {mask.max()}).")
                continue
            color = get_distinct_color(i)
            mask_colored = np.zeros_like(combined_mask)
            mask_colored[mask == 255] = color
            combined_mask[mask == 255] = cv2.addWeighted(mask_colored, 0.5, combined_mask, 0.5, 0)[mask == 255]
        combined_mask_path = os.path.join(output_dir_final, f"masks_combined_{os.path.basename(img_path)}")
        cv2.imwrite(combined_mask_path, combined_mask)
        print(f"✅ Kombinierte Pixelmasken gespeichert: {combined_mask_path}")

        # Vectorize masks
        vectorized_data = vectorize_masks(masks, img_path, iterations=10, tolerance=10.0)
        if not vectorized_data:
            print(f"⚠️ Keine gültigen Polygone für {gml_file}. Überspringe Polygon-Plotting.")
            continue

        # Plot polygons on original image
        polygon_overlay = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if polygon_overlay is None:
            print(f"❌ Konnte Originalbild nicht laden: {img_path}")
            continue

        for i, polygon in enumerate(vectorized_data):
            if polygon.is_empty:
                print(f"⚠️ Polygon {i} ist leer.")
                continue

            polygons_to_draw = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
            for poly in polygons_to_draw:
                x, y = poly.exterior.xy
                pts = []
                for px, py in zip(x, y):
                    x_img = int(px)  # Bereits in Bildkoordinaten
                    y_img = int(py)
                    if 0 <= x_img < target_size[0] and 0 <= y_img < target_size[1]:
                        pts.append((x_img, y_img))
                if len(pts) >= 3:
                    pts_np = np.array(pts, np.int32)
                    cv2.polylines(polygon_overlay, [pts_np], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.fillPoly(polygon_overlay, [pts_np], color=(200, 200, 255, 50))  # Halbtransparente Füllung

        polygon_overlay_path = os.path.join(output_dir_final, f"polygons_overlay_{os.path.basename(img_path)}")
        cv2.imwrite(polygon_overlay_path, polygon_overlay)
        print(f"✅ Polygonüberlagerung gespeichert: {polygon_overlay_path}")

        # Calculate centerlines and generate XML
        centerline_coords = []
        overlay = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if overlay is None:
            print(f"❌ Konnte Originalbild nicht laden: {img_path}")
            continue

        for i, polygon in enumerate(vectorized_data):
            if polygon.geom_type != "Polygon":
                print(f"⚠️ Polygon {i} ist kein einfacher Polygon (Typ: {polygon.geom_type}). Überspringe.")
                continue
            try:
                print(f"➕ Berechne Mittellinie für Polygon {i}")
                centerline, avg_width, avg_xy = polygon_centerline(
                    polygon, dx=5.0, smooth_window=11, smooth_order=1, show_plots=False
                )
                if not isinstance(centerline, LineString):
                    print(f"⚠️ Keine gültige Mittellinie für Polygon {i}.")
                    continue
                print(f"ℹ️ Breite: {avg_width:.2f} | Abstand AVG(x+y): {avg_xy:.2f}")
                print_centerline_geocoords(centerline, min_x, min_y, scale)

                x_line, y_line = centerline.xy
                for j in range(len(x_line) - 1):
                    pt1 = (int(x_line[j]), int(y_line[j]))
                    pt2 = (int(x_line[j + 1]), int(y_line[j + 1]))
                    cv2.line(overlay, pt1, pt2, color=(0, 255, 0), thickness=2)

                distances = np.linspace(0, centerline.length, 10)
                sampled_points = [centerline.interpolate(d) for d in distances]
                for k, pt in enumerate(sampled_points):
                    x_img, y_img = pt.x, pt.y
                    x_orig = x_img / scale + min_x
                    y_orig = (target_size[1] - y_img) / scale + min_y
                    lon, lat = transformer.transform(x_orig, y_orig)
                    centerline_coords.append({
                        "polygon_id": i,
                        "point_index": k,
                        "geocoordinate": f"({lon:.6f}, {lat:.6f})"
                    })
            except Exception as e:
                print(f"⚠️ Fehler bei Polygon {i}: {e}")
                continue

        centerlines_path = os.path.join(output_dir_final, f"centerlines_{os.path.basename(img_path)}")
        cv2.imwrite(centerlines_path, overlay)
        print(f"✅ Bild mit Mittellinien gespeichert: {centerlines_path}")

        centerline_csv_path = os.path.join(output_dir_final, f"centerlines_{os.path.basename(img_path).replace('.png', '.csv')}")
        df_centerline = pd.DataFrame(centerline_coords)
        df_centerline.to_csv(centerline_csv_path, index=False)
        print(f"✅ GPS-Koordinaten der Mittellinien gespeichert: {centerline_csv_path}")

        # Generate MAP ITS XML
        xml_path = os.path.join(output_dir_final, f"map_its_{os.path.basename(img_path).replace('.png', '.xml')}")
        generate_map_its_xml(centerline_coords, xml_path)

    print("🎉 Alle GML-Dateien erfolgreich verarbeitet.")

def print_centerline_geocoords(centerline: LineString, min_x: float, min_y: float, scale: float, num_points: int = 10):
    if centerline.length == 0 or len(centerline.coords) < 2:
        print("❌ Mittellinie ist leer oder zu kurz.")
        return

    distances = np.linspace(0, centerline.length, num_points)
    sampled_points = [centerline.interpolate(d) for d in distances]
    print("🧭 Geokoordinaten entlang der Mittellinie:")
    for i, pt in enumerate(sampled_points):
        x_img, y_img = pt.x, pt.y
        x_orig = x_img / scale + min_x
        # Korrigiere Y-Achse für Rücktransformation
        y_orig = (target_size[1] - y_img) / scale + min_y
        lon, lat = transformer.transform(x_orig, y_orig)
        print(f"  Punkt {i+1}: ({lat:.6f}, {lon:.6f})")

if __name__ == "__main__":
    main()