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
import colorsys

# 🎨 Bildgröße
target_size = (1024, 1024)

# 🌐 Koordinatenreferenzsysteme
source_crs = "EPSG:25832"  # Annahme: GML in UTM Zone 32N
target_crs = "EPSG:4326"   # WGS84 für Geokoordinaten
transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

def chaikin_smooth(coords, iterations=0):
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
            logging.info(f"Keine Linien in {gml_file}. Überspringe.")
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
        vector_path = os.path.join("vectors", os.path.basename(img_path).replace(".png", "_vectors.csv"))
        df_vectors = pd.DataFrame(vectors)
        df_vectors.to_csv(vector_path, index=False)
        print(f"✅ Vektoren gespeichert: {vector_path}")
        logging.info(f"Vektoren gespeichert: {vector_path}")
        
        return vectors
    
    except Exception as e:
        print(f"❌ Fehler beim Plotten: {str(e)}")
        logging.info(f"Fehler beim Plotten für {img_path}: {str(e)}")
        return None

def generate_map_its_xml(centerline_coords, output_path, min_x, min_y, scale, lower_corner_wgs84, upper_corner_wgs84):
    """Generiert eine Utopia-konforme MAPEM XML-Datei basierend auf centerline_coords."""
    try:
        mapem_elem = ET.Element("MAPEM")
        
        # Header
        header_elem = ET.SubElement(mapem_elem, "header")
        ET.SubElement(header_elem, "protocolVersion").text = "2"
        ET.SubElement(header_elem, "messageID").text = "5"
        ET.SubElement(header_elem, "stationID").text = "19531"
        
        # Map
        map_elem = ET.SubElement(mapem_elem, "map")
        ET.SubElement(map_elem, "msgIssueRevision").text = "0"
        
        # Intersections
        intersections_elem = ET.SubElement(map_elem, "intersections")
        intersection_elem = ET.SubElement(intersections_elem, "IntersectionGeometry")
        ET.SubElement(intersection_elem, "name").text = "MAP_ITS_17_1729_2.1"
        
        # ID
        id_elem = ET.SubElement(intersection_elem, "id")
        ET.SubElement(id_elem, "region").text = "3"
        ET.SubElement(id_elem, "id").text = "1729"
        
        ET.SubElement(intersection_elem, "revision").text = "1"
        
        # RefPoint
        if not centerline_coords:
            print("❌ Keine Mittelliniendaten vorhanden. XML-Generierung abgebrochen.")
            logging.info("Keine Mittelliniendaten vorhanden. XML-Generierung abgebrochen.")
            return
        
        first_coord = centerline_coords[0]["geocoordinate"]
        lon, lat = map(float, first_coord.strip("()").split(","))
        ref_point = ET.SubElement(intersection_elem, "refPoint")
        ET.SubElement(ref_point, "lat").text = str(int(lat * 10000000))
        ET.SubElement(ref_point, "long").text = str(int(lon * 10000000))
        
        # Lane Width (Standardwert oder Durchschnitt)
        avg_width = centerline_coords[0].get("avg_width", 3.25)  # Fallback auf 3.25m
        ET.SubElement(intersection_elem, "laneWidth").text = str(int(avg_width * 100))  # In cm
        
        # Speed Limits
        speed_limits_elem = ET.SubElement(intersection_elem, "speedLimits")
        speed_limit_elem = ET.SubElement(speed_limits_elem, "RegulatorySpeedLimit")
        type_elem = ET.SubElement(speed_limit_elem, "type")
        ET.SubElement(type_elem, "vehicleMaxSpeed")
        ET.SubElement(speed_limit_elem, "speed").text = "694"
        
        # Transformer für Koordinaten
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        ref_x, ref_y = transformer.transform(lon, lat)
        
        # LaneSet
        lane_set_elem = ET.SubElement(intersection_elem, "laneSet")
        
        lane_groups = {}
        for coord in centerline_coords:
            polygon_id = coord["polygon_id"]
            if polygon_id not in lane_groups:
                lane_groups[polygon_id] = []
            lane_groups[polygon_id].append(coord)
        
        for lane_id, (polygon_id, coords) in enumerate(lane_groups.items(), 10):
            lane_elem = ET.SubElement(lane_set_elem, "GenericLane")
            ET.SubElement(lane_elem, "laneID").text = str(lane_id)
            ET.SubElement(lane_elem, "ingressApproach").text = "3"
            
            # Lane Attributes
            lane_attrs_elem = ET.SubElement(lane_elem, "laneAttributes")
            ET.SubElement(lane_attrs_elem, "directionalUse").text = "10"
            ET.SubElement(lane_attrs_elem, "sharedWith").text = "0001100000"
            lane_type = ET.SubElement(lane_attrs_elem, "laneType")
            ET.SubElement(lane_type, "vehicle").text = "00000000"
            
            # NodeList
            node_list_elem = ET.SubElement(lane_elem, "nodeList")
            nodes_elem = ET.SubElement(node_list_elem, "nodes")
            
            for i, coord in enumerate(coords):
                lon_p, lat_p = map(float, coord["geocoordinate"].strip("()").split(","))
                x_p, y_p = transformer.transform(lon_p, lat_p)
                x_cm = int((x_p - ref_x) * 100)
                y_cm = int((y_p - ref_y) * 100)
                
                # Berechne relative Breite basierend auf Geokoordinaten
                avg_width = coord.get("avg_width", 3.25)  # Fallback auf 3.25m
                d_width = int(avg_width * 100)  # In cm
                
                node_elem = ET.SubElement(nodes_elem, "NodeXY")
                delta_elem = ET.SubElement(node_elem, "delta")
                node_xy_tag = f"node-XY{6 if i % 2 == 0 else 5}"
                node_xy_elem = ET.SubElement(delta_elem, node_xy_tag)
                ET.SubElement(node_xy_elem, "x").text = str(x_cm)
                ET.SubElement(node_xy_elem, "y").text = str(y_cm)
                
                # Attributes mit dWidth
                attributes_elem = ET.SubElement(node_elem, "attributes")
                ET.SubElement(attributes_elem, "dWidth").text = str(d_width)
            
            # ConnectsTo
            connects_to_elem = ET.SubElement(lane_elem, "ConnectsTo")
            connecting_lane_elem = ET.SubElement(connects_to_elem, "ConnectingLane")
            ET.SubElement(connecting_lane_elem, "lane").text = str(lane_id + 1 if lane_id < 10 + len(lane_groups) - 1 else 10)
            ET.SubElement(connecting_lane_elem, "maneuver").text = "100000000000"
        
        # XML schreiben
        tree = ET.ElementTree(mapem_elem)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ MAPEM XML gespeichert: {output_path}")
        logging.info(f"MAPEM XML gespeichert: {output_path}")
    
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der XML-Datei: {str(e)}")
        logging.info(f"Fehler beim Erstellen der XML-Datei für {output_path}: {str(e)}")

def get_distinct_color(i, total=30):
    """Erzeugt deutlich unterscheidbare RGB-Farben auf Basis von HSL."""
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