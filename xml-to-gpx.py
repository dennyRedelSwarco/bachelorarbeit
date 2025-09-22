import os
import xml.etree.ElementTree as ET
from pyproj import Transformer
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_gpx_from_mapem_xml(xml_path, output_gpx_path):
    """Generiert eine GPX-Datei basierend auf einer MAPEM XML-Datei, indem relative Koordinaten in absolute umgewandelt werden."""
    logging.debug(f"Verarbeite XML-Datei: {xml_path}")
    
    try:
        # XML parsen
        if not os.path.exists(xml_path):
            logging.error(f"XML-Datei existiert nicht: {xml_path}")
            raise FileNotFoundError(f"XML-Datei nicht gefunden: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        logging.debug("XML-Datei erfolgreich geparst")

        # GPX-Element erstellen
        gpx = ET.Element("gpx", version="1.1", creator="GPX Generator Script")
        track_count = 0  # Zähler für Tracks

        # Transformer für Koordinatenumwandlung
        transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        transformer_to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

        # Durch alle IntersectionGeometry-Elemente iterieren
        intersections = root.findall(".//IntersectionGeometry")
        if not intersections:
            logging.warning("Keine IntersectionGeometry-Elemente gefunden")
            return

        for intersection in intersections:
            name = intersection.find("name").text if intersection.find("name") is not None else "Unnamed Intersection"
            logging.debug(f"Verarbeite Intersection: {name}")

            # Referenzpunkt extrahieren
            ref_point = intersection.find("refPoint")
            if ref_point is None:
                logging.warning(f"Kein refPoint in Intersection {name} gefunden. Überspringe.")
                continue
            lat_elem = ref_point.find("lat")
            long_elem = ref_point.find("long")
            if lat_elem is None or long_elem is None:
                logging.warning(f"lat oder long fehlt in refPoint von Intersection {name}")
                continue
            try:
                ref_lat = float(lat_elem.text) / 10000000
                ref_long = float(long_elem.text) / 10000000
            except (ValueError, TypeError) as e:
                logging.error(f"Ungültige Werte für lat/long in refPoint von Intersection {name}: {e}")
                continue

            logging.debug(f"Referenzpunkt: lat={ref_lat}, long={ref_long}")

            # Referenzpunkt in UTM umwandeln
            try:
                ref_x, ref_y = transformer_to_utm.transform(ref_long, ref_lat)
                logging.debug(f"Referenzpunkt in UTM: x={ref_x}, y={ref_y}")
            except Exception as e:
                logging.error(f"Fehler bei UTM-Transformation für Intersection {name}: {e}")
                continue

            # LaneSet verarbeiten
            lane_set = intersection.find("laneSet")
            if lane_set is None:
                logging.warning(f"Kein laneSet in Intersection {name} gefunden")
                continue

            for lane in lane_set.findall("GenericLane"):
                lane_id = lane.find("laneID").text if lane.find("laneID") is not None else "Unnamed Lane"
                logging.debug(f"Verarbeite Lane: {lane_id}")

                # Track für diese Lane erstellen
                trk = ET.SubElement(gpx, "trk")
                ET.SubElement(trk, "name").text = f"{name} - Lane {lane_id}"
                trkseg = ET.SubElement(trk, "trkseg")
                track_count += 1
                point_count = 0  # Zähler für Punkte in dieser Lane

                # NodeList verarbeiten
                node_list = lane.find("nodeList")
                if node_list is None:
                    logging.warning(f"Kein nodeList in Lane {lane_id} gefunden")
                    continue

                nodes = node_list.find("nodes")
                if nodes is None:
                    logging.warning(f"Kein nodes-Element in Lane {lane_id} gefunden")
                    continue

                # Jede NodeXY als absoluten Offset vom Referenzpunkt behandeln
                for node_xy in nodes.findall("NodeXY"):
                    delta = node_xy.find("delta")
                    if delta is None:
                        logging.warning(f"Kein delta-Element in NodeXY von Lane {lane_id}")
                        continue

                    # Den node-XY*-Tag finden und x/y extrahieren
                    xy_elem = None
                    for child in delta:
                        if child.tag.startswith("node-XY"):
                            xy_elem = child
                            break

                    if xy_elem is None:
                        logging.warning(f"Kein node-XY*-Tag in delta von Lane {lane_id}")
                        continue

                    x_text = xy_elem.find("x").text
                    y_text = xy_elem.find("y").text
                    if x_text is None or y_text is None:
                        logging.warning(f"x oder y fehlt in node-XY von Lane {lane_id}")
                        continue

                    try:
                        offset_x_cm = int(x_text)
                        offset_y_cm = int(y_text)
                    except ValueError as e:
                        logging.error(f"Ungültige x/y-Werte in NodeXY von Lane {lane_id}: {e}")
                        continue

                    logging.debug(f"NodeXY in Lane {lane_id}: x={offset_x_cm} cm, y={offset_y_cm} cm")

                    # In Meter umwandeln
                    offset_x_m = offset_x_cm / 100
                    offset_y_m = offset_y_cm / 100

                    # Absolute UTM-Koordinaten berechnen
                    abs_x = ref_x + offset_x_m
                    abs_y = ref_y + offset_y_m

                    # Zurück zu WGS84 (lon, lat) umwandeln
                    try:
                        lon, lat = transformer_to_wgs.transform(abs_x, abs_y)
                        logging.debug(f"Umgewandelte Koordinaten in Lane {lane_id}: lon={lon}, lat={lat}")
                    except Exception as e:
                        logging.error(f"Fehler bei WGS84-Transformation in Lane {lane_id}: {e}")
                        continue

                    # Trackpoint hinzufügen
                    trkpt = ET.SubElement(trkseg, "trkpt", lat=str(lat), lon=str(lon))
                    point_count += 1

                    # Optionale Beschreibung mit dWidth hinzufügen
                    attributes = node_xy.find("attributes")
                    if attributes is not None:
                        d_width = attributes.find("dWidth")
                        if d_width is not None and d_width.text is not None:
                            ET.SubElement(trkpt, "desc").text = f"Width: {d_width.text} cm"
                            logging.debug(f"dWidth hinzugefügt in Lane {lane_id}: {d_width.text} cm")

                logging.debug(f"Anzahl der Punkte in Lane {lane_id}: {point_count}")

        if track_count == 0:
            logging.warning(f"Keine Tracks erstellt für {xml_path}. GPX-Datei wird nicht geschrieben.")
            return

        # GPX-Datei schreiben
        output_dir = os.path.dirname(output_gpx_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.debug(f"Ausgabeordner erstellt: {output_dir}")

        ET.indent(gpx, space="  ", level=0)
        gpx_tree = ET.ElementTree(gpx)
        gpx_tree.write(output_gpx_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ GPX-Datei gespeichert: {output_gpx_path}")
        logging.info(f"GPX-Datei gespeichert: {output_gpx_path} mit {track_count} Tracks")

    except Exception as e:
        print(f"❌ Fehler beim Erstellen der GPX-Datei für {xml_path}: {str(e)}")
        logging.error(f"Fehler beim Erstellen der GPX-Datei für {xml_path}: {str(e)}")

def process_all_xml_files(input_folder, output_folder):
    """Verarbeitet alle XML-Dateien im input_folder und generiert GPX-Dateien im output_folder."""
    logging.debug(f"Verarbeite Ordner: {input_folder}")
    
    try:
        # Sicherstellen, dass der Eingabeordner existiert
        if not os.path.exists(input_folder):
            logging.error(f"Eingabeordner existiert nicht: {input_folder}")
            raise FileNotFoundError(f"Eingabeordner nicht gefunden: {input_folder}")

        # Sicherstellen, dass der Ausgabeordner existiert
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logging.debug(f"Ausgabeordner erstellt: {output_folder}")

        # Alle Dateien im input_folder durchsuchen
        xml_files_found = False
        for filename in os.listdir(input_folder):
            if filename.endswith(".xml"):
                xml_files_found = True
                xml_path = os.path.join(input_folder, filename)
                gpx_filename = os.path.splitext(filename)[0] + ".gpx"
                output_gpx_path = os.path.join(output_folder, gpx_filename)

                print(f"📄 Verarbeite {xml_path}...")
                logging.debug(f"Erstelle GPX-Datei: {output_gpx_path}")
                generate_gpx_from_mapem_xml(xml_path, output_gpx_path)

        if not xml_files_found:
            logging.warning(f"Keine XML-Dateien im Ordner {input_folder} gefunden")

    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten der DESCRIBE YOUR PROBLEM HERE XML-Dateien im Ordner {input_folder}: {str(e)}")
        logging.error(f"Fehler beim Verarbeiten der XML-Dateien im Ordner {input_folder}: {str(e)}")

if __name__ == "__main__":
    input_folder = "output_final"
    output_folder = "output_gpx"
    process_all_xml_files(input_folder, output_folder)