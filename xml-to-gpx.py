import xml.etree.ElementTree as ET
from datetime import datetime
import os

def xml_to_gpx(xml_path, output_gpx):
    try:
        # Überprüfen, ob die Datei existiert
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML-Datei nicht gefunden: {xml_path}")

        # XML-Datei parsen
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Referenzpunkt extrahieren
        ref_point = root.find("refPoint")
        if ref_point is None:
            raise ValueError("Kein <refPoint>-Element in der XML-Datei gefunden.")
        
        lat_elem = ref_point.find("lat")
        long_elem = ref_point.find("long")
        if lat_elem is None or long_elem is None:
            raise ValueError("Fehlende <lat> oder <long> Elemente in <refPoint>.")

        ref_lat = int(lat_elem.text) / 10000000  # Zu Dezimalgraden
        ref_lon = int(long_elem.text) / 10000000
        print(f"Referenzpunkt: (lat={ref_lat:.6f}, lon={ref_lon:.6f})")

        # GPX-Struktur erstellen
        gpx = ET.Element("gpx", version="1.1", creator="Custom Script")
        metadata = ET.SubElement(gpx, "metadata")
        ET.SubElement(metadata, "time").text = datetime.now().isoformat()

        # Nodes und Lanes durchsuchen
        nodes = root.find("nodes")
        if nodes is None:
            raise ValueError("Kein <nodes>-Element in der XML-Datei gefunden.")

        for node in nodes.findall("node"):
            node_id = node.find("nodeId")
            node_id_text = node_id.text if node_id is not None else "unknown"
            lanes = node.find("lanes")
            if lanes is None:
                print(f"Warnung: Keine <lanes> in Node {node_id_text} gefunden.")
                continue

            for lane in lanes.findall("lane"):
                lane_id = lane.find("laneId")
                lane_id_text = lane_id.text if lane_id is not None else "unknown"
                trk = ET.SubElement(gpx, "trk")
                ET.SubElement(trk, "name").text = f"Lane_{lane_id_text}"
                trkseg = ET.SubElement(trk, "trkseg")

                # Delta-Koordinaten verarbeiten
                lane_nodes = lane.find("nodes")
                if lane_nodes is None:
                    print(f"Warnung: Keine <nodes> in Lane {lane_id_text} gefunden.")
                    continue

                # Kumulative Koordinaten berechnen (delta-Werte sind relativ zum vorherigen Punkt)
                current_lat = ref_lat
                current_lon = ref_lon
                trkpt = ET.SubElement(trkseg, "trkpt", lat=str(current_lat), lon=str(current_lon))
                ET.SubElement(trkpt, "name").text = f"RefPoint_Lane_{lane_id_text}"

                for delta_node in lane_nodes.findall("node"):
                    delta = delta_node.find("delta")
                    if delta is None:
                        print(f"Warnung: Kein <delta> in Node von Lane {lane_id_text} gefunden.")
                        continue
                    x_cm = int(delta.find("x").text)
                    y_cm = int(delta.find("y").text)
                    delta_lat = y_cm / (111111 * 100)  # 1 Grad ≈ 111.111 m
                    delta_lon = x_cm / (66666 * 100)   # 1 Grad ≈ 66.666 m bei ~53.6°
                    current_lat += delta_lat
                    current_lon += delta_lon
                    trkpt = ET.SubElement(trkseg, "trkpt", lat=str(current_lat), lon=str(current_lon))
                    ET.SubElement(trkpt, "name").text = f"Point_Lane_{lane_id_text}"

        # GPX-Datei speichern
        tree = ET.ElementTree(gpx)
        ET.indent(tree, space="  ")
        tree.write(output_gpx)
        print(f"✅ GPX gespeichert: {output_gpx}")

    except FileNotFoundError as e:
        print(f"❌ Fehler: {str(e)}")
    except ValueError as e:
        print(f"❌ Fehler: {str(e)}")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Konvertieren: {str(e)}")

if __name__ == "__main__":
    xml_path = "output_final/map_its_features_knoten_0417.xml"  # Passe den Pfad an
    output_gpx = xml_path.replace(".xml", ".gpx")
    xml_to_gpx(xml_path, output_gpx)