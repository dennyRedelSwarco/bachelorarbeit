import xml.etree.ElementTree as ET
import os
from pyproj import Transformer
import logging

def generate_map_its_xml(centerline_coords, output_path, lower_corner_wgs84):
    """Generiert eine Utopia-konforme MAPEM XML-Datei basierend auf centerline_coords."""
    try:
        mapem_elem = ET.Element("MAPEM")
        
        # Header (Standarddaten, aus Beispiel-XML)
        header_elem = ET.SubElement(mapem_elem, "header")
        ET.SubElement(header_elem, "protocolVersion").text = "2"
        ET.SubElement(header_elem, "messageID").text = "5"
        ET.SubElement(header_elem, "stationID").text = "19531"
        
        # Map (Standarddaten, aus Beispiel-XML)
        map_elem = ET.SubElement(mapem_elem, "map")
        ET.SubElement(map_elem, "msgIssueRevision").text = "0"
        
        # Intersections (gemäß Beispiel-XML und MAPEM-TS_2.xsd)
        intersections_elem = ET.SubElement(map_elem, "intersections")
        intersection_elem = ET.SubElement(intersections_elem, "IntersectionGeometry")
        ET.SubElement(intersection_elem, "name").text = "MAP_ITS_17_1729_2.1"
        
        # ID (Standarddaten, aus Beispiel-XML)
        id_elem = ET.SubElement(intersection_elem, "id")
        ET.SubElement(id_elem, "region").text = "3"
        ET.SubElement(id_elem, "id").text = "1729"
        
        ET.SubElement(intersection_elem, "revision").text = "1"
        
        # RefPoint: Verwende lower_corner_wgs84 als Referenzpunkt
        if not centerline_coords:
            print("❌ Keine Mittelliniendaten vorhanden. XML-Generierung abgebrochen.")
            logging.info("Keine Mittelliniendaten vorhanden. XML-Generierung abgebrochen.")
            return
        
        ref_lon, ref_lat = lower_corner_wgs84  # (lon, lat) aus lower_corner_wgs84
        ref_point = ET.SubElement(intersection_elem, "refPoint")
        ET.SubElement(ref_point, "lat").text = str(int(ref_lat * 10000000))
        ET.SubElement(ref_point, "long").text = str(int(ref_lon * 10000000))
        
        # Lane Width (Standardwert oder Durchschnitt aus Daten, in cm)
        avg_width = centerline_coords[0].get("avg_width", 325)  # Fallback auf 325 cm
        ET.SubElement(intersection_elem, "laneWidth").text = str(int(avg_width))  # In cm
        
        # Speed Limits (Standarddaten, aus Beispiel-XML)
        speed_limits_elem = ET.SubElement(intersection_elem, "speedLimits")
        speed_limit_elem = ET.SubElement(speed_limits_elem, "RegulatorySpeedLimit")
        type_elem = ET.SubElement(speed_limit_elem, "type")
        ET.SubElement(type_elem, "vehicleMaxSpeed")
        ET.SubElement(speed_limit_elem, "speed").text = "694"
        
        # Transformer für Koordinaten (UTM-Berechnung für relative Offsets)
        transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        ref_x, ref_y = transformer_to_utm.transform(ref_lon, ref_lat)
        
        # LaneSet: Generiere Lanes basierend auf centerline_coords
        lane_set_elem = ET.SubElement(intersection_elem, "laneSet")
        
        lane_groups = {}
        for coord in centerline_coords:
            polygon_id = coord.get("polygon_id", 0)  # Fallback auf 0, falls nicht vorhanden
            if polygon_id not in lane_groups:
                lane_groups[polygon_id] = []
            lane_groups[polygon_id].append(coord)
        
        for lane_id, (polygon_id, coords) in enumerate(lane_groups.items(), 10):
            lane_elem = ET.SubElement(lane_set_elem, "GenericLane")
            ET.SubElement(lane_elem, "laneID").text = str(lane_id)
            ET.SubElement(lane_elem, "ingressApproach").text = "3"
            
            # Lane Attributes (Standarddaten, aus Beispiel-XML)
            lane_attrs_elem = ET.SubElement(lane_elem, "laneAttributes")
            ET.SubElement(lane_attrs_elem, "directionalUse").text = "10"
            ET.SubElement(lane_attrs_elem, "sharedWith").text = "0001100000"
            lane_type = ET.SubElement(lane_attrs_elem, "laneType")
            ET.SubElement(lane_type, "vehicle").text = "00000000"
            
            # NodeList: Relative Koordinaten berechnen
            node_list_elem = ET.SubElement(lane_elem, "nodeList")
            nodes_elem = ET.SubElement(node_list_elem, "nodes")
            
            for i, coord in enumerate(coords):
                if "geocoordinate" not in coord:
                    print(f"❌ Fehler: Schlüssel 'geocoordinate' fehlt in {coord}")
                    logging.error(f"Schlüssel 'geocoordinate' fehlt in {coord}")
                    continue
                coord_str = coord["geocoordinate"]
                lon_p, lat_p = map(float, coord_str.strip("()").split(","))
                x_p, y_p = transformer_to_utm.transform(lon_p, lat_p)
                x_cm = max(min(int((x_p - ref_x) * 100), 8191), -8191)  # Begrenze auf [-8191, 8191]
                y_cm = max(min(int((y_p - ref_y) * 100), 8191), -8191)  # Begrenze auf [-8191, 8191]
                
                # Berechne Breite (bereits in Zentimetern)
                avg_width = coord.get("avg_width", 325)  # Fallback auf 325 cm
                d_width = int(avg_width)  # Bereits in cm, keine Umrechnung nötig
                
                node_elem = ET.SubElement(nodes_elem, "NodeXY")
                delta_elem = ET.SubElement(node_elem, "delta")
                node_xy_tag = f"node-XY{6 if i % 2 == 0 else 5}"
                node_xy_elem = ET.SubElement(delta_elem, node_xy_tag)
                ET.SubElement(node_xy_elem, "x").text = str(x_cm)
                ET.SubElement(node_xy_elem, "y").text = str(y_cm)
                
                # Attributes mit dWidth
                attributes_elem = ET.SubElement(node_elem, "attributes")
                ET.SubElement(attributes_elem, "dWidth").text = str(d_width)
        
        # XML schreiben
        tree = ET.ElementTree(mapem_elem)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ MAPEM XML gespeichert: {output_path}")
        logging.info(f"MAPEM XML gespeichert: {output_path}")
    
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der XML-Datei: {str(e)}")
        logging.info(f"Fehler beim Erstellen der XML-Datei für {output_path}: {str(e)}")