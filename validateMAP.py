import os
from lxml import etree

def validate_xml(xml_file_path, xsd_file_path):
    try:
        # Überprüfe, ob die XSD-Datei existiert
        if not os.path.exists(xsd_file_path):
            print(f"Fehler: Die XSD-Datei {xsd_file_path} existiert nicht.")
            return False

        # Überprüfe, ob die XML-Datei existiert
        if not os.path.exists(xml_file_path):
            print(f"Fehler: Die XML-Datei {xml_file_path} existiert nicht.")
            return False

        # Lade das XSD-Schema
        try:
            xsd_doc = etree.parse(xsd_file_path)
            xsd = etree.XMLSchema(xsd_doc)
        except etree.XMLSchemaParseError as e:
            print(f"Fehler beim Parsen der XSD-Datei {xsd_file_path}: {e}")
            return False

        # Lade die XML-Datei
        try:
            xml_doc = etree.parse(xml_file_path)
        except etree.XMLSyntaxError as e:
            print(f"Fehler beim Parsen der XML-Datei {xml_file_path}: {e}")
            return False

        # Überprüfe den Namensraum der XML-Datei
        root = xml_doc.getroot()
        xml_namespace = root.nsmap.get(None, None)
        print(f"Namensraum der XML-Datei: {xml_namespace if xml_namespace else 'Kein Namensraum'}")

        # Validiere die XML-Datei gegen das XSD-Schema
        if xsd.validate(xml_doc):
            print(f"Die Datei {xml_file_path} ist gegen das Schema {xsd_file_path} gültig.")
            return True
        else:
            print(f"Die Datei {xml_file_path} ist nicht gültig. Fehler:")
            for error in xsd.error_log:
                print(f"Zeile {error.line}: {error.message}")
            print("\nHinweis: Der Fehler könnte auf einen Namensraum-Konflikt zurückzuführen sein. "
                  "Die XML-Datei verwendet keinen Namensraum. Stelle sicher, dass die XSD-Datei "
                  "ebenfalls keinen Namensraum definiert (kein targetNamespace, elementFormDefault='unqualified').")
            return False

    except Exception as e:
        print(f"Allgemeiner Fehler: {e}")
        return False

def main():
    # Pfade zu den Dateien
    xml_file_path = os.path.join("output_final", "map_its_features_knoten_0417.xml")
    xsd_file_path = "demoCheckMAp.xsd"
    
    # Validiere die XML-Datei
    validate_xml(xml_file_path, xsd_file_path)

if __name__ == "__main__":
    main()