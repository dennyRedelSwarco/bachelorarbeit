import os
import matplotlib
# Setze Backend auf TkAgg für zuverlässige Anzeige
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate
from shapely.ops import substring
import numpy as np
from pyproj import Transformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter

def polygon_centerline(poly: Polygon, dx=10.0, show_plots=True, smooth_window=19, smooth_order=1):
    print("Starte polygon_centerline...")
    
    # Erstelle Ausgabeverzeichnis für Plots
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ausgabeverzeichnis '{output_dir}' erstellt oder vorhanden.")

    try:
        # ==== SCHRITT 1: Polygon rotieren ====
        print("Schritt 1: Berechne Rotation...")
        oriented_rect = poly.minimum_rotated_rectangle
        rect_coords = list(oriented_rect.exterior.coords)
        p0, p1 = rect_coords[0], rect_coords[1]
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)
        print(f"Polygon rotiert um {angle:.2f} Grad.")

        if show_plots:
            print("Erstelle Plot für Schritt 1...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 1: Rotation entlang Hauptachse")
            x1, y1 = poly.exterior.xy
            x2, y2 = poly_rotated.exterior.xy
            plt.plot(x1, y1, label="Original", linestyle='--', alpha=0.5)
            plt.plot(x2, y2, label="Rotiert", color='blue')
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step1_rotation.png"))
            print("Plot für Schritt 1 gespeichert: output_plots/step1_rotation.png")
            plt.show(block=True)  # Blockierend anzeigen
            plt.close()

        # ==== SCHRITT 2: Querschnitts-Mittelpunkte, Breiten und AVG(x+y) ====
        print("Schritt 2: Berechne Querschnitts-Mittelpunkte und Breiten...")
        minx, miny, maxx, maxy = poly_rotated.bounds
        xs = np.arange(minx, maxx, dx)
        centers = []
        cutlines = []
        widths = []
        ortho_vectors = []
        ortho_cutlines_rotated = []

        # Schritt 2.1: Vertikale Querschnittslinien und Mittelpunkte
        print("Schritt 2.1: Berechne vertikale Querschnitte...")
        for x in xs:
            cutline = LineString([(x, miny - 10), (x, maxy + 10)])
            inter = poly_rotated.intersection(cutline)

            if inter.is_empty:
                continue
            if inter.geom_type == 'MultiPoint':
                ys = [pt.y for pt in inter.geoms]
            elif inter.geom_type == 'Point':
                ys = [inter.y]
            elif inter.geom_type.startswith("MultiLine") or inter.geom_type == "LineString":
                if hasattr(inter, 'geoms'):
                    points = [pt for line in inter.geoms for pt in line.coords]
                else:
                    points = list(inter.coords)
                ys = [pt[1] for pt in points]
            else:
                continue

            if ys:
                center_y = np.mean(ys)
                centers.append((x, center_y))
                cutlines.append(cutline)

        xs_c = [p[0] for p in centers]
        ys_c = [p[1] for p in centers]
        print(f"Schritt 2.1: {len(centers)} Mittelpunkte gefunden.")

        if show_plots:
            print("Erstelle Plot für Schritt 2.1...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 2.1: Vertikale Querschnitte & Mittelpunkte (rotiert)")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
            for line in cutlines:
                x_cut, y_cut = line.xy
                plt.plot(x_cut, y_cut, color='gray', linewidth=0.5, alpha=0.5)
            plt.scatter(xs_c, ys_c, color='red', label="Mittelpunkte")
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step2_1_vertical_cuts.png"))
            print("Plot für Schritt 2.1 gespeichert: output_plots/step2_1_vertical_cuts.png")
            plt.show(block=True)  # Blockierend anzeigen
            plt.close()

        # Berechne AVG(x+y) der Mittelpunkte
        avg_xy = np.mean([x + y for x, y in centers]) if centers else 0.0
        print(f"Schritt 2: AVG(x+y) = {avg_xy:.2f}")

        # Schritt 2.2 & 2.3: Breitenberechnung senkrecht zur provisorischen Mittellinie
        print("Schritt 2.2 & 2.3: Berechne Breiten...")
        line_points = list(zip(xs_c, ys_c))
        for i in range(1, len(line_points) - 1):
            p_prev = np.array(line_points[i - 1])
            p_curr = np.array(line_points[i])
            p_next = np.array(line_points[i + 1])
            tangent = ((p_curr - p_prev) + (p_next - p_curr)) / 2
            if np.linalg.norm(tangent) == 0:
                continue
            ortho = np.array([-tangent[1], tangent[0]]) / np.linalg.norm([-tangent[1], tangent[0]])
            p1 = p_curr - ortho * 1000
            p2 = p_curr + ortho * 1000
            cutline = LineString([p1, p2])
            ortho_cutlines_rotated.append(cutline)
            cutline_final = rotate(cutline, angle, origin=poly.centroid, use_radians=False)
            inter = poly.intersection(cutline_final)

            if inter.is_empty:
                continue
            if inter.geom_type == 'MultiPoint':
                points = [(pt.x, pt.y) for pt in inter.geoms]
            elif inter.geom_type == 'Point':
                points = [(inter.x, inter.y)]
            elif inter.geom_type.startswith("MultiLine") or inter.geom_type == "LineString":
                if hasattr(inter, 'geoms'):
                    points = [pt for line in inter.geoms for pt in line.coords]
                else:
                    points = list(inter.coords)
            else:
                continue

            if len(points) >= 2:
                points = np.array(points)
                width = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
                widths.append(width)
                ortho_vectors.append((p_curr, ortho))

        avg_width = np.mean(widths) if widths else 0.0
        print(f"Schritt 2: Durchschnittliche Breite = {avg_width:.2f}")

        if show_plots:
            # Schritt 2.2: Orthogonale Querschnittslinien im rotierten Koordinatensystem
            print("Erstelle Plot für Schritt 2.2...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 2.2: Orthogonale Querschnittslinien (rotiert)")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
            plt.scatter(xs_c, ys_c, color='red', s=10, label="Mittelpunkte", alpha=0.5)
            for cutline in ortho_cutlines_rotated:
                x_cut, y_cut = line.xy
                plt.plot(x_cut, y_cut, color='purple', linewidth=1, alpha=0.7, label="Orthogonale Querschnitte" if cutline == ortho_cutlines_rotated[0] else "")
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step2_2_ortho_cuts_rotated.png"))
            print("Plot für Schritt 2.2 gespeichert: output_plots/step2_2_ortho_cuts_rotated.png")
            plt.show(block=True)  # Blockierend anzeigen
            plt.close()

            # Schritt 2.3: Orthogonale Querschnittslinien im Originalkoordinatensystem
            print("Erstelle Plot für Schritt 2.3...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 2.3: Orthogonale Querschnitte mit Breiten (original)")
            x_orig, y_orig = poly.exterior.xy
            plt.plot(x_orig, y_orig, color='lightgray', label="Originalpolygon")
            for cutline in ortho_cutlines_rotated:
                cutline_final = rotate(cutline, angle, origin=poly.centroid, use_radians=False)
                x_cut, y_cut = cutline_final.xy
                plt.plot(x_cut, y_cut, color='purple', linewidth=1, alpha=0.7, label="Orthogonale Querschnitte" if cutline == ortho_cutlines_rotated[0] else "")
            plt.axis("equal")
            plt.legend()
            plt.text(min(x_orig), max(y_orig), f"Durchschnittliche Breite: {avg_width:.2f}", fontsize=12, color='black')
            plt.savefig(os.path.join(output_dir, "step2_3_ortho_cuts_original.png"))
            print("Plot für Schritt 2.3 gespeichert: output_plots/step2_3_ortho_cuts_original.png")
            plt.show(block=True)  # Blockierend anzeigen
            plt.close()

        # ==== SCHRITT 3: Kubische Polynom-Anpassung mit RANSAC und Glättung ====
        print("Schritt 3: Kubische Polynom-Anpassung der Mittellinie...")
        if len(centers) < 4:
            print(f"Warnung: Zu wenige Schnittpunkte für 3. Grad-Polynom: {len(centers)} gefunden, mindestens 4 benötigt.")
            raise ValueError(f"Zu wenige Schnittpunkte für 3. Grad-Polynom: {len(centers)} gefunden, mindestens 4 benötigt.")

        # Bereite Daten für RANSAC vor
        X = np.array(xs_c).reshape(-1, 1)
        y = np.array(ys_c)
        # Erstelle ein 3. Grad-Polynom-Modell mit RANSAC
        polyreg = make_pipeline(PolynomialFeatures(degree=3), RANSACRegressor(random_state=42))
        polyreg.fit(X, y)
        # Extrahiere Inlier-Maske
        inlier_mask = polyreg.named_steps['ransacregressor'].inlier_mask_
        print(f"Schritt 3: {np.sum(inlier_mask)} von {len(centers)} Punkten als Inlier erkannt.")

        # Generiere glatte Kurve
        x_smooth = np.linspace(min(xs_c), max(xs_c), 100).reshape(-1, 1)
        y_smooth = polyreg.predict(x_smooth)
        
        # Wende Savitzky-Golay-Glättung an
        try:
            if smooth_window > 0 and smooth_order > 0:
                if smooth_window % 2 == 0:
                    smooth_window += 1  # Sicherstellen, dass Fenstergröße ungerade ist
                if smooth_window <= smooth_order:
                    print(f"Warnung: smooth_window ({smooth_window}) muss größer als smooth_order ({smooth_order}) sein. Glättung übersprungen.")
                else:
                    y_smooth = savgol_filter(y_smooth, window_length=smooth_window, polyorder=smooth_order)
                    print(f"Schritt 3: Savitzky-Golay-Glättung angewendet (window={smooth_window}, order={smooth_order}).")
            else:
                print("Schritt 3: Keine Glättung angewendet (smooth_window oder smooth_order <= 0).")
        except ValueError as e:
            print(f"Warnung: Glättung fehlgeschlagen: {str(e)}. Verwende ungegättete Kurve.")

        center_rotated = LineString(zip(x_smooth.flatten(), y_smooth))
        print("Schritt 3: Kubische Polynom-Anpassung und Glättung abgeschlossen.")

        if show_plots:
            print("Erstelle Plot für Schritt 3...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 3: Mittellinie mit 3. Grad-Polynom und Glättung (rotiert)")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
            plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label="Geglättetes Polynom")
            plt.scatter(np.array(xs_c)[inlier_mask], np.array(ys_c)[inlier_mask], color='red', s=10, label="Inlier-Mittelpunkte", alpha=0.5)
            plt.scatter(np.array(xs_c)[~inlier_mask], np.array(ys_c)[~inlier_mask], color='orange', s=10, label="Ausreißer", alpha=0.5)
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step3_polynomial_centerline.png"))
            print("Plot für Schritt 3 gespeichert: output_plots/step3_polynomial_centerline.png")
            plt.show(block=True)  # Blockierend anzeigen
            plt.close()

        # ==== SCHRITT 4: Rückrotation und Visualisierung mit durchschnittlicher Breite ====
        print("Schritt 4: Rückrotation und finale Visualisierung...")
        center_final = rotate(center_rotated, angle, origin=poly.centroid, use_radians=False)

        ortho_cutlines = []
        num_samples = 10
        line_length = center_final.length
        samples = np.linspace(0, line_length, num_samples)

        for s in samples[1:-1]:
            pt = center_final.interpolate(s)
            delta = 1e-3 * line_length
            pt_before = center_final.interpolate(s - delta)
            pt_after = center_final.interpolate(s + delta)

            tangent = np.array([pt_after.x - pt_before.x, pt_after.y - pt_before.y])
            if np.linalg.norm(tangent) == 0:
                continue
            tangent = tangent / np.linalg.norm(tangent)

            ortho = np.array([-tangent[1], tangent[0]])
            
            p1 = (pt.x - ortho[0] * avg_width / 2, pt.y - ortho[1] * avg_width / 2)
            p2 = (pt.x + ortho[0] * avg_width / 2, pt.y + ortho[1] * avg_width / 2)
            
            ortho_cutlines.append(LineString([p1, p2]))

        for s in [0.0, line_length]:
            pt = center_final.interpolate(s)
            if s == 0.0:
                pt_next = center_final.interpolate(s + 1e-3 * line_length)
                tangent = np.array([pt_next.x - pt.x, pt_next.y - pt.y])
            else:
                pt_prev = center_final.interpolate(s - 1e-3 * line_length)
                tangent = np.array([pt.x - pt_prev.x, pt.y - pt_prev.y])

            if np.linalg.norm(tangent) == 0:
                continue
            tangent = tangent / np.linalg.norm(tangent)
            ortho = np.array([-tangent[1], tangent[0]])

            p1 = (pt.x - ortho[0] * avg_width / 2, pt.y - ortho[1] * avg_width / 2)
            p2 = (pt.x + ortho[0] * avg_width / 2, pt.y + ortho[1] * avg_width / 2)
            ortho_cutlines.append(LineString([p1, p2]))

        if show_plots:
            print("Erstelle Plot für Schritt 4...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 4: Finales Ergebnis mit geglätteter Mittellinie")
            
            # Originalpolygon
            x_orig, y_orig = poly.exterior.xy
            plt.plot(x_orig, y_orig, label="Originalpolygon", color="lightgray")
            
            # Rückrotierte Mittellinie (finale Mittellinie)
            x_line, y_line = center_final.xy
            plt.plot(x_line, y_line, color='red', linewidth=2, label="Mittellinie (rückrotiert)")
            
            # Geklättetes Polynom (aus Schritt 3, aber noch in rotierten Koordinaten)
            x_poly, y_poly = center_rotated.xy
            x_poly_final, y_poly_final = rotate(center_rotated, angle, origin=poly.centroid, use_radians=False).xy
            plt.plot(x_poly_final, y_poly_final, color='green', linestyle='--', linewidth=2, label="Geglättetes Polynom (rückrotiert)")

            # Breitenlinien
            for line in ortho_cutlines:
                x_cut, y_cut = line.xy
                plt.plot(x_cut, y_cut, color='blue', linewidth=1, alpha=0.7, label="Breiten" if line == ortho_cutlines[0] else "")

            plt.axis("equal")
            plt.legend()
            plt.text(min(x_orig), max(y_orig), f"Durchschnittliche Breite: {avg_width:.2f}\nAVG(x+y): {avg_xy:.2f}", fontsize=12, color='black')
            plt.savefig(os.path.join(output_dir, "step4_final_centerline.png"))
            print("Plot für Schritt 4 gespeichert: output_plots/step4_final_centerline.png")
            plt.show(block=True)
            plt.close()


        print("polygon_centerline abgeschlossen.")
        return center_final, avg_width, avg_xy

    except Exception as e:
        print(f"Fehler in polygon_centerline: {str(e)}")
        raise

def print_centerline_geocoords(centerline: LineString, min_x: float, min_y: float, scale: float, num_points: int = 10):
    print("Starte print_centerline_geocoords...")
    if centerline.length == 0 or len(centerline.coords) < 2:
        print("❌ Mittellinie ist leer oder zu kurz.")
        return

    distances = np.linspace(0, centerline.length, num_points)
    sampled_points = [centerline.interpolate(d) for d in distances]

    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

    geo_points = []
    for i, pt in enumerate(sampled_points):
        x_img, y_img = pt.x, pt.y
        x_orig = x_img / scale + min_x
        y_orig = y_img / scale + min_y
        lon, lat = transformer.transform(x_orig, y_orig)
        geo_points.append((lon, lat))
        print(f"🌍 Punkt {i+1}: ({lon:.6f}, {lat:.6f})")

    print("print_centerline_geocoords abgeschlossen.")
    return geo_points

# Beispiel-Polygon
coords = [
    [236.0, 590.49560546875],
    [164.74634552001953, 931.8354721069336],
    [166.37900733947754, 970.3330554962158],
    [185.90957260131836, 981.7312297821045],
    [200.83003616333008, 978.218729019165],
    [296.9057865142822, 613.0392112731934],
    [270.0, 588.49755859375],
    [236.0, 590.49560546875]
]
polygon = Polygon(coords)

if __name__ == "__main__":
    print("Ausführung von test.py...")
    try:
        center_final, avg_width, avg_xy = polygon_centerline(polygon, dx=10.0, show_plots=True, smooth_window=11, smooth_order=1)
        print(f"Ergebnis: avg_width={avg_width:.2f}, avg_xy={avg_xy:.2f}")
        # Beispielaufruf für print_centerline_geocoords (mit Dummy-Werten für min_x, min_y, scale)
        print_centerline_geocoords(center_final, min_x=0, min_y=0, scale=1.0)
    except Exception as e:
        print(f"Fehler beim Ausführen von test.py: {str(e)}")
    print("test.py beendet.")