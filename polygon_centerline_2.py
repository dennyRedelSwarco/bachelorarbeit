import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter
from pyproj import Transformer
import pandas as pd
import logging

def polygon_centerline_polynomial_only(poly: Polygon, dx=5.0, degrees=[2, 3], smooth_window=19, smooth_order=1, min_x=0.0, min_y=0.0, scale=1.0, show_plots=False):
    logging.info("Starte polygon_centerline_polynomial_only...")
    print("Starte polygon_centerline_polynomial_only...")

    output_dir = "output_polynomial_only"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ausgabeverzeichnis '{output_dir}' erstellt oder vorhanden.")
    print(f"Ausgabeverzeichnis '{output_dir}' erstellt oder vorhanden.")

    try:
        # Schritt 1: Rotation entlang Hauptachse
        logging.info("Schritt 1: Berechne Rotation...")
        print("Schritt 1: Berechne Rotation...")
        oriented_rect = poly.minimum_rotated_rectangle
        rect_coords = list(oriented_rect.exterior.coords)
        p0, p1 = rect_coords[0], rect_coords[1]
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

        minx, miny, maxx, maxy = poly_rotated.bounds
        width = maxx - minx
        height = maxy - miny
        if width < height:
            logging.info("Hauptachse liegt vertikal, korrigiere um 90 Grad.")
            print("Hauptachse liegt vertikal, korrigiere um 90 Grad.")
            angle += 90.0
            poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

        logging.info(f"Polygon rotiert um {angle:.2f} Grad.")
        print(f"Polygon rotiert um {angle:.2f} Grad.")

        if show_plots:
            logging.info("Erstelle Plot für Schritt 1...")
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
            logging.info("Plot für Schritt 1 gespeichert: output_plots/step1_rotation.png")
            print("Plot für Schritt 1 gespeichert: output_plots/step1_rotation.png")
            plt.show(block=True)
            plt.close()

        # Schritt 2: Vertikale Schnitte & Mittelpunkte
        logging.info("Schritt 2: Berechne Querschnitts-Mittelpunkte...")
        print("Schritt 2: Berechne Querschnitts-Mittelpunkte...")
        minx, miny, maxx, maxy = poly_rotated.bounds
        xs = np.arange(minx, maxx, dx)
        centers = []
        cutlines = []
        for x in xs:
            cutline = LineString([(x, miny - 10), (x, maxy + 10)])
            inter = poly_rotated.intersection(cutline)

            if inter.is_empty:
                continue
            ys = []
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
        logging.info(f"Schritt 2: {len(centers)} Mittelpunkte gefunden.")
        print(f"Schritt 2: {len(centers)} Mittelpunkte gefunden.")

        if show_plots:
            logging.info("Erstelle Plot für Schritt 2...")
            print("Erstelle Plot für Schritt 2...")
            plt.figure(figsize=(8, 6))
            plt.title("Schritt 2: Vertikale Querschnitte & Mittelpunkte (rotiert)")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
            for line in cutlines:
                x_cut, y_cut = line.xy
                plt.plot(x_cut, y_cut, color='gray', linewidth=0.5, alpha=0.5)
            plt.scatter(xs_c, ys_c, color='red', label="Mittelpunkte")
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step2_vertical_cuts.png"))
            logging.info("Plot für Schritt 2 gespeichert: output_plots/step2_vertical_cuts.png")
            print("Plot für Schritt 2 gespeichert: output_plots/step2_vertical_cuts.png")
            plt.show(block=True)
            plt.close()

        if len(centers) < 4:
            logging.error(f"Zu wenige Mittelpunkte ({len(centers)}) für Polynom-Anpassung.")
            print(f"❌ Zu wenige Mittelpunkte ({len(centers)}) für Polynom-Anpassung.")
            return None, 0.0, 0.0, []

        # Hilfsfunktion zur Berechnung des kleinsten Gesamtwinkels
        def max_turn_angle(linestring):
            coords = list(linestring.coords)
            if len(coords) < 3:
                logging.info("Zu wenige Punkte für Winkelberechnung.")
                print("Zu wenige Punkte für Winkelberechnung.")
                return 0.0

            start_tangent = np.array(coords[1]) - np.array(coords[0])
            end_tangent = np.array(coords[-1]) - np.array(coords[-2])
            norm_start = np.linalg.norm(start_tangent)
            norm_end = np.linalg.norm(end_tangent)
            if norm_start == 0 or norm_end == 0:
                logging.info("Ungültige Tangentenvektoren (keine Länge).")
                print("Ungültige Tangentenvektoren (keine Länge).")
                return 0.0

            dot = np.dot(start_tangent, end_tangent)
            cos_theta = dot / (norm_start * norm_end)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
            smallest_angle = min(angle, 180.0 - angle)
            logging.info(f"Gesamtwinkel zwischen Tangenten: {angle:.2f} Grad, Kleinster Winkel: {smallest_angle:.2f} Grad")
            print(f"Gesamtwinkel zwischen Tangenten: {angle:.2f} Grad, Kleinster Winkel: {smallest_angle:.2f} Grad")
            return smallest_angle

        # Schritt 3: Polynom-Anpassung für jeden Grad
        best_centerline = None
        best_angle = float('inf')
        best_avg_width = 0.0
        centerline_coords = []
        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

        for degree in degrees:
            try:
                logging.info(f"→ Starte Polynom-Anpassung Grad {degree}...")
                print(f"→ Starte Polynom-Anpassung Grad {degree}...")
                X = np.array(xs_c).reshape(-1, 1)
                y = np.array(ys_c)

                model = make_pipeline(
                    PolynomialFeatures(degree=degree),
                    RANSACRegressor(
                        estimator=Ridge(alpha=10.0),
                        residual_threshold=2.0,
                        min_samples=0.5,
                        random_state=42
                    )
                )
                model.fit(X, y)
                inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
                logging.info(f"Grad {degree}: {np.sum(inlier_mask)} von {len(centers)} Punkten als Inlier erkannt.")
                print(f"Grad {degree}: {np.sum(inlier_mask)} von {len(centers)} Punkten als Inlier erkannt.")

                x_smooth = np.linspace(min(xs_c), max(xs_c), 300).reshape(-1, 1)
                y_smooth = model.predict(x_smooth)

                if smooth_window > 0 and smooth_order > 0 and smooth_window > smooth_order:
                    if smooth_window % 2 == 0:
                        smooth_window += 1
                    y_smooth = savgol_filter(y_smooth, window_length=smooth_window, polyorder=smooth_order)
                    logging.info(f"Glättung angewendet: window={smooth_window}, order={smooth_order}")
                    print(f"✓ Glättung angewendet: window={smooth_window}, order={smooth_order}")
                else:
                    logging.info(f"Glättung übersprungen (window/order ungültig)")
                    print(f"⚠️ Glättung übersprungen (window/order ungültig)")

                centerline = LineString(zip(x_smooth.flatten(), y_smooth))
                centerline_rotated_back = rotate(centerline, angle, origin=poly.centroid, use_radians=False)
                max_angle = max_turn_angle(centerline_rotated_back)
                angle_threshold = 90.0
                logging.info(f"Kleinster Gesamtwinkel für Grad {degree}: {max_angle:.2f} Grad")
                print(f"Kleinster Gesamtwinkel für Grad {degree}: {max_angle:.2f} Grad")

                # Breitenberechnung
                widths = []
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
                avg_width = np.mean(widths) if widths else 0.0
                logging.info(f"Durchschnittliche Breite für Grad {degree}: {avg_width:.2f}")
                print(f"Durchschnittliche Breite für Grad {degree}: {avg_width:.2f}")

                # Wähle die beste Mittellinie
                if max_angle <= angle_threshold and max_angle < best_angle:
                    best_centerline = centerline_rotated_back
                    best_angle = max_angle
                    best_avg_width = avg_width

                    # Geokoordinaten
                    distances = np.linspace(0, best_centerline.length, 10)
                    sampled_points = [best_centerline.interpolate(d) for d in distances]
                    centerline_coords = []
                    for i, pt in enumerate(sampled_points):
                        x_img, y_img = pt.x, pt.y
                        x_orig = x_img / scale + min_x
                        y_orig = (1024 - y_img) / scale + min_y
                        lon, lat = transformer.transform(x_orig, y_orig)
                        centerline_coords.append({
                            "point_index": i,
                            "image_x": x_img,
                            "image_y": y_img,
                            "original_x": x_orig,
                            "original_y": y_orig,
                            "lon": lon,
                            "lat": lat,
                            "avg_width": avg_width
                        })
                        logging.info(f"Punkt {i+1}: ({lon:.6f}, {lat:.6f})")
                        print(f"Punkt {i+1}: ({lon:.6f}, {lat:.6f})")

                # Plotten
                if show_plots:
                    logging.info(f"Erstelle Plot für Grad {degree}...")
                    print(f"Erstelle Plot für Grad {degree}...")
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Mittellinie – Grad {degree} mit Glättung")
                    x_r, y_r = poly_rotated.exterior.xy
                    plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
                    if max_angle <= angle_threshold:
                        plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label=f"Grad {degree}")
                    else:
                        plt.text(min(x_r) + (max(x_r) - min(x_r)) / 2, min(y_r) + (max(y_r) - miny) / 2,
                                 f"Mittellinie ungültig\n(Gesamtwinkel: {max_angle:.2f}° > {angle_threshold}°)",
                                 fontsize=12, color='red', ha='center')
                    plt.scatter(np.array(xs_c)[inlier_mask], np.array(ys_c)[inlier_mask], color='red', s=10, label="Inlier")
                    plt.scatter(np.array(xs_c)[~inlier_mask], np.array(ys_c)[~inlier_mask], color='orange', s=10, label="Outlier")
                    plt.axis("equal")
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f"centerline_degree_{degree}.png"))
                    logging.info(f"Plot gespeichert: {output_dir}/centerline_degree_{degree}.png")
                    print(f"✅ Plot gespeichert: {output_dir}/centerline_degree_{degree}.png")
                    plt.show(block=True)
                    plt.close()

            except Exception as e:
                logging.error(f"Fehler bei Grad {degree}: {str(e)}")
                print(f"❌ Fehler bei Grad {degree}: {str(e)}")

        if best_centerline is None:
            logging.error("Keine gültige Mittellinie gefunden.")
            print("❌ Keine gültige Mittellinie gefunden.")
            return None, 0.0, 0.0, []

        # CSV speichern
        centerline_df = pd.DataFrame(centerline_coords)
        csv_path = os.path.join(output_dir, f"centerline_coords.csv")
        centerline_df.to_csv(csv_path, index=False)
        logging.info(f"Centerline-Koordinaten gespeichert: {csv_path}")
        print(f"Centerline-Koordinaten gespeichert: {csv_path}")

        # Berechne AVG(x+y)
        avg_xy = np.mean([x + y for x, y in centers]) if centers else 0.0
        logging.info(f"AVG(x+y): {avg_xy:.2f}")
        print(f"AVG(x+y): {avg_xy:.2f}")

        logging.info("polygon_centerline_polynomial_only abgeschlossen.")
        print("polygon_centerline_polynomial_only abgeschlossen.")
        return best_centerline, best_avg_width, avg_xy, centerline_coords

    except Exception as e:
        logging.error(f"Fehler in polygon_centerline_polynomial_only: {str(e)}")
        print(f"❌ Fehler in polygon_centerline_polynomial_only: {str(e)}")
        return None, 0.0, 0.0, []