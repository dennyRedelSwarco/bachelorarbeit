from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pyproj import Transformer

def polygon_centerline(poly: Polygon, dx=10.0, smooth_window=21, smooth_order=1, show_plots=True):
    # ==== SCHRITT 1: Polygon rotieren ====
    oriented_rect = poly.minimum_rotated_rectangle
    rect_coords = list(oriented_rect.exterior.coords)
    p0, p1 = rect_coords[0], rect_coords[1]
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.title("Schritt 1: Rotation entlang Hauptachse")
        x1, y1 = poly.exterior.xy
        x2, y2 = poly_rotated.exterior.xy
        plt.plot(x1, y1, label="Original", linestyle='--', alpha=0.5)
        plt.plot(x2, y2, label="Rotiert", color='blue')
        plt.axis("equal")
        plt.legend()
        plt.show()

    # ==== SCHRITT 2: Querschnitts-Mittelpunkte, Breiten und AVG(x+y) ====
    minx, miny, maxx, maxy = poly_rotated.bounds
    xs = np.arange(minx, maxx, dx)
    centers = []
    cutlines = []
    widths = []
    ortho_vectors = []

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

    # Berechne AVG(x+y) der Mittelpunkte
    avg_xy = np.mean([x + y for x, y in centers]) if centers else 0.0

    # Breitenberechnung senkrecht zur provisorischen Mittellinie ohne Skalierung
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
            ortho_vectors.append((p_curr, ortho))

    avg_width = np.mean(widths) if widths else 0.0

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.title("Schritt 2: Querschnitte & Mittelpunkte")
        x_r, y_r = poly_rotated.exterior.xy
        plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
        for line in cutlines:
            x_cut, y_cut = line.xy
            plt.plot(x_cut, y_cut, color='gray', linewidth=0.5, alpha=0.5)
        plt.scatter(xs_c, ys_c, color='red', label="Mittelpunkte")
        plt.axis("equal")
        plt.legend()
        plt.show()

    # ==== SCHRITT 3: Glätten der Mittellinie ====
    if len(centers) < smooth_window:
        raise ValueError("Zu wenige Schnittpunkte für Glättung.")
    ys_smooth = savgol_filter(ys_c, window_length=smooth_window, polyorder=smooth_order)
    center_rotated = LineString(zip(xs_c, ys_smooth))

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.title("Schritt 3: Geglättete Mittellinie (rotiert)")
        plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
        plt.plot(xs_c, ys_smooth, color='green', linewidth=2, label="Geglättete Linie")
        plt.axis("equal")
        plt.legend()
        plt.show()

    # ==== SCHRITT 4: Rückrotation und Visualisierung mit durchschnittlicher Breite ====
    center_final = rotate(center_rotated, angle, origin=poly.centroid, use_radians=False)

    # Erstelle Querschnittslinien mit durchschnittlicher Breite
    from shapely.ops import substring

    # Querschnittslinien direkt orthogonal zur geglätteten finalen Mittellinie
    ortho_cutlines = []
    num_samples = 10
    line_length = center_final.length
    samples = np.linspace(0, line_length, num_samples)

    #Alle Punkte außer den letzen und den ersten
    for s in samples[1:-1]:  # nicht die allerersten/letzten Punkte verwenden
        pt = center_final.interpolate(s)
        delta = 1e-3 * line_length
        pt_before = center_final.interpolate(s - delta)
        pt_after = center_final.interpolate(s + delta)

        # Tangentenrichtung (dx, dy)
        tangent = np.array([pt_after.x - pt_before.x, pt_after.y - pt_before.y])
        if np.linalg.norm(tangent) == 0:
            continue
        tangent = tangent / np.linalg.norm(tangent)

        # Orthogonale Richtung
        ortho = np.array([-tangent[1], tangent[0]])
        
        # Linienendpunkte berechnen
        p1 = (pt.x - ortho[0] * avg_width / 2, pt.y - ortho[1] * avg_width / 2)
        p2 = (pt.x + ortho[0] * avg_width / 2, pt.y + ortho[1] * avg_width / 2)
        
        ortho_cutlines.append(LineString([p1, p2]))

        # Ersten und letzten Punkt separat behandeln
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
        plt.figure(figsize=(8, 6))
        plt.title("Schritt 4: Mittellinie und Breiten im Originalkoordinatensystem")
        x_orig, y_orig = poly.exterior.xy
        x_line, y_line = center_final.xy
        plt.plot(x_orig, y_orig, label="Originalpolygon")
        plt.plot(x_line, y_line, color='red', linewidth=2, label="Mittellinie")
        for line in ortho_cutlines:
            x_cut, y_cut = line.xy
            plt.plot(x_cut, y_cut, color='blue', linewidth=1, alpha=0.7, label="Breiten" if line == ortho_cutlines[0] else "")
        plt.axis("equal")
        plt.legend()
        plt.text(min(x_orig), max(y_orig), f"Durchschnittliche Breite: {avg_width:.2f}\nAVG(x+y): {avg_xy:.2f}", fontsize=12, color='black')
        plt.show()

    return center_final, avg_width, avg_xy

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

# Vectors-Liste erstellen


def print_centerline_geocoords(centerline: LineString, min_x: float, min_y: float, scale: float, num_points: int = 10):
    """
    Gibt gleichmäßig verteilte Punkte entlang einer Mittellinie in WGS84 aus.

    :param centerline: Die Mittellinie als Shapely LineString (z. B. von polygon_centerline)
    :param min_x: min_x aus render_gml_to_image
    :param min_y: min_y aus render_gml_to_image
    :param scale: Skalierung (ebenfalls aus render_gml_to_image)
    :param num_points: Anzahl der Punkte (Standard: 10, inkl. Anfang & Ende)
    """

    if centerline.length == 0 or len(centerline.coords) < 2:
        print("❌ Mittellinie ist leer oder zu kurz.")
        return

    # Punkte gleichmäßig entlang der Länge der Linie
    distances = np.linspace(0, centerline.length, num_points)
    sampled_points = [centerline.interpolate(d) for d in distances]

    # Zurücktransformieren: Bild → Original → WGS84
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

    geo_points = []
    for i, pt in enumerate(sampled_points):
        x_img, y_img = pt.x, pt.y
        x_orig = x_img / scale + min_x
        y_orig = y_img / scale + min_y
        lon, lat = transformer.transform(x_orig, y_orig)
        geo_points.append((lon, lat))
        print(f"🌍 Punkt {i+1}: ({lon:.6f}, {lat:.6f})")

    return geo_points
