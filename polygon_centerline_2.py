
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

def polygon_centerline_polynomial_only(poly: Polygon, dx=20.0, degrees=[2, 3], smooth_window=51, smooth_order=3):
    print("Starte polygon_centerline_polynomial_only...")

    output_dir = "output_polynomial_only"
    os.makedirs(output_dir, exist_ok=True)

    # Schritt 1: Rotation entlang Hauptachse
    oriented_rect = poly.minimum_rotated_rectangle
    rect_coords = list(oriented_rect.exterior.coords)
    p0, p1 = rect_coords[0], rect_coords[1]
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

    # Prüfe, ob die Hauptachse korrekt entlang der x-Achse liegt
    minx, miny, maxx, maxy = poly_rotated.bounds
    width = maxx - minx  # Ausdehnung entlang x-Achse
    height = maxy - miny  # Ausdehnung entlang y-Achse
    if width < height:
        print("Hauptachse liegt vertikal, korrigiere um 90 Grad.")
        angle += 90.0  # Drehe um weitere 90 Grad
        poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

    # Schritt 2: Vertikale Schnitte & Mittelpunkte
    minx, miny, maxx, maxy = poly_rotated.bounds
    xs = np.arange(minx, maxx, dx)
    centers = []
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

    xs_c = [p[0] for p in centers]
    ys_c = [p[1] for p in centers]

    if len(centers) < 4:
        print(f"❌ Zu wenige Mittelpunkte ({len(centers)}) für Polynom-Anpassung.")
        return

    # Hilfsfunktion zur Berechnung des kleinsten Gesamtwinkels
    def max_turn_angle(linestring):
        coords = list(linestring.coords)
        if len(coords) < 3:
            print("Zu wenige Punkte für Winkelberechnung.")
            return 0.0

        # Berechne Tangentenvektoren am Start- und Endpunkt
        start_tangent = np.array(coords[1]) - np.array(coords[0])  # Vektor vom ersten zum zweiten Punkt
        end_tangent = np.array(coords[-1]) - np.array(coords[-2])  # Vektor vom vorletzten zum letzten Punkt

        # Prüfe, ob die Tangentenvektoren gültig sind
        norm_start = np.linalg.norm(start_tangent)
        norm_end = np.linalg.norm(end_tangent)
        if norm_start == 0 or norm_end == 0:
            print("Ungültige Tangentenvektoren (keine Länge).")
            return 0.0

        # Berechne den Winkel zwischen den Tangentenvektoren
        dot = np.dot(start_tangent, end_tangent)
        cos_theta = dot / (norm_start * norm_end)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        # Kleinster Gesamtwinkel
        smallest_angle = min(angle, 180.0 - angle)
        print(f"Gesamtwinkel zwischen Tangenten: {angle:.2f} Grad, Kleinster Winkel: {smallest_angle:.2f} Grad")

        return smallest_angle

    # Schritt 3: Für jeden gewünschten Grad
    for degree in degrees:
        try:
            print(f"→ Starte Polynom-Anpassung Grad {degree}...")
            X = np.array(xs_c).reshape(-1, 1)
            y = np.array(ys_c)

            # RANSAC mit Regularisierung (Ridge) und angepassten Parametern
            model = make_pipeline(
                PolynomialFeatures(degree=degree),
                RANSACRegressor(
                    estimator=Ridge(alpha=1.0),  # Regularisierung mit Ridge
                    residual_threshold=2.0,  # Größere Toleranz für Ausreißer
                    min_samples=0.1,  # Mindestens 10% der Punkte für Modell
                    random_state=42
                )
            )
            model.fit(X, y)
            inlier_mask = model.named_steps['ransacregressor'].inlier_mask_

            x_smooth = np.linspace(min(xs_c), max(xs_c), 300).reshape(-1, 1)
            y_smooth = model.predict(x_smooth)

            # Verstärkte Glättung
            if smooth_window > 0 and smooth_order > 0 and smooth_window > smooth_order:
                if smooth_window % 2 == 0:
                    smooth_window += 1
                y_smooth = savgol_filter(y_smooth, window_length=smooth_window, polyorder=smooth_order)
                print(f"✓ Glättung angewendet: window={smooth_window}, order={smooth_order}")
            else:
                print(f"⚠️ Glättung übersprungen (window/order ungültig)")

            # Erstelle Mittellinie und prüfe den kleinsten Gesamtwinkel
            centerline = LineString(zip(x_smooth.flatten(), y_smooth))
            centerline_rotated_back = rotate(centerline, angle, origin=poly.centroid, use_radians=False)
            max_angle = max_turn_angle(centerline_rotated_back)
            angle_threshold = 90.0  # Schwellwert für zu stark gekrümmte Linien
            print(f"Kleinster Gesamtwinkel für Grad {degree}: {max_angle:.2f} Grad")

            # Plotten
            plt.figure(figsize=(8, 6))
            plt.title(f"Mittellinie – Grad {degree} mit Glättung")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (rotiert)")
            if max_angle <= angle_threshold:  # Gültig, wenn Gesamtwinkel <= 90 Grad
                plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label=f"Grad {degree}")
            else:
                plt.text(min(x_r) + (max(x_r) - min(x_r)) / 2, min(y_r) + (max(y_r) - miny) / 2,
                         f"Mittellinie ungültig\n(Gesamtwinkel: {max_angle:.2f}° > {angle_threshold}°)",
                         fontsize=12, color='red', ha='center')
            plt.scatter(np.array(xs_c)[inlier_mask], np.array(ys_c)[inlier_mask], color='red', s=10, label="Inlier")
            plt.scatter(np.array(xs_c)[~inlier_mask], np.array(ys_c)[~inlier_mask], color='orange', s=10, label="Outlier")
            plt.axis("equal")
            plt.legend()
            output_path = os.path.join(output_dir, f"centerline_degree_{degree}.png")
            plt.savefig(output_path)
            plt.show(block=True)
            plt.close()
            print(f"✅ Plot gespeichert: {output_path}")
        except Exception as e:
            print(f"❌ Fehler bei Grad {degree}: {str(e)}")

    print("polygon_centerline_polynomial_only abgeschlossen.\n")

csv_text = """
0,"568,765 567,822 560,848 556,895 536,979 534,983 528,983 528,1023 572,1023 579,993 583,933 591,910 591,853 599,786 603,773 611,771 611,765 568,765"
1,"605,772 589,861 583,966 578,976 572,1017 568,1023 599,1023 614,963 615,948 619,940 622,918 620,879 625,852 625,821 633,798 635,772 605,772"
2,"605,765 599,814 592,832 591,849 576,903 575,921 580,964 573,987 567,988 567,1023 601,1023 615,954 622,935 625,907 625,823 628,807 636,804 636,765 605,765"
3,"93,612 93,628 115,628 143,635 227,639 246,641 264,647 323,647 354,652 499,654 512,659 521,659 521,639 472,639 463,637 462,632 465,629 456,628 452,624 449,629 445,629 444,637 375,636 353,632 266,628 254,624 208,623 188,618 152,618 151,612 93,612"
4,"701,682 701,703 756,703 866,710 993,711 1007,713 1008,719 1023,719 1023,693 864,688 863,682 701,682"
5,"720,644 720,660 751,665 951,667 1023,673 1023,655 909,652 893,650 892,644 720,644"
6,"720,624 720,647 733,644 889,647 919,651 1014,654 1023,657 1023,624 720,624"
7.0,"296,585 297,585 298,586 298,590 299,590 300,591 300,596 301,597 301,599 302,599 303,600 303,601 311,601 312,602 312,603 318,603 319,604 327,604 328,605 328,606 339,606 339,605 340,604 348,604 349,603 350,603 350,599 351,598 356,598 356,597 357,596 364,596 365,595 369,595 369,594 370,593 374,593 374,591 373,591 372,590 372,589 371,588 370,588 369,587 369,580 296,580 296,585"
7.1,"93,607 107,607 108,608 108,609 123,609 124,610 124,611 131,611 132,612 142,612 143,613 143,614 171,614 172,615 188,615 189,616 189,617 196,617 197,618 197,619 207,619 208,620 223,620 224,621 224,622 246,622 247,623 259,623 260,624 260,625 275,625 276,626 276,627 324,627 325,628 339,628 340,629 340,630 348,630 349,631 371,631 372,632 372,633 401,633 402,634 402,635 478,635 479,636 489,636 490,637 490,643 518,643 518,636 519,635 519,631 520,630 521,630 521,628 522,627 527,627 527,618 522,618 521,617 521,616 520,616 518,614 518,613 511,613 511,614 510,615 504,615 503,616 469,616 468,615 466,615 465,614 465,610 466,609 467,609 467,608 468,607 473,607 474,606 478,606 478,605 477,605 476,604 469,604 468,603 468,602 464,602 463,601 463,600 460,600 459,599 455,599 454,600 452,600 452,601 451,602 448,602 448,603 447,604 444,604 443,605 440,605 440,606 438,608 431,608 431,609 430,610 426,610 426,611 425,612 423,612 422,613 408,613 408,614 407,615 314,615 313,614 313,613 312,613 311,612 309,612 308,611 308,610 300,610 299,609 299,608 268,608 267,607 256,607 255,606 255,605 250,605 249,604 244,604 243,603 243,602 237,602 236,601 236,600 228,600 227,599 221,599 220,598 220,597 213,597 212,596 207,596 206,595 206,594 200,594 199,593 199,592 178,592 178,593 177,594 148,594 147,593 147,592 138,592 137,591 133,591 132,590 132,589 127,589 126,588 122,588 121,587 121,586 119,586 118,585 118,580 93,580 93,607"
8,"708,663 708,681 826,686 958,687 1023,694 1023,672 996,674 962,671 954,669 953,663 708,663"
9,"600,375 599,382 592,383 592,553 623,553 623,490 628,479 636,476 636,407 628,404 620,387 620,375 600,375"
10,"605,381 605,553 627,553 628,543 636,540 636,397 628,395 625,381 605,381"
11,"228,650 228,659 273,660 304,667 350,668 416,675 515,675 515,650 228,650"
12,"106,656 106,665 111,665 125,675 135,678 203,679 213,683 245,686 284,686 284,656 106,656"
13,"141,631 141,636 149,638 235,643 258,652 286,651 338,657 371,655 404,663 427,662 432,657 452,655 471,669 471,675 513,675 515,666 521,665 521,631 512,631 512,636 507,639 477,637 476,631 141,631"
14,"228,631 228,639 251,641 271,647 326,647 340,651 380,654 494,654 509,659 516,659 519,652 516,631 228,631"
15,"93,576 93,607 136,612 171,612 199,619 254,622 258,624 258,630 527,630 527,612 511,610 502,615 476,615 475,610 486,605 462,597 448,597 439,602 424,604 423,611 419,613 344,615 321,621 291,619 292,614 300,612 300,604 320,609 383,604 382,597 390,591 369,583 264,581 257,584 205,589 158,584 122,586 108,578 93,576"
16,"116,656 120,668 135,676 179,676 256,686 289,683 331,687 344,691 393,692 401,694 402,700 481,700 481,693 488,686 492,686 501,693 501,700 521,700 521,674 514,674 511,671 515,656 485,656 485,663 495,668 495,673 491,677 458,672 457,666 462,662 462,656 437,656 439,673 435,676 380,671 375,668 375,664 383,662 383,656 373,656 373,663 363,669 320,668 310,662 310,656 116,656"
17,"247,669 247,686 295,684 341,691 390,692 398,694 399,700 483,700 483,693 487,688 495,686 495,679 480,677 478,669 247,669"
18,"240,612 240,627 316,631 354,638 379,639 390,643 391,649 519,649 521,632 527,631 527,612 240,612"
"""

# Parsen der CSV-Daten zu Polygonen
polygons = []
for line in csv_text.strip().splitlines():
    parts = line.split(',', 1)
    if len(parts) < 2:
        continue
    points_str = parts[1].strip().strip('"')
    coords = []
    for p in points_str.split(' '):
        x_str, y_str = p.split(',')
        coords.append([float(x_str), float(y_str)])
    polygons.append(Polygon(coords))

if __name__ == "__main__":
    print("Ausführung von test_polynomial_only.py...")
    for idx, poly in enumerate(polygons):
        print(f"\n🔹 Verarbeite Polygon {idx + 1}...")
        polygon_centerline_polynomial_only(poly, dx=5.0, degrees=[3], smooth_window=51, smooth_order=3)
    print("Fertig.")
