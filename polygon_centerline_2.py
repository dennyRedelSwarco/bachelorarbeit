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

def polygon_centerline_polynomial_only(poly: Polygon, dx=5.0, degree=3, smooth_window=19, smooth_order=1, min_x=0.0, min_y=0.0, scale=1.0, lower_corner_wgs84=None, upper_corner_wgs84=None, show_plots=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting polygon_centerline_polynomial_only for degree 3...")
    print("Starting polygon_centerline_polynomial_only for degree 3...")

    output_dir = "output_polynomial_degree3"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory '{output_dir}' created or exists.")
    print(f"Output directory '{output_dir}' created or exists.")

    try:
        # Step 1: Rotation along main axis
        logging.info("Step 1: Calculating rotation along main axis...")
        print("Step 1: Calculating rotation along main axis...")
        oriented_rect = poly.minimum_rotated_rectangle
        rect_coords = list(oriented_rect.exterior.coords)
        p0, p1 = rect_coords[0], rect_coords[1]
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

        minx, miny, maxx, maxy = poly_rotated.bounds
        width = maxx - minx
        height = maxy - miny
        if width < height:
            logging.info("Main axis is vertical, correcting by 90 degrees.")
            print("Main axis is vertical, correcting by 90 degrees.")
            angle += 90.0
            poly_rotated = rotate(poly, -angle, origin='centroid', use_radians=False)

        logging.info(f"Polygon rotated by {angle:.2f} degrees.")
        print(f"Polygon rotated by {angle:.2f} degrees.")

        if show_plots:
            logging.info("Creating plot for Step 1: Rotation...")
            print("Creating plot for Step 1: Rotation...")
            plt.figure(figsize=(8, 6))
            plt.title("Step 1: Rotation along Main Axis")
            x1, y1 = poly.exterior.xy
            x2, y2 = poly_rotated.exterior.xy
            plt.plot(x1, y1, label="Original Polygon", linestyle='--', alpha=0.5)
            plt.plot(x2, y2, label="Rotated Polygon", color='blue')
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step1_rotation.png"))
            logging.info("Plot for Step 1 saved: output_polynomial_degree3/step1_rotation.png")
            print("Plot for Step 1 saved: output_polynomial_degree3/step1_rotation.png")
            plt.close()

        # Step 2: Vertical cuts & midpoints
        logging.info("Step 2: Calculating cross-section midpoints...")
        print("Step 2: Calculating cross-section midpoints...")
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
        logging.info(f"Step 2: Found {len(centers)} midpoints.")
        print(f"Step 2: Found {len(centers)} midpoints.")

        if show_plots:
            logging.info("Creating plot for Step 2: Vertical Cuts...")
            print("Creating plot for Step 2: Vertical Cuts...")
            plt.figure(figsize=(8, 6))
            plt.title("Step 2: Vertical Cross-Sections & Midpoints (Rotated)")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (Rotated)")
            for line in cutlines:
                x_cut, y_cut = line.xy
                plt.plot(x_cut, y_cut, color='gray', linewidth=0.5, alpha=0.5)
            plt.scatter(xs_c, ys_c, color='red', label="Midpoints")
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step2_vertical_cuts.png"))
            logging.info("Plot for Step 2 saved: output_polynomial_degree3/step2_vertical_cuts.png")
            print("Plot for Step 2 saved: output_polynomial_degree3/step2_vertical_cuts.png")
            plt.close()

        if len(centers) < 4:
            logging.error(f"Too few midpoints ({len(centers)}) for polynomial fitting.")
            print(f"❌ Too few midpoints ({len(centers)}) for polynomial fitting.")
            return None, 0.0, 0.0, []

        # Step 3: Polynomial fitting (degree 3 only)
        logging.info("Step 3: Performing polynomial fitting for degree 3...")
        print("Step 3: Performing polynomial fitting for degree 3...")
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
        logging.info(f"Degree 3: {np.sum(inlier_mask)} of {len(centers)} points detected as inliers.")
        print(f"Degree 3: {np.sum(inlier_mask)} of {len(centers)} points detected as inliers.")

        x_smooth = np.linspace(min(xs_c), max(xs_c), 300).reshape(-1, 1)
        y_smooth = model.predict(x_smooth)

        if smooth_window > 0 and smooth_order > 0 and smooth_window > smooth_order:
            if smooth_window % 2 == 0:
                smooth_window += 1
            y_smooth = savgol_filter(y_smooth, window_length=smooth_window, polyorder=smooth_order)
            logging.info(f"Smoothing applied: window={smooth_window}, order={smooth_order}")
            print(f"✓ Smoothing applied: window={smooth_window}, order={smooth_order}")
        else:
            logging.info(f"Smoothing skipped (invalid window/order)")
            print(f"⚠️ Smoothing skipped (invalid window/order)")

        centerline = LineString(zip(x_smooth.flatten(), y_smooth))

        if show_plots:
            logging.info("Creating plot for Step 3: Polynomial Fitting...")
            print("Creating plot for Step 3: Polynomial Fitting...")
            plt.figure(figsize=(8, 6))
            plt.title("Step 3: Polynomial Centerline (Degree 3, Rotated)")
            x_r, y_r = poly_rotated.exterior.xy
            plt.plot(x_r, y_r, color='lightgray', label="Polygon (Rotated)")
            plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label="Centerline (Degree 3)")
            plt.scatter(np.array(xs_c)[inlier_mask], np.array(ys_c)[inlier_mask], color='red', s=10, label="Inlier")
            plt.scatter(np.array(xs_c)[~inlier_mask], np.array(ys_c)[~inlier_mask], color='orange', s=10, label="Outlier")
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step3_centerline_degree3.png"))
            logging.info("Plot for Step 3 saved: output_polynomial_degree3/step3_centerline_degree3.png")
            print("Plot for Step 3 saved: output_polynomial_degree3/step3_centerline_degree3.png")
            plt.close()

        # Step 4: Rotate centerline back
        logging.info("Step 4: Rotating centerline back to original orientation...")
        print("Step 4: Rotating centerline back to original orientation...")
        centerline_rotated_back = rotate(centerline, angle, origin=poly.centroid, use_radians=False)

        def max_turn_angle(linestring):
            coords = list(linestring.coords)
            if len(coords) < 3:
                logging.info("Too few points for angle calculation.")
                print("Too few points for angle calculation.")
                return 0.0

            start_tangent = np.array(coords[1]) - np.array(coords[0])
            end_tangent = np.array(coords[-1]) - np.array(coords[-2])
            norm_start = np.linalg.norm(start_tangent)
            norm_end = np.linalg.norm(end_tangent)
            if norm_start == 0 or norm_end == 0:
                logging.info("Invalid tangent vectors (zero length).")
                print("Invalid tangent vectors (zero length).")
                return 0.0

            dot = np.dot(start_tangent, end_tangent)
            cos_theta = dot / (norm_start * norm_end)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
            smallest_angle = min(angle, 180.0 - angle)
            logging.info(f"Total angle between tangents: {angle:.2f} degrees, Smallest angle: {smallest_angle:.2f} degrees")
            print(f"Total angle between tangents: {angle:.2f} degrees, Smallest angle: {smallest_angle:.2f} degrees")
            return smallest_angle

        max_angle = max_turn_angle(centerline_rotated_back)
        angle_threshold = 90.0
        logging.info(f"Smallest total angle for degree 3: {max_angle:.2f} degrees")
        print(f"Smallest total angle for degree 3: {max_angle:.2f} degrees")

        if show_plots:
            logging.info("Creating plot for Step 4: Back-Rotated Centerline...")
            print("Creating plot for Step 4: Back-Rotated Centerline...")
            plt.figure(figsize=(8, 6))
            plt.title("Step 4: Centerline (Degree 3, Back-Rotated)")
            x_orig, y_orig = poly.exterior.xy
            plt.plot(x_orig, y_orig, color='lightgray', label="Original Polygon")
            x_cl, y_cl = centerline_rotated_back.xy
            if max_angle <= angle_threshold:
                plt.plot(x_cl, y_cl, color='green', linewidth=2, label="Centerline (Degree 3)")
            else:
                plt.text(min(x_orig) + (max(x_orig) - min(x_orig)) / 2, min(y_orig) + (max(y_orig) - min(y_orig)) / 2,
                         f"Centerline invalid\n(Total angle: {max_angle:.2f}° > {angle_threshold}°)",
                         fontsize=12, color='red', ha='center')
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step4_centerline_back_rotated.png"))
            logging.info("Plot for Step 4 saved: output_polynomial_degree3/step4_centerline_back_rotated.png")
            print("Plot for Step 4 saved: output_polynomial_degree3/step4_centerline_back_rotated.png")
            plt.close()

        # Step 5: Width calculation and geocoordinates
        logging.info("Step 5: Calculating roadway widths and geocoordinates...")
        print("Step 5: Calculating roadway widths and geocoordinates...")
        if lower_corner_wgs84 is None:
            lower_corner_wgs84 = (13.0, 52.0)
        if upper_corner_wgs84 is None:
            upper_corner_wgs84 = (13.1, 52.1)

        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        utm_min_x, utm_min_y = transformer_to_utm.transform(lower_corner_wgs84[0], lower_corner_wgs84[1])
        utm_max_x, utm_max_y = transformer_to_utm.transform(upper_corner_wgs84[0], upper_corner_wgs84[1])
        utm_distance = np.sqrt((utm_max_x - utm_min_x)**2 + (utm_max_y - utm_min_y)**2)
        pixel_distance = np.sqrt(1024**2 + 1024**2)
        cm_per_pixel = (utm_distance * 100) / pixel_distance
        logging.info(f"Scaling: {cm_per_pixel:.6f} centimeters per pixel")
        print(f"Scaling: {cm_per_pixel:.6f} centimeters per pixel")

        widths = []
        width_lines = []
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
                width_pixels = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
                width_cm = width_pixels * cm_per_pixel
                widths.append(width_cm)
                width_lines.append((cutline_final, points, width_cm))
        avg_width = np.mean(widths) if widths else 0.0
        logging.info(f"Average width for degree 3: {avg_width:.2f} cm")
        print(f"Average width for degree 3: {avg_width:.2f} cm")

        if show_plots:
            logging.info("Creating plot for Step 5: Roadway Width Calculation...")
            print("Creating plot for Step 5: Roadway Width Calculation...")
            plt.figure(figsize=(8, 6))
            plt.title("Step 5: Roadway Width Calculation with Perpendicular Cuts")
            x_orig, y_orig = poly.exterior.xy
            plt.plot(x_orig, y_orig, color='lightgray', label="Original Polygon")
            x_cl, y_cl = centerline_rotated_back.xy
            plt.plot(x_cl, y_cl, color='green', linewidth=2, label="Centerline (Degree 3)")
            # Zoom in by setting limits based on polygon bounds
            minx, miny, maxx, maxy = poly.bounds
            x_margin = (maxx - minx) * 0.1  # 10% margin
            y_margin = (maxy - miny) * 0.1
            plt.xlim(minx - x_margin, maxx + x_margin)
            plt.ylim(miny - y_margin, maxy + y_margin)
            # Plot every 5th cutline, clipped to polygon bounds
            for idx, (cutline, points, width_cm) in enumerate(width_lines[::5]):
                # Clip cutline to plot bounds for visualization
                cutline_clipped = cutline.intersection(Polygon([
                    (minx - x_margin, miny - y_margin),
                    (maxx + x_margin, miny - y_margin),
                    (maxx + x_margin, maxy + y_margin),
                    (minx - x_margin, maxy + y_margin)
                ]))
                if cutline_clipped.is_empty:
                    continue
                if cutline_clipped.geom_type == 'LineString':
                    x_cut, y_cut = cutline_clipped.xy
                    plt.plot(x_cut, y_cut, color='purple', linewidth=0.5, alpha=0.7, label="Perpendicular Cuts" if idx == 0 else "")
                plt.scatter([p[0] for p in points], [p[1] for p in points], color='purple', s=30, label="Intersection Points" if idx == 0 else "")
            plt.axis("equal")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "step5_roadway_width.png"))
            logging.info("Plot for Step 5 saved: output_polynomial_degree3/step5_roadway_width.png")
            print("Plot for Step 5 saved: output_polynomial_degree3/step5_roadway_width.png")
            plt.close()

        if max_angle > angle_threshold:
            logging.error("Centerline invalid due to excessive angle.")
            print("❌ Centerline invalid due to excessive angle.")
            return None, 0.0, 0.0, []

        # Geocoordinates
        distances = np.linspace(0, centerline_rotated_back.length, 10)
        sampled_points = [centerline_rotated_back.interpolate(d) for d in distances]
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
                "geocoordinate": f"({lon},{lat})",
                "avg_width": avg_width
            })
            logging.info(f"Point {i+1}: ({lon:.6f}, {lat:.6f})")
            print(f"Point {i+1}: ({lon:.6f}, {lat:.6f})")

        # Save coordinates to CSV
        centerline_df = pd.DataFrame(centerline_coords)
        csv_path = os.path.join(output_dir, "centerline_coords.csv")
        centerline_df.to_csv(csv_path, index=False)
        logging.info(f"Centerline coordinates saved: {csv_path}")
        print(f"Centerline coordinates saved: {csv_path}")

        # Calculate AVG(x+y)
        avg_xy = np.mean([x + y for x, y in centers]) if centers else 0.0
        logging.info(f"AVG(x+y): {avg_xy:.2f}")
        print(f"AVG(x+y): {avg_xy:.2f}")

        logging.info("polygon_centerline_polynomial_only completed.")
        print("polygon_centerline_polynomial_only completed.")
        return centerline_rotated_back, avg_width, avg_xy, centerline_coords

    except Exception as e:
        logging.error(f"Error in polygon_centerline_polynomial_only: {str(e)}")
        print(f"❌ Error in polygon_centerline_polynomial_only: {str(e)}")
        return None, 0.0, 0.0, []

# Example usage
if __name__ == "__main__":
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
    best_centerline, avg_width, avg_xy, centerline_coords = polygon_centerline_polynomial_only(
        poly=polygon,
        dx=5.0,
        degree=3,
        smooth_window=19,
        smooth_order=1,
        min_x=0.0,
        min_y=0.0,
        scale=1.0,
        lower_corner_wgs84=(13.0, 52.0),
        upper_corner_wgs84=(13.1, 52.1),
        show_plots=True
    )