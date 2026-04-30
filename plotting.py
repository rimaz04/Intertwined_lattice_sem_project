# from svgpathtools import Path, CubicBezier
import numpy as np
from scipy.interpolate import BSpline
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain
import sys

def plot_weighted_graph_on_circle(nodes, edges):
    """
    Plots nodes on a circle and draws edges with weights as labels.

    Args:
        nodes (list): List of node identifiers.
        edges (list of tuples): Each tuple is (node1, node2, weight).
    """
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    # Plot nodes
    for node, (x, y) in positions.items():
        plt.scatter(x, y, color='black', s=2000, zorder=2)
        plt.text(x, y, f"$\ell_{node+1}$", fontsize=40, ha='center', va='center', color='white', zorder=3)

    # Plot edges with weights
    for node1, node2, weight in edges:
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]
        plt.plot([x1, x2], [y1, y2], color='black', linewidth=4, zorder=1)
        # Place weight label at the center of the edge, slightly above in the normal direction
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # For specific edges, place the label at 1/4 along the edge instead of the midpoint
        if (node1, node2) == (3, 0) or (node2, node1) == (3, 0):
            mx, my = x1 + (x2 - x1) * 0.25, y1 + (y2 - y1) * 0.25
        elif (node1, node2) == (1, 4) or (node2, node1) == (1, 4):
            mx, my = x1 + (x2 - x1) * 0.25, y1 + (y2 - y1) * 0.25
        # Compute direction vector and normal
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            nx, ny = 0, 0
        else:
            nx, ny = -dy / length, dx / length  # normal vector (90 deg CCW)
        # Offset label slightly along the normal for visibility
        offset = -0.1  # adjust as needed

        label_x = mx + nx * offset
        label_y = my + ny * offset
        # Compute angle for text rotation
        angle_deg = np.degrees(np.arctan2(dy, dx))
        # Adjust angle so text is never upside down (bottom closer to bottom of figure)
        if angle_deg > 90 or angle_deg < -90:
            angle_deg += 180
        plt.text(
            label_x, label_y,
            f"$\\mathrm{{Ent}}(\ell_{node1+1},\\ell_{node2+1})={weight}$",
            fontsize=35,
            color='black',
            ha='center',
            va='center',
            rotation=angle_deg,
            rotation_mode='anchor',
            zorder=4
        )

    plt.axis('equal')
    plt.axis('off')
    plt.show()

def resample_curve_points(curve_points, num_points, equidistant=False, must_include=None):
    """
    Refines or coarsens a curve by resampling its points.

    Args:
        curve_points (list of tuple): Original curve as list of (x, y) points.
        num_points (int): Number of points in the resampled curve.
        equidistant (bool): If True, output points are spaced equally along the curve length.
        must_include (sequence of (float, float), optional): Points for which the closest discrete points
            (on the linear interpolations of the old curve) must be included in the output.

    Returns:
        list of (x, y): Resampled curve points.
    """
    curve_points = np.array(curve_points)
    if len(curve_points) < 2 or num_points < 2:
        return curve_points.tolist()

    # Compute cumulative arc length
    deltas = np.diff(curve_points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc_lengths[-1]

    # Parameter values for original points (normalized arc length)
    t_orig = arc_lengths / total_length if total_length > 0 else np.linspace(0, 1, len(curve_points))

    # --- Initial resampling based on equidistant or index-based ---
    if equidistant:
        t_base = np.linspace(0, 1, num_points)
    else:
        # For non-equidistant, we need to map indices to a 0-1 parameter space
        # This will be used for interpolation later if must_include is used
        t_base = np.linspace(0, 1, num_points) # For initial resampling, this effectively maps original indices to 0-1
        # If num_points == len(curve_points), t_base would correspond to t_orig
        # If subsampling/upsampling, this creates a uniform distribution in original index space

    x_new = np.interp(t_base, t_orig, curve_points[:, 0])
    y_new = np.interp(t_base, t_orig, curve_points[:, 1])
    resampled_initial = np.column_stack([x_new, y_new])

    # --- Handle must_include points ---
    if must_include:
        must_include = np.array(must_include)
        
        # Function to find closest point on original curve segment and its parameter 't'
        def closest_point_on_curve_and_t(pt):
            min_dist_sq = float('inf')
            closest_proj = None
            closest_t_on_orig = None # Parameter t on the original curve's length
            
            for i in range(len(curve_points) - 1):
                a, b = curve_points[i], curve_points[i + 1]
                ab = b - a
                ab_len2 = np.dot(ab, ab)
                
                if ab_len2 == 0: # Handle coincident points
                    proj = a
                    t_segment = 0.0 # Or 1.0, doesn't matter for 0 length
                else:
                    t_segment = np.clip(np.dot(pt - a, ab) / ab_len2, 0, 1)
                    proj = a + t_segment * ab
                
                dist_sq = np.sum((pt - proj)**2)
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_proj = proj
                    
                    # Calculate the 't' value (normalized arc length) for this projection
                    # t_orig[i] is the start of the segment, t_segment is its progress along the segment
                    segment_length = seg_lengths[i]
                    segment_arc_start_t = t_orig[i]
                    segment_arc_end_t = t_orig[i+1]
                    
                    # If total_length is 0, this scaling can be problematic
                    if total_length > 0:
                        closest_t_on_orig = segment_arc_start_t + t_segment * (segment_arc_end_t - segment_arc_start_t)
                    else: # If total_length is 0, all points are at the same location, use average t
                        closest_t_on_orig = (t_orig[i] + t_orig[i+1]) / 2


            return closest_proj, closest_t_on_orig

        # Collect all points to be part of the final curve
        all_t_values = list(t_base) # Start with t-values from initial resampling
        points_to_include = list(resampled_initial) # Points from initial resampling

        for pt_mi in must_include:
            proj_pt, t_proj = closest_point_on_curve_and_t(pt_mi)
            
            # Check if this projected point is already sufficiently close to an existing point
            # in the combined set (resampled_initial + previously added must_include points)
            # This avoids adding near-duplicates and helps prevent kinks if a must_include point
            # is extremely close to an already generated resampled point.
            is_duplicate = False
            for existing_pt in points_to_include:
                if np.linalg.norm(np.array(existing_pt) - proj_pt) < 1e-8: # Define a small tolerance
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_t_values.append(t_proj)
                # We don't add proj_pt to points_to_include yet, as we will re-interpolate
                # all points based on sorted t_values.

        # Sort the t_values to ensure monotonic progression along the curve
        all_t_values = np.array(sorted(list(set(all_t_values)))) # Use set to remove duplicates before sorting

        # Now, re-interpolate the curve using all_t_values and the original curve_points
        # This is where the actual resampling happens with all desired points integrated
        final_x = np.interp(all_t_values, t_orig, curve_points[:, 0])
        final_y = np.interp(all_t_values, t_orig, curve_points[:, 1])
        resampled = np.column_stack([final_x, final_y])

        # If after including must_include points, we have more than num_points, reduce them
        # This reduction must also be done carefully to maintain curve integrity.
        # A simple approach is to remove points that are closest to their neighbors,
        # ensuring must_include points (or their very close projections) are preserved.
        if len(resampled) > num_points:
            # Create a mask for points that are very close to a must_include point
            is_must_include_proxy = np.zeros(len(resampled), dtype=bool)
            for mi_pt in must_include:
                proj_pt, _ = closest_point_on_curve_and_t(mi_pt)
                # Find indices of points in resampled that are very close to proj_pt
                proxies = np.where(np.linalg.norm(resampled - proj_pt, axis=1) < 1e-8)[0]
                is_must_include_proxy[proxies] = True

            # Iteratively remove non-must_include points with smallest distances to neighbors
            while len(resampled) > num_points:
                if np.sum(~is_must_include_proxy) == 0: # All remaining are must_include, cannot remove more
                    break 
                
                # Calculate distances only for segments *not* involving must_include proxies
                dists = np.full(len(resampled) - 1, np.inf)
                for i in range(len(resampled) - 1):
                    # Only consider removing points if neither of them is a must_include proxy
                    if not is_must_include_proxy[i] and not is_must_include_proxy[i+1]:
                        dists[i] = np.linalg.norm(resampled[i+1] - resampled[i])
                
                if np.all(dists == np.inf): # All remaining points are part of must_include pairs or isolated must_include
                    # Fallback: if we can't find non-must_include segments, remove any point that isn't a must_include proxy
                    # (this might be less ideal but necessary to reach num_points)
                    non_proxy_indices = np.where(~is_must_include_proxy)[0]
                    if len(non_proxy_indices) > 0:
                        # Find the non-proxy point whose removal causes the least disturbance (e.g., shortest combined segments)
                        # This part can be complex; for simplicity, we can pick one of them, e.g., the middle one.
                        # For now, let's just pick the first non-proxy, or a smarter method later.
                        # A better approach would be to calculate the "cost" of removing each non-proxy point
                        # (e.g., how much it changes local curvature or segment length) and remove the one with min cost.
                        # For demonstration, let's find the point that has the smallest sum of distances to its neighbors
                        # among non-must_include points.
                        
                        min_total_dist = float('inf')
                        remove_idx_candidate = -1
                        for idx_to_check in non_proxy_indices:
                            current_total_dist = 0
                            if idx_to_check > 0:
                                current_total_dist += np.linalg.norm(resampled[idx_to_check] - resampled[idx_to_check-1])
                            if idx_to_check < len(resampled) - 1:
                                current_total_dist += np.linalg.norm(resampled[idx_to_check] - resampled[idx_to_check+1])
                            
                            if current_total_dist < min_total_dist:
                                min_total_dist = current_total_dist
                                remove_idx_candidate = idx_to_check
                        
                        if remove_idx_candidate != -1:
                            resampled = np.delete(resampled, remove_idx_candidate, axis=0)
                            is_must_include_proxy = np.delete(is_must_include_proxy, remove_idx_candidate)
                            continue # Continue the while loop
                    else: # No non-proxy points left, but still > num_points. This should ideally not happen
                          # if initial must_include handling is correct, but as a safeguard.
                          break # Cannot remove more points.

                remove_idx_in_dists = np.argmin(dists)
                remove_idx_actual = remove_idx_in_dists + 1 # Remove the second point of the shortest segment
                                                              # (or first point of next shortest if multiple min)
                
                resampled = np.delete(resampled, remove_idx_actual, axis=0)
                is_must_include_proxy = np.delete(is_must_include_proxy, remove_idx_actual)


        # If too few points after adding must_include and potentially cleaning up duplicates
        elif len(resampled) < num_points:
            # Re-interpolate to reach num_points, ensuring existing points are kept
            # This is essentially what the initial resampling did, but now with the added points.
            arc_lengths_resampled = np.concatenate([[0], np.cumsum(np.hypot(np.diff(resampled[:, 0]), np.diff(resampled[:, 1])))])
            total_length_resampled = arc_lengths_resampled[-1]
            t_resampled = arc_lengths_resampled / total_length_resampled if total_length_resampled > 0 else np.linspace(0, 1, len(resampled))
            
            t_final_target = np.linspace(0, 1, num_points)
            
            final_x = np.interp(t_final_target, t_resampled, resampled[:, 0])
            final_y = np.interp(t_final_target, t_resampled, resampled[:, 1])
            resampled = np.column_stack([final_x, final_y])

    else: # No must_include points, so the initial resampled_initial is the final result
        resampled = resampled_initial

    return resampled.tolist()

def de_casteljau(control_points, t):
    """
    Evaluates a Bezier curve at parameter t using De Casteljau's algorithm.
    Args:
        control_points (list of tuple): List of (x, y) tuples.
        t (float): Parameter between 0 and 1.
    Returns:
        (x, y): Point on the Bezier curve at parameter t.
    """
    points = np.array(control_points, dtype=float)
    n = len(points)
    for r in range(1, n):
        points = (1 - t) * points[:-1] + t * points[1:]
    return points[0]

def bezier_curve_points(control_points, num_points=100):
    """
    Returns points of a Bezier curve of any degree given a list of control points.
    Args:
        control_points (list of tuple): List of (x, y) tuples.
        num_points (int): Number of points to sample along the curve.
    Returns:
        list of (x, y): Points on the Bezier curve.
    """
    if len(control_points) < 2:
        raise ValueError("At least 2 control points are required for a Bezier curve.")

    ts = np.linspace(0, 1, num_points)
    curve_points = [de_casteljau(control_points, t) for t in ts]
    return curve_points

def bspline_curve_points(control_points, degree=3, num_points=100):
    """
    Returns points of a B-spline curve given a list of control points.
    Args:
        control_points (list of tuple): List of (x, y) tuples.
        degree (int): Degree of the B-spline.
        num_points (int): Number of points to sample along the curve.
    Returns:
        list of (x, y): Points on the B-spline curve.
    """
    control_points = np.array(control_points, dtype=float)
    n = len(control_points)
    if n <= degree:
        raise ValueError("Number of control points must be greater than degree.")

    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n - degree + 1),
        np.ones(degree)
    ))

    t = np.linspace(0, 1, num_points)
    spline_x = BSpline(knots, control_points[:, 0], degree)(t)
    spline_y = BSpline(knots, control_points[:, 1], degree)(t)
    return list(zip(spline_x, spline_y))

def nurbs_curve_points(control_points, weights=None, degree=3, num_points=100):
    """
    Returns points of a NURBS curve given a list of control points and weights.
    Args:
        control_points (list of tuple): List of (x, y) tuples.
        weights (list of float, optional): List of weights, same length as control_points. Defaults to all 1s.
        degree (int): Degree of the NURBS curve.
        num_points (int): Number of points to sample along the curve.
    Returns:
        list of (x, y): Points on the NURBS curve.
    """
    control_points = np.array(control_points, dtype=float)
    n = len(control_points)
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights, dtype=float)
    if n != len(weights):
        raise ValueError("Number of control points and weights must be equal.")
    if n <= degree:
        raise ValueError("Number of control points must be greater than degree.")

    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n - degree + 1),
        np.ones(degree)
    ))

    t = np.linspace(0, 1, num_points)
    weighted_ctrl_pts = control_points * weights[:, np.newaxis]
    spline_x = BSpline(knots, weighted_ctrl_pts[:, 0], degree)(t)
    spline_y = BSpline(knots, weighted_ctrl_pts[:, 1], degree)(t)
    spline_w = BSpline(knots, weights, degree)(t)
    spline_w[spline_w == 0] = 1e-8
    nurbs_x = spline_x / spline_w
    nurbs_y = spline_y / spline_w
    return list(zip(nurbs_x, nurbs_y))

def catmull_rom_curve_points(control_points, num_points=100, pad_ends=False):
    """
    Returns points of a Catmull-Rom spline through the given control points.
    Args:
        control_points (list of tuple): List of (x, y) tuples.
        num_points (int): Number of points to sample along the curve.
        pad_ends (bool): If True, pad the first and last control points so the curve passes through them.
    Returns:
        list of (x, y): Points on the Catmull-Rom spline.
    """
    control_points = np.array(control_points, dtype=float)
    n = len(control_points)
    if n < 2:
        raise ValueError("At least 2 control points are required for Catmull-Rom spline.")

    if pad_ends:
        control_points = np.vstack([control_points[0], control_points, control_points[-1]])
        n = len(control_points)
    if n < 4:
        raise ValueError("At least 4 control points are required for Catmull-Rom spline.")

    curve_points = []
    segments = n - 3
    points_per_segment = max(2, num_points // segments)
    for i in range(segments):
        p0, p1, p2, p3 = control_points[i:i+4]
        ts = np.linspace(0, 1, points_per_segment, endpoint=(i == segments - 1))
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2 * p1[0]) +
                        (-p0[0] + p2[0]) * t +
                        (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) +
                        (-p0[1] + p2[1]) * t +
                        (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            curve_points.append((x, y))
    return curve_points

def arc_points(center, radius, start_angle, end_angle, num_points=100):
    """
    Returns points along an arc of a circle.
    Args:
        center (tuple): (x, y) center of the circle.
        radius (float): Radius of the circle.
        start_angle (float): Start angle in radians.
        end_angle (float): End angle in radians.
        num_points (int): Number of points along the arc.
    Returns:
        list of (x, y): Points along the arc.
    """
    angles = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return list(zip(x, y))

def arc_between_points(center, radius, point1, point2, angle_coeff=1.0, num_points=100):
    """
    Returns points along an arc of a circle between two points, but after scaling the angle,
    the arc is centered between the original endpoints (so neither endpoint coincides with the original points).
    Args:
        center (tuple): (x, y) center of the circle.
        radius (float): Radius of the circle.
        point1 (tuple): First point on the circumference.
        point2 (tuple): Second point on the circumference.
        angle_coeff (float): Fraction of the angle to span (0 to 1, 1 means full arc between points).
        num_points (int): Number of points along the arc.
    Returns:
        list of (x, y): Points along the arc.
    """
    v1 = np.array(point1) - np.array(center)
    v2 = np.array(point2) - np.array(center)
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    # Compute smallest angle difference
    angle_diff = (angle2 - angle1) % (2 * np.pi)
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    # Center the scaled arc between the two points
    mid_angle = angle1 + angle_diff / 2
    scaled_angle = angle_diff * angle_coeff
    start_angle = mid_angle - scaled_angle / 2
    end_angle = mid_angle + scaled_angle / 2
    angles = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return list(zip(x, y))

def bezier_inner_to_outer(point0, center, radius, point1, point2, num_points=100, circum_scale=0.5):
    """
    Returns points for a cubic Bezier curve:
    - Starts at point1,
    - Second control point is the intersection of the line from center through point1 with the circle,
    - Third control point is point2 plus the circumferential direction at point2 (in the direction of point1), scaled by circum_scale,
    - Ends at point2.

    Args:
        center (tuple): (x, y) center of the circle.
        radius (float): Radius of the circle.
        point1 (tuple): Start point of the Bezier curve.
        point2 (tuple): End point of the Bezier curve.
        num_points (int): Number of points to sample along the curve.
        circum_scale (float): Scale factor for the circumferential control point near point2.

    Returns:
        list of (x, y): Points on the cubic Bezier curve.
    """
    center = np.array(center, dtype=float)
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    # Direction from center to point1
    dir_vec = p1 - point0
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        raise ValueError("point1 cannot coincide with point0")
    dir_unit = dir_vec / norm
    # Find intersection(s) of the line through point0 in direction dir_unit with the circle centered at center with given radius
    # Line: p = point0 + t * dir_unit
    # Circle: ||p - center||^2 = radius^2
    # Solve for t:
    oc = point0 - center
    a = np.dot(dir_unit, dir_unit)
    b = 2 * np.dot(oc, dir_unit)
    c = np.dot(oc, oc) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No intersection between line and circle")
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    # Choose the intersection that is aligned with point1 - point0
    t = t1 if t1 > 0 else t2
    control2 = point0 + t * dir_unit

    # Circumferential direction at point2 (perpendicular to radius at point2, in direction of point1)
    v2 = p2 - center
    v2_norm = np.linalg.norm(v2)
    if v2_norm == 0:
        raise ValueError("point2 cannot coincide with center")
    # Perpendicular vector (counterclockwise 90 deg)
    perp = np.array([-v2[1], v2[0]]) / v2_norm
    # Determine sign: should point toward point1
    if np.dot(perp, p1 - p2) < 0:
        perp = -perp
    control3 = p2 + perp * circum_scale * radius

    return bezier_curve_points([tuple(p1), tuple(control2), tuple(control3), tuple(p2)], num_points=num_points)

def composite_points(sequence, num_points=100, circum_scale=0.1, angle_coeff=0.5, type = 'bezier', loops=False):
    """
    Builds a composite curve from a sequence of (control_points, (center, radius)) tuples.
    For each tuple:
        - Adds a Bezier curve with the given control points.
        - Connects the end of the Bezier to the start of the arc using bezier_inner_to_outer.
        - Adds the arc between the last and first control points (or to the next segment).
        - Connects the end of the arc to the first control point of the next segment (or first if last) using bezier_inner_to_outer.

    Args:
        sequence (list): Each element is (control_points, (center, radius)).
        num_points (int): Number of points per segment.
        circum_scale (float): Scale for circumferential control in bezier_inner_to_outer.

    Returns:
        list of (x, y): Composite curve points.
    """
    if not sequence:
        return []

    composite_pts = []
    n = len(sequence)
    for i, (control_points, (center, radius)) in enumerate(sequence):
        if type == 'bezier':
            # Bezier curve for current control points
            bezier_pts = bezier_curve_points(control_points, num_points=num_points)
            if composite_pts:
                bezier_pts = bezier_pts[1:]  # Avoid duplicate at join
            composite_pts.extend(bezier_pts)
            point1 = control_points[-1]
            point0 = control_points[-2]
        elif type == 'catmull':
            # Catmull-Rom curve for current control points
            catmull_pts = catmull_rom_curve_points(control_points, num_points=num_points, pad_ends=True)

            point1 = control_points[-1]
            catmull_pt_prev = np.array(catmull_pts[-2])
            point1_arr = np.array(point1)
            direction = catmull_pt_prev - point1_arr
            norm = np.linalg.norm(direction)
            if norm == 0:
                point0 = point1
            else:
                point0 = tuple(point1_arr + direction / norm)

            if composite_pts:
                catmull_pts = catmull_pts[1:]  # Avoid duplicate at join
            composite_pts.extend(catmull_pts)

        if loops:
            # Prepare for arc connection to next segment
            next_idx = (i + 1) % n
            next_control_points, _ = sequence[next_idx]

            # Arc from end of current bezier to start of next
            if i < n - 1:
                arc_start = control_points[-1]
                arc_end = next_control_points[0]
            else:
                arc_start = control_points[-1]
                arc_end = sequence[0][0][0]
            arc_pts = arc_between_points(center, radius, arc_start, arc_end, num_points=num_points, angle_coeff=angle_coeff)
            # Bezier from end of bezier to start of arc
            bezier_to_arc = bezier_inner_to_outer(point0, center, radius, point1, arc_pts[0], num_points=num_points, circum_scale=circum_scale)
            bezier_to_arc = bezier_to_arc[1:]  # Avoid duplicate at join
            composite_pts.extend(bezier_to_arc)

            # Add arc points (excluding first, already included)
            if len(arc_pts) > 1:
                composite_pts.extend(arc_pts[1:])

            # Bezier from end of arc to start of next bezier
            if i < n - 1:
                next_point = next_control_points[0]
                next_next_point = next_control_points[1]
            else:
                next_point = sequence[0][0][0]
                next_next_point = sequence[0][0][1]
            bezier_from_arc = bezier_inner_to_outer(next_next_point, center, radius, next_point, arc_pts[-1], num_points=num_points, circum_scale=circum_scale)
            bezier_from_arc = bezier_from_arc[::-1][1:]  # Avoid duplicate at join, flip in reverse
            composite_pts.extend(bezier_from_arc)

    # Remove duplicate consecutive points before returning
    if composite_pts:
        unique_pts = [composite_pts[0]]
        for pt in composite_pts[1:]:
            if np.allclose(pt, unique_pts[-1], atol=1e-10):
                print(f"Duplicate consecutive point found and removed: {pt}")
            else:
                unique_pts.append(pt)
        composite_pts = unique_pts

    return composite_pts

def plot_control_points(control_points, color='red', marker='o', label='Control Points'):
    """
    Plots the control points.
    Args:
        control_points (list of tuple): List of (x, y) tuples.
        color (str): Color of the points.
        marker (str): Marker style.
        label (str): Label for the legend.
    """
    control_points = np.array(control_points)
    plt.plot(control_points[:, 0], control_points[:, 1], marker=marker, color=color, linestyle='None', label=label)

def plot_curve_points(curve_points, color='blue', linewidth=2, linestyle='-', label='Curve', highlight_points=None, offset_length=0.2, offset_width=10):
    """
    Plots the points representing a curve, optionally interrupting or highlighting at specified points.
    Args:
        curve_points (list of tuple): List of (x, y) tuples.
        color (str): Color of the curve.
        linewidth (int): Width of the curve line.
        label (str): Label for the legend.
        highlight_points (sequence of ((float, float), bool), optional): Each tuple is (point, is_drawn).
            The point is a (x, y) coordinate (not necessarily a discrete curve point).
            If is_drawn is True, the curve is drawn fully at that point with a small white offset.
            If False, the curve is interrupted briefly at that point.
    """
    ax = plt.gca()
    zorders = [artist.get_zorder() for artist in ax.get_children()]
    max_zorder = max(zorders)

    curve_points = np.array(curve_points)
    n = len(curve_points)
    if not highlight_points:
        plt.plot(curve_points[:, 0], curve_points[:, 1], color=color, linewidth=linewidth, label=label, linestyle=linestyle)
        return

    def find_closest_point_index(point, curve_points):
        curve_points_arr = np.array(curve_points)
        dists = np.linalg.norm(curve_points_arr - np.array(point), axis=1)
        return np.argmin(dists)

    highlight_with_indices = []
    for pt, is_drawn in highlight_points:
        idx = find_closest_point_index(pt, curve_points)
        highlight_with_indices.append((idx, is_drawn))

    highlight_with_indices = sorted(highlight_with_indices, key=lambda x: x[0])
    last_idx = 0

    for idx, is_drawn in highlight_with_indices:
        # At the highlight/interruption point
        idx_minus_1 = (idx - 1) % n
        idx_plus_1 = (idx + 1) % n
        p1 = curve_points[idx_minus_1]
        p2 = curve_points[idx_plus_1]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        direction = direction / norm
        offset_start = curve_points[idx] - direction * offset_length/2
        offset_end = curve_points[idx] + direction * offset_length/2
        if is_drawn:
            # Draw a black offset segment at idx+1 in the direction from idx-1 to idx+1
            plt.plot([offset_start[0], offset_end[0]], [offset_start[1], offset_end[1]], color='white', linewidth=linewidth + offset_width, zorder=max_zorder + 1)
            # Draw the colored curve segment as before
            closest_idx_end_plus_1 = (find_closest_point_index(offset_end, curve_points) + 1) % n
            plt.plot(curve_points[last_idx:idx, 0], curve_points[last_idx:idx, 1], color=color, linewidth=linewidth, linestyle=linestyle, zorder=max_zorder + 2, label=None)
            last_idx = (idx - 1) % n
        else:
            # Find the closest index in curve_points to offset_start
            closest_idx_start = find_closest_point_index(offset_start, curve_points)
            closest_idx_start_plus_1 = (closest_idx_start + 1) % n
            plt.plot(curve_points[last_idx:closest_idx_start_plus_1, 0], curve_points[last_idx:closest_idx_start_plus_1, 1], color=color, linewidth=linewidth, linestyle=linestyle, zorder=max_zorder + 2, label=None)
            last_idx = find_closest_point_index(offset_end, curve_points)
            # last_idx = (last_idx + 1) % n

    # Draw the rest of the curve
    if last_idx < n:
        plt.plot(curve_points[last_idx:, 0], curve_points[last_idx:, 1], color=color, linewidth=linewidth, linestyle=linestyle, zorder=max_zorder + 2, label=None)

def find_curve_intersections(curve1_points, curve2_points):
    """
    Finds intersection points between two curves given as lists of (x, y) points.
    Args:
        curve1_points (list of tuple): First curve as list of (x, y) points.
        curve2_points (list of tuple): Second curve as list of (x, y) points.
    Returns:
        list of (x, y): Intersection points.
    """
    line1 = LineString(curve1_points)
    line2 = LineString(curve2_points)
    intersection = line1.intersection(line2)
    if intersection.is_empty:
        return []
    if intersection.geom_type == 'Point':
        return [(intersection.x, intersection.y)]
    elif intersection.geom_type == 'MultiPoint':
        return [(pt.x, pt.y) for pt in intersection.geoms]
    elif intersection.geom_type == 'GeometryCollection':
        return [(pt.x, pt.y) for pt in intersection.geoms if pt.geom_type == 'Point']
    else:
        return []

def plot_curve_with_linking_highlights(curve, curve_linking_sequence, offset_length=0.1, offset_width=5, color='blue', linewidth=2, linestyle='-'):
    """
    Plots a curve with highlights/interruptions at intersections with other curves, according to their linking numbers.

    Args:
        curve (list of (x, y)): The main curve to plot.
        curve_linking_sequence (list of (curve_points, linking_number)): 
            Each tuple contains a curve (list of (x, y)) and an integer linking number.
        offset_length (float): Length of the highlight/interruption.
        offset_width (int): Width of the highlight.
        color (str): Color of the main curve.
        linewidth (int): Width of the main curve.
        linestyle (str): Line style of the main curve.
    """
    all_highlights = []
    for other_curve, linking_number in curve_linking_sequence:
        intersections = find_curve_intersections(curve, other_curve)
        if not intersections:
            continue
        if linking_number == 0:
            for i, pt in enumerate(intersections):
                all_highlights.append((pt, True))
            continue
        # Sort intersections along the curve
        curve_np = np.array(curve)
        def curve_param(pt):
            dists = np.linalg.norm(curve_np - pt, axis=1)
            return np.argmin(dists)
        intersections_sorted = sorted(intersections, key=curve_param)
        # If more intersections than linking_number*2, truncate
        n_crossings = len(intersections_sorted)
        n_segments = min(linking_number * 2, n_crossings)
        if n_segments == 0:
            continue
        # If not enough intersections, repeat as needed
        if n_crossings < n_segments:
            # Repeat the sequence to fill
            intersections_sorted = (intersections_sorted * ((n_segments + n_crossings - 1) // n_crossings))[:n_segments]
        else:
            intersections_sorted = intersections_sorted[:n_segments]
        # Alternate is_drawn (False, True, False, True, ...)
        for i, pt in enumerate(intersections_sorted):
            is_drawn = (i % 2 == 1)
            all_highlights.append((pt, is_drawn))
    # Plot the curve with highlights
    plot_curve_points(curve, color=color, linewidth=linewidth, linestyle=linestyle, highlight_points=all_highlights, offset_length=offset_length, offset_width=offset_width)




def find_minimum_internal_nodes(edges, start_node, end_node):
    """
    Find the minimum number of internal nodes required to link two nodes and provide the nodes.

    Args:
        edges: A dictionary where keys are edge IDs and values are tuples of two nodes.
        start_node: The starting node.
        end_node: The ending node.

    Returns:
        A tuple containing the minimum number of internal nodes and the list of nodes in the path.
    """
    # Build a graph representation from the edges
    graph = defaultdict(list)
    for edge_id, (node1, node2) in edges.items():
        graph[node1].append(node2)
        graph[node2].append(node1)

    # Perform BFS to find the shortest path
    queue = [(start_node, [start_node])]
    visited = set()

    while queue:
        current_node, path = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        # Check if we reached the end node
        if current_node == end_node:
            # Exclude the start and end nodes to count internal nodes
            internal_nodes = path[1:-1]
            return internal_nodes

        # Add neighbors to the queue
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    # If no path is found, return infinity and an empty list
    return []

def junction2ports(self):
    """
    Find all ports connected to each junction.

    Returns:
        A dictionary mapping each junction to a list of connected ports.
    """
    junction_to_ports = defaultdict(list)
    for edge_id, (n1, n2) in self.edges.items():
        if n1 in self.junctions:
            other_node = n2
            if other_node in self.ports:
                junction_to_ports[n1].append(other_node)
        if n2 in self.junctions:
            other_node = n1
            if other_node in self.ports:
                junction_to_ports[n2].append(other_node)
    return dict(junction_to_ports)

def plot_solution(self, solution):
    """
    Plot a 2D representation of a solution, where nodes are placed on circumferences
    and edges are drawn based on the paths in the solution. Each node (port or junction)
    has sub-nodes (strands) equal to self.N times the number of edges involving that node.

    Args:
        solution: A set of edges representing a path and their multiplicities.
    """
    import matplotlib.pyplot as plt

    # Separate ports and junctions
    all_nodes = set(chain(*[edge for edge in self.edges.values()]))
    junctions = all_nodes - set(self.ports)

    # Compute positions for ports and junctions
    num_ports = len(self.ports)
    num_junctions = len(junctions)
    if num_junctions == 1:
        junction_positions = {
            next(iter(junctions)): (0, 0)  # Place the single junction at the center
        }
    else:
        junction_positions = {
            junction: (0.5 * np.cos(2 * np.pi * i / num_junctions), 0.5 * np.sin(2 * np.pi * i / num_junctions))
            for i, junction in enumerate(junctions)
        }
    port_positions = {}
    i = 0
    for j, junction in enumerate(junctions):
        print(self.junction2ports()[junction])
        # i = 0
        for _, port in enumerate(self.junction2ports()[junction]):
            if self.counterports[port] not in port_positions.keys():
                angle = 2 * np.pi * (i / num_ports)
                i += 1
                # angle = angle - np.pi / (num_ports / len(junctions) -1)
                port_positions[port] = (np.cos(angle), np.sin(angle))
    if num_junctions == 1:
        # Symmetric with respect to the center
        counterport_positions = {
        self.counterports[port]: (-x, -y) for port, (x, y) in port_positions.items()
        }
    else:
        # Symmetric with respect to the line from the center to the junction or the center
        counterport_positions = {}
        for port, (x, y) in port_positions.items():
            linked_junctions = self.find_minimum_internal_nodes(self.edges, port, self.counterports[port])
            if len(linked_junctions) == 1:
                # Symmetric with respect to the line to the linked junction
                junction = linked_junctions[0]
                junction_x, junction_y = junction_positions[junction]
                counterport_positions[self.counterports[port]] = (x, -y)
            else:
                # Symmetric with respect to the center
                counterport_positions[self.counterports[port]] = (-x, -y)
    positions = {**port_positions, **counterport_positions, **junction_positions}

    # Compute sub-node positions for ports and junctions
    subnode_positions = {}
    for node in all_nodes:
        num_edges = sum(1 for edge in self.edges.values() if node in edge)
        if node in self.ports:
            num_subnodes = self.N * num_edges
            base_x, base_y = positions[node]
            radius = 0.1  # Outer circumference for ports
            if num_subnodes == 1:
                # If only one subnode, its position coincides with the node
                subnode_positions[node] = [(base_x, base_y)]
            else:
                # Place sub-nodes symmetrically on a tangent line at the port location
                tangent_length = 0.2  # Total length of the tangent segment
                angle = np.arctan2(base_y, base_x)  # Angle of the port position
                tangent_dx = -np.sin(angle) * tangent_length / (num_subnodes - 1)
                tangent_dy = np.cos(angle) * tangent_length / (num_subnodes - 1)
                subnode_positions[node] = [
                    (
                    base_x + (i - (num_subnodes - 1) / 2) * tangent_dx,
                    base_y + (i - (num_subnodes - 1) / 2) * tangent_dy
                    )
                    for i in range(num_subnodes)
                ]
        else:
            num_subnodes = self.N * num_edges // 2
            base_x, base_y = positions[node]
            radius = 0.25  # Inner circumference for junctions
            angle_offset = np.pi / num_subnodes / 2 + np.arctan2(base_y, base_x) + np.pi / len(junctions)  # Orient opposite with respect to the center
            angle_section = 2 * np.pi / len(junctions)  # Section of the circle based on N
            subnode_positions[node] = [
                (
                base_x + radius * np.cos(angle_offset + angle_section * (j / num_subnodes)),
                base_y + radius * np.sin(angle_offset + angle_section * (j / num_subnodes))
                )
                for j in range(num_subnodes)
            ]

    # Plot nodes and sub-nodes
    plt.figure(figsize=(4, 4))
    for node, (x, y) in positions.items():
        if node in self.ports and node < num_ports // 2:
            color = 'red'
        elif node in self.counterports.values():
            color = 'blue'
        else:
            color = 'green'
        plt.scatter(x, y, color=color, s=100, zorder=2)
        plt.text(x, y, str(node), fontsize=10, ha='center', va='center', zorder=3)

        # Plot sub-nodes
        for sx, sy in subnode_positions[node]:
            plt.scatter(sx, sy, color='grey', s=20, zorder=1)
    
    # Plot edges using sub-nodes
    used_subnodes = {node: [0] * len(subnodes) for node, subnodes in subnode_positions.items()}
    for path, multiplicity in list(self.solutions)[solution]:
        for _ in range(multiplicity):
            for e in path:
                n1, n2 = self.edges[e]
                subnodes1 = subnode_positions[n1]
                subnodes2 = subnode_positions[n2]

                # Find the first available sub-nodes for each node
                i1 = next(i for i, count in enumerate(used_subnodes[n1]) if count < (2 if n1 not in self.ports else 1))
                i2 = next(i for i, count in enumerate(used_subnodes[n2]) if count < (2 if n2 not in self.ports else 1))

                # Mark these sub-nodes as used
                used_subnodes[n1][i1] += 1
                used_subnodes[n2][i2] += 1

                # Connect the selected sub-nodes
                x1, y1 = subnodes1[i1]
                x2, y2 = subnodes2[i2]
                plt.plot([x1, x2], [y1, y2], color='black', alpha=0.7, zorder=0)
    
    # Plot outer circumference
    outer_radius = 1
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(outer_radius * np.cos(theta), outer_radius * np.sin(theta), color='black', linestyle='--', alpha=0.5)

    plt.axis('equal')
    plt.title("Solution Visualization with Sub-Nodes")
    plt.show()

if __name__ == "__main__":
    nodes = {0, 1, 2, 3, 4, 5}
    edges = [(0,1,1), (1,2,1), (2,3,1), (3,0,1), (1,4,1)]
    plot_weighted_graph_on_circle(nodes, edges)
    sys.exit()

    cp1 = [(0, 0), (2, 2), (4, 0)]
    cp2 = [(2, 0), (0, 2), (2, 4)]
    cp3 = [(-2,1), (3,-2), (1,3)]
    cp4 = [(-2,4), (-1,2), (-2,0)]
    center = (1, 1.2)
    s1 = composite_points([(cp1, (center, 4))], type='catmull')
    s2 = composite_points([(cp2, (center, 5)), (cp4, (center, 5))], type='catmull')
    # s2cat = composite_points([(cp2, (center, 4))], type='catmull')
    s3 = composite_points([(cp3, (center, 4))], type='catmull')

    plt.figure(figsize=(8, 8))
    plot_curve_points(s2, color='red', linewidth=2, linestyle='-', label='Curve 2')
    # plot_curve_points(s2cat, color='blue', linewidth=2, linestyle='-', label='Curve 2 Catmull')
    # i12 = find_curve_intersections(s1, s2)
    # print(i12)
    # s1 = resample_curve_points(s1,250, must_include=i12)
    # plot_curve_with_linking_highlights(s1, [(s2, 1)])
    # i13 = find_curve_intersections(s1, s3)
    # i23 = find_curve_intersections(s2, s3)
    # s3 = resample_curve_points(s3, 250, must_include=i13+i23)
    # plot_curve_with_linking_highlights(s3, [(s2, 1), (s1, 0)], color='green')
    plt.show()

    # # Plot the curves
    # plt.figure(figsize=(8, 8))
    # plot_curve_points(s1, color='green', linewidth=2, linestyle='-', label='Curve 1')
    # plot_control_points(cp1, color='green', marker='o', label='Control Points 1')
    # plot_curve_points(s2, color='red', linewidth=2, linestyle='-', label='Curve 2')
    # plot_control_points(cp2, color='red', marker='o', label='Control Points 2')
    # plt.scatter(center[0], center[1], color='blue', s=80, marker='x', label='Center')
    # plt.show()
    