import numpy as np
from collections import defaultdict
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from symmetry import *
from topology2 import *
from export import *
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping
import matplotlib.animation as animation
import vedo
import pickle
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio
import matplotlib
from matplotlib import cm



def update_plot(frame, frames, params_original, optimizer, point_tangents, ax, scatter, cycles):
    # Update params dynamically
    params = params_original.copy()
    frames_cycle = frames // cycles
    frame_cycle = frame % frames_cycle
    param_index = frame_cycle // (frames_cycle // len(params))
    sub_frames = frames_cycle // len(params)
    if param_index < len(params):
        if param_index > 2 and param_index % 2 == 1:
            params[param_index] += 2*np.pi * np.sin(0.5 * np.pi * frame_cycle / sub_frames)  # Vary the parameter sinusoidally
        else:
            params[param_index] *= (1 - np.sin(2 * np.pi * frame_cycle / sub_frames))  # Vary the parameter sinusoidally

    # Clear the previous plot
    ax.cla()

    # Generate the updated Bezier curve
    bezier, control_points = optimizer.control2Bezier(point_tangents, params, n=15)
    bezier.plot(ax, color='b', linewidth=2)

    # Plot control points
    control_points = np.array(control_points)
    scatter._offsets3d = (control_points[:, 0], control_points[:, 1], control_points[:, 2])
    ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], color='r', label='Control Points')

    # Add arrows from first to second control point and from last to second last control point
    ax.quiver(control_points[0, 0], control_points[0, 1], control_points[0, 2],
            control_points[1, 0] - control_points[0, 0],
            control_points[1, 1] - control_points[0, 1],
            control_points[1, 2] - control_points[0, 2],
            color='k', arrow_length_ratio=0.1, linewidth=1, label='First to Second')

    ax.quiver(control_points[-1, 0], control_points[-1, 1], control_points[-1, 2],
            control_points[-2, 0] - control_points[-1, 0],
            control_points[-2, 1] - control_points[-1, 1],
            control_points[-2, 2] - control_points[-1, 2],
            color='k', arrow_length_ratio=0.1, linewidth=1, label='Last to Second Last')

    # Plot the line connecting the first and last control points
    ax.plot([control_points[0, 0], control_points[-1, 0]], 
            [control_points[0, 1], control_points[-1, 1]], 
            [control_points[0, 2], control_points[-1, 2]], 
            color='grey', linestyle='--')
            

    # Add intermediate points and quivers
    p1 = control_points[0]
    p2 = control_points[-1]
    for i in range(len(params)//2-1):
        p3 = p1 + (i+1) * (p2 - p1) / (len(params)//2)
        ax.scatter(p3[0], p3[1], p3[2], color='grey')
        ax.quiver(p3[0], p3[1], p3[2],
                    control_points[2+i][0] - p3[0],
                    control_points[2+i][1] - p3[1],
                    control_points[2+i][2] - p3[2],
                    color='k', arrow_length_ratio=0.1)

    # Rotate the point of view
    ax.view_init(elev=15, azim=(frame * 360 / frames) % 360 + 270)

    # ax.legend()
    ax.set_xlim([-0., 1])
    ax.set_ylim([-0., 1])
    ax.set_zlim([-0., 1])
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # # Initialize parameters for each subplot
    # point_tangents = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, -1, 0]])
    # params_list = [[0.5, 0.5], [0.5, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0, 0.5, 0]]
    # cycles_list = [6, 3, 2]

    # # Set up the figure and 3D axes
    # fig = plt.figure(figsize=(15, 5))
    # axes = [fig.add_subplot(131, projection='3d'),
    #         fig.add_subplot(132, projection='3d'),
    #         fig.add_subplot(133, projection='3d')]

    # # Initialize scatter objects for each subplot
    # scatters = [ax.scatter([], [], [], color='r', label='Control Points') for ax in axes]

    # # Create the animation
    # frames = 600
    # ani = animation.FuncAnimation(
    #     fig,
    #     lambda frame: [update_plot(frame, frames, params_list[i], Optimizer(point_tangents, 0.1, min_degree=3, max_degree=3, fixed_curves=[], fixed_radii=[], sequential=None), point_tangents, axes[i], scatters[i], cycles_list[i]) for i in range(3)],
    #     frames=frames,
    #     interval=100
    # )

    # # Save the animation as a GIF
    # ani.save('animation.gif', writer='pillow', fps=30)




def plot(params, point_tangents):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    optimizer = Optimizer(point_tangents, 0.1, min_degree=3, max_degree=3, fixed_curves=[], fixed_radii=[], sequential=None)
    bezier, control_points = optimizer.control2Bezier(point_tangents, params, n=15)

    # Plot the Bezier curve
    bezier.plot(ax, color='b', linewidth=2)

    # Plot control points
    control_points = np.array(control_points)
    ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], color='r', label='Control Points')

    # Add arrows from first to second control point and from last to second last control point
    ax.quiver(control_points[0, 0], control_points[0, 1], control_points[0, 2],
                control_points[1, 0] - control_points[0, 0],
                control_points[1, 1] - control_points[0, 1],
                control_points[1, 2] - control_points[0, 2],
                color='k', arrow_length_ratio=0.1, linewidth=1, label='First to Second')

    ax.quiver(control_points[-1, 0], control_points[-1, 1], control_points[-1, 2],
                control_points[-2, 0] - control_points[-1, 0],
                control_points[-2, 1] - control_points[-1, 1],
                control_points[-2, 2] - control_points[-1, 2],
                color='k', arrow_length_ratio=0.1, linewidth=1, label='Last to Second Last')

    # Plot the line connecting the first and last control points
    ax.plot([control_points[0, 0], control_points[-1, 0]], 
            [control_points[0, 1], control_points[-1, 1]], 
            [control_points[0, 2], control_points[-1, 2]], 
            color='grey', linestyle='--')

    # Add intermediate points and quivers
    p1 = control_points[0]
    p2 = control_points[-1]
    for i in range(len(params)//2-1):
        p3 = p1 + (i+1) * (p2 - p1) / (len(params)//2)
        ax.scatter(p3[0], p3[1], p3[2], color='grey')
        ax.quiver(p3[0], p3[1], p3[2],
                    control_points[2+i][0] - p3[0],
                    control_points[2+i][1] - p3[1],
                    control_points[2+i][2] - p3[2],
                    color='k', arrow_length_ratio=0.1)

    ax.set_xlim([-0., 1])
    ax.set_ylim([-0., 1])
    ax.set_zlim([-0., 1])
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    plt.show()

    # # Example usage
    # params = [1, 1, 1, 0, 1, 0]
    # p1= np.array([0, 0, 0])
    # p2= np.array([1, 1, 1])
    # t1= np.array([1, 0, 0])
    # t2= np.array([-1, 0, 0])
    # t1 = t1/np.linalg.norm(t1)
    # t2 = t2/np.linalg.norm(t2)
    # point_tangents = np.array([p1, t1, p2, t2])
    # plot(params, point_tangents)





def segment_distance(P1, Q1, P2, Q2):
    """Compute the shortest distance between two line segments in 3D."""
    def clamp(x, lower, upper):
        return max(lower, min(x, upper))

    d1 = Q1 - P1  # Direction vector of segment 1
    d2 = Q2 - P2  # Direction vector of segment 2
    r = P1 - P2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)

    denom = a * c - b * b  # Determinant of the coefficient matrix
    if denom > 1e-10:  # Avoid division by zero for nearly parallel lines
        t = clamp((b * e - c * d) / denom, 0, 1)
        s = clamp((a * e - b * d) / denom, 0, 1)
    else:  # Lines are nearly parallel
        t = 0
        s = clamp(e / c if c > 0 else 0, 0, 1)

    # Compute the closest points and return the distance
    return np.linalg.norm((P1 + t * d1) - (P2 + s * d2))





class Optimizer:

    def __init__(self, points_tangents, radii, min_degree=3, max_degree=3, fixed_helices=None, max_initial_guess=100, sequential=None):
        self.points_tangents = list(points_tangents)
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.max_initial_guess = max_initial_guess
        self.radii = radii
        if sequential is None or len(points_tangents) % sequential == 0:
            self.sequential = sequential
        else:
            raise ValueError("The number of control points must be divisible by the sequential value.")
        self.optimized_beziers = []
        self.fixed_helices = fixed_helices if fixed_helices is not None else []

    def optimize(self, method='SLSQP', options=None):
        if self.sequential is None:
            # Optimize all curves simultaneously
            for degree in range(self.min_degree, self.max_degree + 1):
                bounds = self.bounds(degree, self.points_tangents)
                args = (tuple(range(len(self.points_tangents))),)
                x0 = self.initial_guess(degree, bounds)

                result = self._run_optimizer(method, self.objective_function, x0, bounds, args, options)

                if self.objective_function(result.x, args, badness_type=['contact']) < 1e-12:
                    self._store_optimized_curves(result.x, len(self.points_tangents))
                    break
        else:
            # Optimize sequentially in groups
            for i in range(len(self.points_tangents) // self.sequential):
                start, end = i * self.sequential, (i + 1) * self.sequential
                points_tangents = self.points_tangents[start:end]
                print(f"Optimizing curves {start} to {end-1}...")
                
                success = False
                for degree in range(self.min_degree, self.max_degree + 1):
                    print(f"Degree: {degree}")
                    bounds = self.bounds(degree, points_tangents)
                    args = (tuple(range(start, end)),)
                    x0 = self.initial_guess(degree, bounds)

                    i = 0
                    while not success:
                        if i < 1:
                            method = 'SLSQP'
                            options['maxiter'] = 1
                        elif i > 1:
                            break
                        else:
                            method = 'dual_annealing'
                            options['maxiter'] = 50
                        i += 1
                        result = self._run_optimizer(method, self.objective_function, x0, bounds, args, options)
                        print(result.success, self.objective_function(result.x, args[0], badness_type=['curvature']), self.objective_function(result.x, args[0], badness_type=['contact']))
                        success = self.objective_function(result.x, args[0], badness_type=['contact']) < 1e-12
                        # The following line enforces a stricter condition on curvature
                        # if success:
                        #     success = self.objective_function(result.x, args[0], badness_type=['curvature']) < 10
                        # success = True
                        if degree == 5 and method == 'dual_annealing':
                            success = True
                    if success:
                        self._store_optimized_curves(result.x, len(points_tangents), start)
                        break

        return result

    def _run_optimizer(self, method, objective_function, x0, bounds, args, options):
        if method == 'SLSQP':
            return minimize(objective_function, x0=x0, args=args, method='SLSQP', bounds=bounds, options=options)
        elif method == 'basinhopping':
            minimizer_kwargs = {'method': 'L-BFGS-B', 'args': args, 'bounds': bounds, 'options': options}
            return basinhopping(objective_function, x0=x0, minimizer_kwargs=minimizer_kwargs, niter=options['maxiter'])
        elif method == 'differential_evolution':
            return differential_evolution(objective_function, bounds=bounds, args=args, strategy='best1bin', maxiter=options['maxiter'], popsize=15, tol=1e-6)
        elif method == 'dual_annealing':
            return dual_annealing(objective_function, bounds=bounds, args=args, maxiter=options['maxiter'])
        else:
            raise ValueError(f"Unknown method: {method}")


    def _store_optimized_curves(self, x, num_curves, start_index=0):
        n_param = len(x) // num_curves
        for curve in range(num_curves):
            params = x[curve * n_param:(curve + 1) * n_param]
            self.optimized_beziers.append(Bezier(points_tangents=self.points_tangents[start_index + curve], params=params, points_per_curve=15, radius=self.radii[start_index + curve]))
            

    # def optimize(self, method='SLSQP', options=None):
    #     if self.sequential == None:
    #         # Optimize all the curves simultaneously
    #         for degree in range(self.min_degree, self.max_degree + 1):
    #             result = minimize(
    #                 self.objective_function,
    #                 args=(tuple(range(len(self.points_tangents))),),
    #                 x0=self.initial_guess(degree, self.bounds(degree, self.points_tangents)),
    #                 method=method,
    #                 bounds=self.bounds(degree, self.points_tangents),
    #                 options=options
    #             )
    #             if result.success and result.fun < 1e-6:
    #                 n_param = len(result.x) // len(self.points_tangents)
    #                 for curve, pt in enumerate(self.points_tangents):
    #                     params = result.x[curve * n_param:(curve + 1) * n_param]
    #                     self.optimized_curves.append(self.control2Bezier(pt, params))
    #                     self.optimized_radii.append(self.radii[curve])
    #                 break
    #     else:
    #         # For each sequential group of curves
    #         for i in range(len(self.points_tangents) // self.sequential):
    #             start = i * self.sequential
    #             end = (i + 1) * self.sequential
    #             points_tangents = self.points_tangents[start:end]
    #             for degree in range(self.min_degree, self.max_degree + 1):
    #                 result = minimize(
    #                     self.objective_function,
    #                     args=(tuple(range(start, end)),),
    #                     x0=self.initial_guess(degree, self.bounds(degree, self.points_tangents)),
    #                     method=method,
    #                     bounds=self.bounds(degree, points_tangents),
    #                     options=options
    #                 )
    #                 if result.success and result.fun < 1e-6:
    #                     n_param = len(result.x) // len(points_tangents)
    #                     for curve, pt in enumerate(points_tangents):
    #                         params = result.x[curve * n_param:(curve + 1) * n_param]
    #                         self.optimized_curves.append(self.control2Bezier(pt, params))
    #                         self.optimized_radii.append(self.radii[start + curve])
    #                     break

    #     return result

    def length_badness(self, curves):
        """
        Compute the length of the curves and their radii.

        Parameters:
            curves (list): A list of fibers, where each fiber is represented as an array of points in 3D space (n, 3).
            radii (list): A list of radii corresponding to each fiber.

        Returns:
            float: The total length of the curves.
        """
        total_length = 0.0
        for i in range(len(curves)):
            fiber = curves[i]
            length = np.sum(np.linalg.norm(np.diff(fiber, axis=0), axis=1))
            total_length += length
        return total_length
    
    def curvature_badness(self, curves):
        """
        Compute the curvature of the curves.

        Parameters:
            curves (list): A list of fibers, where each fiber is represented as an array of points in 3D space (n, 3).

        Returns:
            float: The total curvature of the curves.
        """
        max_curvature = 0.0
        for fiber in curves:
            # Compute the curvature using finite differences
            arc_lengths = np.cumsum(np.linalg.norm(np.diff(fiber, axis=0), axis=1))
            arc_lengths = np.insert(arc_lengths, 0, 0)  # Include the starting point
            tangent = np.gradient(fiber, arc_lengths, axis=0)
            tangent_norm = np.linalg.norm(tangent, axis=1)
            tangent /= tangent_norm[:, np.newaxis]
            curvature = np.linalg.norm(np.gradient(tangent, arc_lengths, axis=0), axis=1)
            max_curvature = max(max_curvature, np.max(curvature))
        return max_curvature
    
    def contact_badness_spheres(self, curves, radii, check_optimized=False):
        """
        Compute the amount of contact between fibers.

        Parameters:
            curves (list): A list of fibers, where each fiber is represented as an array of points in 3D space (n, 3).
            radii (list): A list of radii corresponding to each fiber.

        Returns:
            float: The total amount of contact between fibers.
        """
        total_contact = 0.0

        # Between new curves
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                fiber1 = curves[i]
                fiber2 = curves[j]
                radius_sum = radii[i] + radii[j]

                for p1 in fiber1:
                    for p2 in fiber2:
                        distance = np.linalg.norm(p1 - p2)
                        if distance < radius_sum:
                            total_contact += radius_sum - distance

        # Between new and optimized curves
        for i in range(len(curves)):
            for j in range(len(self.optimized_curves)):
                fiber1 = curves[i]
                fiber2 = self.optimized_curves[j].strands[0]
                radius_sum = radii[i] + self.optimized_radii[j]

                for p1 in fiber1:
                    for p2 in fiber2:
                        distance = np.linalg.norm(p1 - p2)
                        if distance < radius_sum:
                            total_contact += radius_sum - distance
        
        # Between new and fixed curves
        if self.fixed_curves is not None:
            for i in range(len(curves)):
                for j in range(len(self.fixed_curves)):
                    fiber1 = curves[i]
                    fiber2 = self.fixed_curves[j]
                    radius_sum = radii[i] + self.fixed_radii[j]

                    for p1 in fiber1:
                        if np.linalg.norm(fiber1[0] - p1) < radii[i] or np.linalg.norm(fiber1[-1] - p1) < radii[i]:
                            continue
                        for p2 in fiber2:
                            if np.linalg.norm(fiber2[0] - p2) < self.fixed_radii[j] or np.linalg.norm(fiber2[-1] - p2) < self.fixed_radii[j]:
                                continue
                            distance = np.linalg.norm(p1 - p2)
                            if distance < radius_sum:
                                total_contact += radius_sum - distance

        if check_optimized:
            # Between optimized curves
            for i in range(len(self.optimized_curves)):
                for j in range(i + 1, len(self.optimized_curves)):
                    fiber1 = self.optimized_curves[i].strands[0]
                    fiber2 = self.optimized_curves[j].strands[0]
                    radius_sum = self.optimized_radii[i] + self.optimized_radii[j]

                    for p1 in fiber1:
                        for p2 in fiber2:
                            distance = np.linalg.norm(p1 - p2)
                            if distance < radius_sum:
                                total_contact += radius_sum - distance

            # Between optimized and fixed curves
            if self.fixed_curves is not None:
                for i in range(len(self.optimized_curves)):
                    for j in range(len(self.fixed_curves)):
                        fiber1 = self.optimized_curves[i].strands[0]
                        fiber2 = self.fixed_curves[j]
                        radius_sum = self.optimized_radii[i] + self.fixed_radii[j]

                        for p1 in fiber1:
                            for p2 in fiber2:
                                distance = np.linalg.norm(p1 - p2)
                                if distance < radius_sum:
                                    total_contact += radius_sum - distance

        return total_contact
    
    def contact_badness_cylinders(self, curves, radii, check_optimized=False):
        """
        Compute the amount of contact between fibers.

        Parameters:
            curves (list): A list of fibers, where each fiber is represented as an array of points in 3D space (n, 3).
            radii (list): A list of radii corresponding to each fiber.

        Returns:
            float: The total amount of contact between fibers.
        """
        total_contact = 0.0

        # Between new curves
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                fiber1 = curves[i]
                fiber2 = curves[j]
                radius_sum = radii[i] + radii[j]

                for p in range(len(fiber1)-1):
                    for q in range(len(fiber2)-1):
                        p1, p2 = fiber1[p] , fiber1[p+1]
                        q1, q2 = fiber2[q] , fiber2[q+1]
                        distance = segment_distance(p1, p2, q1, q2)
                        if distance < radius_sum:
                            total_contact += radius_sum - distance

        # Between new and optimized curves
        for i in range(len(curves)):
            for j in range(len(self.optimized_beziers)):
                fiber1 = curves[i]
                fiber2 = self.optimized_beziers[j].strands
                radius_sum = radii[i] + self.radii[j] # HERE NEED TO CHANGE RADII FOT OPTIMIZED CURVES

                for p in range(len(fiber1)-1):
                    for q in range(len(fiber2)-1):
                        p1, p2 = fiber1[p] , fiber1[p+1]
                        q1, q2 = fiber2[q] , fiber2[q+1]
                        distance = segment_distance(p1, p2, q1, q2)
                        if distance < radius_sum:
                            total_contact += radius_sum - distance
        
        # Between new and fixed curves
        if self.fixed_helices is not None:
            for i in range(len(curves)):
                for h in self.fixed_helices:
                    fiber1 = curves[i]

                    # Check intersection with fixed helix bounding box
                    if not any(np.all(h.bounding_box[0] <= p) and np.all(p <= h.bounding_box[1]) for p in fiber1):
                        continue

                    for j in range(len(h.strands)):
                        fiber2 = h.strands[j]
                        radius_sum = radii[i] + self.radii[j] # HERE NEED TO CHANGE RADII FOR FIXED CURVES

                        contiguous = False
                        if np.linalg.norm(fiber1[0] - fiber2[0]) < 1e-12 or \
                            np.linalg.norm(fiber1[-1] - fiber2[-1]) < 1e-12 or \
                            np.linalg.norm(fiber1[0] - fiber2[-1]) < 1e-12 or \
                            np.linalg.norm(fiber1[-1] - fiber2[0]) < 1e-12:
                            contiguous = True

                        for p in range(len(fiber1)-1):
                            p1, p2 = fiber1[p] , fiber1[p+1]
                            if contiguous and (np.linalg.norm(fiber1[0] - p1) < radii[i] or np.linalg.norm(fiber1[-1] - p2) < radii[i]):
                                continue
                            for q in range(len(fiber2)-1):
                                q1, q2 = fiber2[q] , fiber2[q+1]
                                if contiguous and (np.linalg.norm(fiber2[0] - q1) < self.radii[j] or np.linalg.norm(fiber2[-1] - q2) < self.radii[j]): # HERE NEED TO CHANGE RADII FOR FIXED CURVES
                                    continue
                                distance = segment_distance(p1, p2, q1, q2)
                                if distance < radius_sum:
                                    total_contact += radius_sum - distance
        
        return total_contact
    
    def initial_guess(self, degree, bounds, randomize=False):
        if randomize:
            if degree == 3:
                return [np.random.uniform(low, high) for low, high in bounds]
            elif degree == 4:
                return [np.random.uniform(low, high) for low, high in bounds]
            elif degree == 5:
                return [np.random.uniform(low, high) for low, high in bounds]
        else:
            if degree == 3:
                return [(low + high) / 2 for low, high in bounds]
            elif degree == 4:
                return [(low + high) / 2 for low, high in bounds]
            elif degree == 5:
                return [(low + high) / 2 for low, high in bounds]
        
    def bounds(self, degree, points_tangents):
        if degree == 3:
            return [(0.5, 1) for _ in range(2 * len(points_tangents))]
        elif degree == 4:
            return [(0.67, 1), (0.67, 1), (0, 1), (-np.pi, np.pi)] * len(points_tangents)
        elif degree == 5:
            return [(0.75, 1), (0.75, 1), (0, 1), (-np.pi, np.pi), (0, 1), (-np.pi, np.pi)] * len(points_tangents)
    
    def control2Bezier(self, points_tangents, params, n=15):
        # Create a Bezier curve given the control points (-> extremes and tangents) and the parameters (-> degree)
        p1, t1, p2, t2 = points_tangents
        distance = np.linalg.norm(p1 - p2)
        if len(params) == 2:
            control_points = [p1, p1 + params[0] * t1 * distance, p2 + params[1] * t2 * distance, p2]
        elif len(params) == 4:
            p3 = (p1 + p2) / 2
            t12 = (t1 + t2) / 2
            if np.linalg.norm(t12) < 1e-12: # If t1 // t2
                t12 = t1                    # Take t1
            if np.linalg.norm(np.cross((p2 - p1) / np.linalg.norm(p2 - p1), t12)) < 1e-12: # If t12 // (p2 - p1)
                t12 = np.random.rand(3) - 0.5  # Generate a random vector
            t12 = np.cross(np.cross(p2 - p1, t12), p2 - p1) # Take the orthogonal to p2 - p1 in the plane of t12
            t12 = t12 / np.linalg.norm(t12)
            # Normalize the axis of rotation
            axis = (p2 - p1) / np.linalg.norm(p2 - p1)
            # Compute the rotation matrix using Rodrigues' rotation formula
            angle = params[3]
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            # Rotate t3
            t3 = rotation_matrix @ t12
            control_points = [p1, p1 + params[0] * t1 * distance / 2, p3 + params[2] * t3 * distance / 2, p2 + params[1] * t2 * distance / 2, p2]
        elif len(params) == 6:
            p3 = p1 + (p2 - p1) / 3
            p4 = p1 + 2 * (p2 - p1) / 3
            t12 = (t1 + t2) / 2
            if np.linalg.norm(t12) < 1e-12: # If t1 // t2
                t12 = t1                    # Take t1
            if np.linalg.norm(np.cross((p2 - p1) / np.linalg.norm(p2 - p1), t12)) < 1e-12: # If t12 // (p2 - p1)
                t12 = np.random.rand(3) - 0.5  # Generate a random vector
            t12 = np.cross(np.cross(p2 - p1, t12), p2 - p1) # Take the orthogonal to p2 - p1 in the plane of t12
            t12 = t12 / np.linalg.norm(t12)
            # Normalize the axis of rotation
            axis = (p2 - p1) / np.linalg.norm(p2 - p1)
            # Compute the rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            angle = params[3]
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            # Rotate t3
            t3 = rotation_matrix @ t12
            angle = params[5]
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            # Rotate t4
            t4 = rotation_matrix @ t12
            control_points = [p1, p1 + params[0] * t1 * distance / 3, p3 + params[2] * t3 * distance / 2, p4 + params[4] * t4 * distance / 2, p2 + params[1] * t2 * distance / 3, p2]

        return Bezier(control_points, n)
    
    def control2Bezier_fast(self, points_tangents, params, n=15):
        # Create a Bezier curve given the control points (-> extremes and tangents) and the parameters (-> degree)
        p1, t1, p2, t2 = points_tangents
        distance = np.linalg.norm(p1 - p2)
        axis = (p2 - p1) / distance  # Precompute axis for rotation

        def compute_rotation_matrix(angle):
            # Compute the rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        if len(params) == 2:
            control_points = [p1, p1 + params[0] * t1 * distance, p2 + params[1] * t2 * distance, p2]
        elif len(params) == 4:
            p3 = (p1 + p2) / 2
            t12 = (t1 + t2) / 2
            t12_norm = np.linalg.norm(t12)
            if t12_norm < 1e-12:  # If t1 // t2
                t12 = t1
            else:
                t12 /= t12_norm
            if np.linalg.norm(np.cross(axis, t12)) < 1e-12:  # If t12 // axis
                t12 = np.random.rand(3) - 0.5  # Generate a random vector
            t12 = np.cross(np.cross(axis, t12), axis)  # Orthogonalize t12 to axis
            t12 /= np.linalg.norm(t12)
            rotation_matrix = compute_rotation_matrix(params[3])
            t3 = rotation_matrix @ t12
            control_points = [p1, p1 + params[0] * t1 * distance / 2, p3 + params[2] * t3 * distance / 2, p2 + params[1] * t2 * distance / 2, p2]
        elif len(params) == 6:
            p3 = p1 + (p2 - p1) / 3
            p4 = p1 + 2 * (p2 - p1) / 3
            t12 = (t1 + t2) / 2
            t12_norm = np.linalg.norm(t12)
            if t12_norm < 1e-12:  # If t1 // t2
                t12 = t1
            else:
                t12 /= t12_norm
            if np.linalg.norm(np.cross(axis, t12)) < 1e-12:  # If t12 // axis
                t12 = np.random.rand(3) - 0.5  # Generate a random vector
            t12 = np.cross(np.cross(axis, t12), axis)  # Orthogonalize t12 to axis
            t12 /= np.linalg.norm(t12)
            rotation_matrix_3 = compute_rotation_matrix(params[3])
            t3 = rotation_matrix_3 @ t12
            rotation_matrix_5 = compute_rotation_matrix(params[5])
            t4 = rotation_matrix_5 @ t12
            control_points = [p1, p1 + params[0] * t1 * distance / 3, p3 + params[2] * t3 * distance / 2, p4 + params[4] * t4 * distance / 2, p2 + params[1] * t2 * distance / 3, p2]

        return Bezier(control_points, n)
    
    def objective_function(self, params, ids, badness_type=['contact', 'curvature']):

        n_param = len(params) // len(ids)
        j = 0
        curves = []
        radii = []
        for curve, points_tangents in enumerate(self.points_tangents):
            if curve not in ids:
                continue
            curves.append(self.control2Bezier(points_tangents, params[j*n_param:(j+1)*n_param]).strands)
            radii.append(self.radii[curve])
            j += 1

        # Compute and return the badnesses only if requested
        badness = {}
        if 'contact' in badness_type:
            badness['contact'] = 1e6 * self.contact_badness_cylinders(curves, radii)
        if 'length' in badness_type:
            badness['length'] = self.length_badness(curves)
        if 'curvature' in badness_type:
            badness['curvature'] = self.curvature_badness(curves)

        return sum([badness[b] for b in badness_type])
    
def grading_field(x, y, z, center=None):
    # for wavelength grading
    # scale = np.array([1, 0.5, 0.])
    # center_field = np.array([1.5, 2, 1])

    # for helix radius grading
    scale = np.array([0.05, 0.1, 0.0])
    center_field = np.array([0.35, 0.25, 0.25])

    center = np.array(center) if center is not None else np.zeros(3)
    return np.stack([(center_field[0] + scale[0] * (x - center[0])),
                        (center_field[1] + scale[1] * (y - center[1])),
                         (center_field[2] + scale[2] * (z - center[2]))], axis=0)

class WovenLattice:

    def __init__(self, filename=None, N=4, **kwargs):
        # From input
        self.filename = filename
        self.N = N

        # Topology
        self.nodes = None
        self.types = None
        self.status = None
        self.edges = None
        self.counteredges = None
        self.topology = None
        self.port_paths = []
        self.loops = []

        # Helper
        self.connections = []
        self.solution = None
        self.expanded_solution = None

        # Internal structures
        self.helices = {}
        self.reference_frames = {}
        self.beziers = {}
        self.strands = StrandsCollection()
        self.expanded_points = []
        self.expanded_junctions = []

        # Geometric parameters
        self.height = 0.5
        self.strand_radius = kwargs.pop('strand_radius', 0.05)
        self.helix_radius_field = kwargs.pop('helix_radius_grading', None)
        if self.helix_radius_field is None:
            self.helix_radius = kwargs.pop('helix_radius', None)
        self.helix_wavelength_field = kwargs.pop('helix_wavelength_grading', None)
        if self.helix_wavelength_field is None:
            self.wavelength = kwargs.pop('helix_wavelength', None)
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def generate_topology(self, conjugacy=True, max_multiplicity=None, stop_early=False, discard_threshold=None, restart=1, permuted_solutions=[0]):
        self.topology = Topology(self.filename, self.N, conjugacy=conjugacy, max_multiplicity=max_multiplicity)
        # self.topology = Topology.load("topology_triangular_extruded_N4.pickle")
        self.topology.generate(stop_early=stop_early, discard_threshold=discard_threshold, restart=restart, permuted_solutions=permuted_solutions)

        # Move nodes, types, edges and counteredges from topology.graph to the WovenLattice instance
        self.nodes = self.topology.graph.points
        self.types = self.topology.graph.types
        self.edges = self.topology.graph.edges
        self.status = self.topology.graph.status
        self.counteredges = self.topology.graph.counteredges

    def generate_geometry(self, solution=0, permuted_solution=0, expanded=False, **kwargs):
        """
        Generate the geometry of the lattice based on the provided solution and expanded solution.
        
        Args:
            solution (list): A list of tuples representing the solution.
            expanded_solution (list): A list of tuples representing the expanded solution.
            **kwargs: Additional keyword arguments for customization.
        """
        # Move solution and expanded_solution from topology to the WovenLattice instance
        self.solution = self.topology.permuted_solutions[solution][permuted_solution]
        if expanded:
            self.expanded_solution = self.topology.expanded_solutions[solution, permuted_solution]

        min_degree = kwargs.get('bezier_min_degree', 3)
        max_degree = kwargs.get('bezier_max_degree', 5)
        sequential = kwargs.get('sequential', 2)
        self.generate_base()
        self.generate_points_tangents()
        self.generate_connections(min_degree, max_degree, sequential)
        self.compute_strands(expanded)


    def read_lattice_file(self, filename):
        self.nodes, connectivities, self.types = read_lattice_file(filename)
        self.edges = defaultdict(int)
        for i, (n1, n2) in enumerate(connectivities):
            self.edges[i] = tuple(sorted((n1, n2)))

    def find_side(self, current_edge, next_edge):
        """
        Find the correct side of the helix-edge to connect to the next helix-edge.
        """
        # Extract nodes from the current and next edges
        current_nodes = set(self.edges[current_edge])
        next_nodes = set(self.edges[next_edge])

        # Find the common node (j) and the unique nodes (n1, n2)
        j = current_nodes.intersection(next_nodes).pop()
        n1 = (current_nodes - {j}).pop()
        n2 = (next_nodes - {j}).pop()

        if self.types[j] not in ["junction"]:
            raise ValueError(f"Node {j} is not a junction.")

        # Helix-edges with junctions only are oriented from smaller to larger node id
        c1 = -1 if n1 < j else 0
        c2 = -1 if n2 < j else 0
        c1_prev = -2 if n1 < j else 1
        c2_prev = -2 if n2 < j else 1
        if self.types[n1] in ["port", "counterport"]:
            c1 = -1
            c1_prev = -2
        if self.types[n2] in ["port", "counterport"]:
            c2 = -1
            c2_prev = -2
            
        return c1, c2, c1_prev, c2_prev

    def generate_points_tangents(self):
        # Reset connections
        self.connections = []
        for path in self.solution:
            for i in range(len(path) - 1):
                edge, strand = path[i]
                next_edge, next_strand = path[i + 1]
                # Append position of the extremes and their tangents
                c1, c2, c1_prev, c2_prev = self.find_side(edge, next_edge)
                p1, p2 = self.helices[edge].strands[strand][c1], self.helices[next_edge].strands[next_strand][c2]
                t1, t2 = p1 - self.helices[edge].strands[strand][c1_prev], p2 - self.helices[next_edge].strands[next_strand][c2_prev]
                t1, t2 = t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2)
                self.connections.append(((edge, strand), (next_edge, next_strand), (p1, t1, p2, t2)))
            # Inner loops
            if len(path) > 2 and bool(set(self.edges[path[0][0]]).intersection(set(self.edges[path[-1][0]]))):
                edge, strand = path[0]
                next_edge, next_strand = path[-1]
                c1, c2, c1_prev, c2_prev = self.find_side(edge, next_edge)
                p1, p2 = self.helices[edge].strands[strand][c1], self.helices[next_edge].strands[next_strand][c2]
                t1, t2 = p1 - self.helices[edge].strands[strand][c1_prev], p2 - self.helices[next_edge].strands[next_strand][c2_prev]
                t1, t2 = t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2)
                self.connections.append(((edge, strand), (next_edge, next_strand), (p1, t1, p2, t2)))

    def generate_base(self):
        if self.helices != {}:
            return
        e1 = np.array([np.sqrt(2), np.sqrt(3), 1])
        for e, edge in self.edges.items():
            Oxyz = ReferenceFrame()
            # Helix-edge with port has node1 on boundary and node2 on interior
            if self.types[edge[0]] in ["port", "counterport"]:
                point1 = self.nodes[edge[0]]
                point2 = self.nodes[edge[1]]
            elif self.types[edge[1]] in ["port", "counterport"]:
                point1 = self.nodes[edge[1]]
                point2 = self.nodes[edge[0]]
            else:
                # Helix-edge with junctions only has node1 on junction with smaller id
                point1 = self.nodes[edge[0]]
                point2 = self.nodes[edge[1]]
                point1 = point1 + 0.5 * (point2 - point1) - 0.5 * self.height * (point2 - point1) / np.linalg.norm(point2 - point1)
            Oxyz.align(point1, point2, e1)
            self.reference_frames[e] = (Oxyz)
            helix = Helix(self.N, self.helix_radius, self.wavelength, self.height, self.helix_radius_field, self.helix_wavelength_field, newOxyz=Oxyz, strand_radius=self.strand_radius)
            helix.rotate(np.eye(3), Oxyz.basis_vectors)
            helix.translate(Oxyz.origin)
            helix.compute_bounding_box()
            self.helices[e] = (helix)
            # Revert the strand ids for counteredges
            if self.counteredges[e] in self.helices.keys():
                self.helices[e].strands[1::] = self.helices[e].strands[:0:-1]
                self.helices[e].radii[1::] = self.helices[e].radii[:0:-1]

    def generate_connections(self, min_degree=3, max_degree=3, sequential=None):
        points_tangents = [connection for _, _, connection in self.connections]
        strands_radii = [self.strand_radius for _ in range(len(points_tangents))]
        fixed_helices = list(self.helices.values())
        # Create an Optimizer instance
        optimizer = Optimizer(points_tangents, strands_radii, min_degree=min_degree, max_degree=max_degree, fixed_helices=fixed_helices, sequential=sequential)
        optimizer.optimize(method='SLSQP', options={'disp': False, 'maxiter': 10})
        for i, bezier in enumerate(optimizer.optimized_beziers):
            self.beziers[self.connections[i][0], self.connections[i][1]] = bezier
        
    def clean_connections(self):
        self.beziers = {}

    def compute_strands(self, expanded=False):
        # Clean strands
        self.strands = StrandsCollection()
        if not expanded:
            for path in self.solution:
                strand = []
                port_path = (path[0][0],)
                for i in range(len(path) - 1):
                    edge, id = path[i]
                    next_edge, next_id = path[i + 1]
                    port_path += (next_edge,)
                    c1, c2, _, _ = self.find_side(edge, next_edge)
                    if i == 0:
                        # Append helix
                        strand.append(self.helices[edge].strands[id][::1 if c1 == -1 else -1])
                    # Append bezier
                    strand.append(self.beziers[(edge, id), (next_edge, next_id)].strands[1::])
                    # Append helix
                    start = 1 if c2 ==0 else -2
                    step = 1 if c2 == 0 else -1
                    strand.append(self.helices[next_edge].strands[next_id][start::step])
                # Inner loops
                if len(path) > 2 and bool(set(self.edges[path[0][0]]).intersection(set(self.edges[path[-1][0]]))):
                    edge, id = path[0]
                    next_edge, next_id = path[-1]
                    c1, c2, _, _ = self.find_side(edge, next_edge)
                    # Append bezier
                    strand.append(self.beziers[(edge, id), (next_edge, next_id)].strands[-1::-1])
                self.expanded_points.append([strand[0][0], strand[-1][-1]])
                self.strands.add_strand(np.concatenate(strand, axis=0))
                self.port_paths.append(port_path)
            # Add junctions
            for i, node in enumerate(self.nodes):
                if self.types[i] == "junction":
                    self.expanded_junctions.append(node)
        else:
            for paths_and_translations in self.expanded_solution:
                strand = []
                self.expanded_points.append([])
                uc_explored = set()
                for path, translation in paths_and_translations:
                    # Add junctions
                    if tuple(np.round(translation, 2)) not in uc_explored:
                        for i, node in enumerate(self.nodes):
                            if self.types[i] == "junction":
                                self.expanded_junctions.append(node + translation)
                    # Add translation (rounded to 2 decimals) with its relative position in the strand
                    uc_explored.add(tuple(np.round(translation, 2)))
                    curve = []
                    for i in range(len(path) - 1):
                        edge, id = path[i]
                        next_edge, next_id = path[i + 1]
                        c1, c2, _, _ = self.find_side(edge, next_edge)
                        if i == 0:
                            # Append helix
                            curve.append(self.helices[edge].strands[id][::1 if c1 == -1 else -1])
                        # Append bezier
                        curve.append(self.beziers[(edge, id), (next_edge, next_id)].strands[1::])
                        # Append helix
                        start = 1 if c2 ==0 else -2
                        step = 1 if c2 == 0 else -1
                        curve.append(self.helices[next_edge].strands[next_id][start::step])
                    curve = np.concatenate(curve, axis=0) + translation
                    if strand:
                        if np.allclose(strand[-1], curve[0], atol=1e-5):
                            strand.extend(curve[1:])
                            self.expanded_points[-1].append(curve[-1])
                        elif np.allclose(strand[-1], curve[-1], atol=1e-5):
                            strand.extend(curve[-2::-1])
                            self.expanded_points[-1].append(curve[0])
                        elif np.allclose(strand[0], curve[0], atol=1e-5):
                            strand[0:0] = curve[-1:0:-1]
                            self.expanded_points[-1].append(curve[-1])
                        elif np.allclose(strand[0], curve[-1], atol=1e-5):
                            strand[0:0] = curve[:-1]
                            self.expanded_points[-1].append(curve[0])
                        else:
                            print("Warning: strands are not connected")
                        # If close loop
                        # if np.allclose(strand[0], strand[-1], atol=1e-5):
                        #     strand = strand[0:-1]
                    else:
                        strand.extend(curve)
                        self.expanded_points[-1].append(curve[0])
                        self.expanded_points[-1].append(curve[-1])
                self.expanded_points[-1].append(strand[-1])
                self.strands.add_strand(np.array(strand))
                self.port_paths.append(path) # This is just temporary! Need to implemetn the port paths for expanded strands
                # break
                    



    

    def plot(self, ax=None, base=False, frames=False, label=False, **kwargs):
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        if frames:
            for frame in self.reference_frames.values():
                frame.plot(ax=ax, scale=0.25, linewidth=4)
        if base:
            for i, helix in enumerate(self.helices.values()):
                helix.plot(ax=ax, color='k', linewidth=2, label=label, **kwargs)
                if label:
                    midpoint = np.mean([strand[0] for strand in helix.strands], axis=0)
                    ax.text(midpoint[0], midpoint[1], midpoint[2], color = 'k', fontsize=10, fontweight='bold', s=str(i))
        else:
            self.strands.plot(ax=ax, **kwargs)
        
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        plt.show()

    def plot_vedo(self, select_strands=None, color_map=None, base = False, radius=None, subnodes=False, starting_seed=0, **kwargs):
        """
        Plot the strands using vedo.

        Parameters:
            color_map (str or tuple, optional): Color of the strands. Defaults to None.
            radius (float, optional): Radius of the strands. Defaults to None.
        """
        seed = starting_seed
        objects = []
        if base:
            for i, helix in enumerate(self.helices.values()):
                if color_map is None:
                    color_map = 'lightgrey'
                if radius is None:
                    radius = 0.035
                for j, strand in enumerate(helix.strands):
                    # if i > 3:
                    #     if j == 1:
                    #         color = 'lightblue'
                    #         helix_mesh = vedo.Sphere(strand[-1], r=radius*2, c=color)
                    #         helix_mesh.color('red')
                    #         objects.append(helix_mesh)
                    #     else:
                    #         color = 'black'
                    helix_mesh = vedo.Tube(strand, r=radius, res=12)
                    helix_mesh.color(color)
                    objects.append(helix_mesh)
                # add subnodes
                # if subnodes:
                #     for point in self.expanded_points[i]:
                #         vedo_points = vedo.Sphere(point, r=radius*2.5, c=color)
                #         objects.append(vedo_points)
        else:
            if color_map is None:
                color_map = 'lightgrey'
            unpaired_paths = {}
            seed = starting_seed
            for i, strand in enumerate(self.strands.strands):
                # Here still need to check if by selective plot the compute color is not affected
                # It should not be a problem cause i is indexing inside compute_color
                # if select_strands is not None and i not in select_strands:
                #     seed += 1
                #     continue
                if select_strands is not None and i not in select_strands:
                    seed += 1
                    color = 'red'
                    continue
                else:
                    color, seed, unpaired_paths = compute_color(color_map, self.port_paths[i], unpaired_paths, self.topology.edges, self.topology.counterports, seed=seed, conjugacy_type=0)
                # if i in [2]:
                #     color = 'red'
                # else:
                #     color = 'lightgrey'
                # if i == 4:
                #     color = 'blue'
                # if i == 0:
                #     color = 'red'
                # if i == 3:
                #     color = 'blue'
                # if i == 2:
                #     color = 'green'
                # if i == 1:
                #     color = 'orange'
                strand_mesh = vedo.Tube(strand, r=self.strand_radius, res=12)
                strand_mesh.color(color)
                objects.append(strand_mesh)

                # if i == 0 or i == 4:
                #     color = 'lightgrey'
                #     translation = strand[-1] - strand[0]
                #     N = 2
                #     for j in range(-N, N+1):
                #         if j == 0:
                #             continue
                #         strand_mesh = vedo.Tube(strand + j * translation, r=radius, res=12)
                #         strand_mesh.color(color)
                #         objects.append(strand_mesh)

                # add subnodes
                if subnodes:
                    for point in self.expanded_points[i]:
                        vedo_points = vedo.Sphere(point, r=radius*2.5, c=color)
                        objects.append(vedo_points)
        # add junctions
        if subnodes:
            for junction in self.expanded_junctions:
                # if junction[0] <=0 and junction[2]>=0:
                vedo_junction = vedo.Sphere(junction, r=radius*16, c='grey', alpha=0.5)
                objects.append(vedo_junction)

        for edge in self.edges.values():
            n1 = self.nodes[edge[0]]
            n2 = self.nodes[edge[1]]
            # Use vedo.Line with dashed style instead of Tube
            line = vedo.Line([n1, n2], c='black', lw=4)
            line.lighting('off')
            line.linewidth(4)
            line.pattern('- -')
            objects.append(line)

        vedo.show(objects, axes=1, **kwargs)
        # return objects

    def plot_vedo_curvature(self, select_strands=None, color_map=None, base=False, radius=None, subnodes=False, starting_seed=0, **kwargs):
        """
        Plot the strands using vedo, coloring by local curvature (grey to red).

        Parameters:
            color_map (str or tuple, optional): Not used, replaced by curvature colormap.
            radius (float, optional): Radius of the strands. Defaults to None.
        """

        seed = starting_seed
        objects = []
        if base:
            for i, helix in enumerate(self.helices.values()):
                if radius is None:
                    radius = 0.035
                for j, strand in enumerate(helix.strands):
                    helix_mesh = vedo.Tube(strand, r=radius, res=12)
                    helix_mesh.color('lightgrey')
                    objects.append(helix_mesh)
        else:
            # First, compute the maximum curvature among all strands
            max_curvature = 0.0
            all_curvatures = []
            for strand in self.strands.strands:
                points = np.asarray(strand)
                arc_lengths = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                arc_lengths = np.insert(arc_lengths, 0, 0)
                tangent = np.gradient(points, arc_lengths, axis=0)
                tangent_norm = np.linalg.norm(tangent, axis=1)
                tangent[tangent_norm > 0] /= tangent_norm[tangent_norm > 0][:, np.newaxis]
                curvature = np.linalg.norm(np.gradient(tangent, arc_lengths, axis=0), axis=1)
                all_curvatures.append(curvature)
                if np.max(curvature) > max_curvature:
                    max_curvature = np.max(curvature)

            for i, strand in enumerate(self.strands.strands):
                if radius is None:
                    radius = 0.035
                if select_strands is not None and i not in select_strands:
                    continue

                # Use precomputed curvature
                curvature = all_curvatures[i]
                # Use a colormap based on normalized curvature (scalar from 0 to 1)
                colors = vedo.color_map(curvature, name='pink_r', vmin=0, vmax=max_curvature)
                strand_mesh = vedo.Tube(strand, r=radius, res=12, c=colors)
                objects.append(strand_mesh)

                if subnodes:
                    for point in self.expanded_points[i]:
                        vedo_points = vedo.Sphere(point, r=radius*2.5, c='grey')
                        objects.append(vedo_points)
        # add junctions
        if subnodes:
            for junction in self.expanded_junctions:
                vedo_junction = vedo.Sphere(junction, r=radius*16, c='grey', alpha=0.5)
                objects.append(vedo_junction)

        for edge in self.edges.values():
            n1 = self.nodes[edge[0]]
            n2 = self.nodes[edge[1]]
            line = vedo.Line([n1, n2], c='black', lw=4)
            line.lighting('off')
            line.linewidth(4)
            line.pattern('- -')
            objects.append(line)

        vedo.show(objects, axes=0, **kwargs)

    def plot_vedo_animate(self, select_strands=None, color_map=None, base=False, radius=None, subnodes=False, starting_seed=0, n_frames=120, **kwargs):
        """
        Render an animation where the plot is rotated 360 degrees using vedo.

        Parameters:
            n_frames (int): Number of frames for the rotation animation.
        """
        seed = starting_seed
        objects = []
        objects1 = []
        objects2 = []
        if base:
            for i, helix in enumerate(self.helices.values()):
                if color_map is None:
                    color_map = 'lightgrey'
                if radius is None:
                    radius = 0.035
                for j, strand in enumerate(helix.strands):
                    helix_mesh = vedo.Tube(strand, r=radius, res=12)
                    helix_mesh.color(color_map)
                    objects.append(helix_mesh)
        else:
            if color_map is None:
                color_map = 'lightgrey'
            unpaired_paths = {}
            seed = starting_seed
            for i, strand in enumerate(self.strands.strands):
                if radius is None:
                    radius = 0.035
                color, seed, unpaired_paths = compute_color(
                    color_map, self.port_paths[i], unpaired_paths, self.topology.edges, self.topology.counterports, seed=seed, conjugacy_type=0
                )
                strand_mesh = vedo.Tube(strand, r=radius, res=12)
                strand_mesh.color(color)
                objects1.append(strand_mesh)
            seed = starting_seed
            for i, strand in enumerate(self.strands.strands):
                if select_strands is not None and i not in select_strands:
                    seed += 1
                    continue
                if radius is None:
                    radius = 0.035
                color, seed, unpaired_paths = compute_color(
                    color_map, self.port_paths[i], unpaired_paths, self.topology.edges, self.topology.counterports, seed=seed, conjugacy_type=0
                )
                strand_mesh = vedo.Tube(strand, r=radius, res=12)
                strand_mesh.color(color)
                objects2.append(strand_mesh)
        if subnodes:
            for junction in self.expanded_junctions:
                vedo_junction = vedo.Sphere(junction, r=radius*16, c='grey', alpha=0.5)
                objects.append(vedo_junction)
        for edge in self.edges.values():
            n1 = self.nodes[edge[0]]
            n2 = self.nodes[edge[1]]
            line = vedo.Line([n1, n2], c='black', lw=4)
            line.lighting('off')
            line.linewidth(4)
            line.pattern('- -')
            objects.append(line)

        # Animation: rotate the camera around the z axis
        plt = vedo.Plotter(interactive=False, offscreen=False, axes=0, **kwargs)
        # Always show objects and objects2
        plt.show(objects, resetcam=True)
        plt.camera.SetViewUp(0, 0, 1)
        plt.camera.SetFocalPoint(0, 0, 0)
        plt.camera.SetPosition(12, 12, 12)
        images = []
        n_frames = 360

        # Prepare objects1 for alpha animation
        for obj in objects1:
            obj.alpha(0.0)
        for obj in objects2:
            obj.alpha(1.0)

        for i in range(n_frames):
            # Compute alpha for objects1: fade in and out in a loop
            # Use a smooth sinusoidal function for alpha
            alpha = 0.5 * (1 + np.sin(2 * np.pi * i / n_frames))
            for obj in objects1:
                obj.alpha(alpha)
                if obj not in plt.actors:
                    plt.add(obj)
            for obj in objects2:
                obj.alpha(1.0)
                if obj not in plt.actors:
                    plt.add(obj)
            angle = 360 * i / n_frames
            plt.camera.Azimuth(360 / n_frames)
            plt.render()
            img = plt.screenshot(asarray=True)
            images.append(img)
        plt.close()

        # Save the images as a video (mp4)
        video_filename = "temp.mp4"
        imageio.mimsave(video_filename, images, fps=24)
        print(f"Saved animation to {video_filename}")


    def plot_original_lattice_vedo(self, color=None, radius=None, nodes=False, **kwargs):
        """
        Plot the original lattice using vedo.

        Parameters:
            color (str or tuple, optional): Color of the strands. Defaults to None.
            radius (float, optional): Radius of the strands. Defaults to None.
        """
        objects = []
        if radius is None:
            radius = 0.135
        if color is None:
            color = 'lightgrey'
        for edge in self.edges.values():
            n1 = self.nodes[edge[0]]
            n2 = self.nodes[edge[1]]
            objects.append(vedo.Tube([n1, n2], r=radius, res=36, c=color))
        if nodes:
            for i, node in enumerate(self.nodes):
                if self.types[i] == "junction":
                    vedo_node = vedo.Sphere(node, r=radius*2.5, c='k', alpha=0.5)
                    objects.append(vedo_node)
                elif self.types[i] in ["port", "counterport"]:
                    vedo_node = vedo.Sphere(node, r=radius*1.5, c='k', alpha=0.5)
                    vedo_text = vedo.shapes.Text3D(str(i+1), pos=node, s=1, c='k')
                    # objects.append(vedo_node)
                    objects.append(vedo_text)

        # for axis in [[[0,0,0], [1,0,0]], [[0,0,0], [0,1,0]], [[0,0,0], [0,0,-1]]]:
        #     n1 = np.array(axis[0])*1.2
        #     n2 = np.array(axis[1])*1.2
        #     # Use vedo.Line with dashed style instead of Tube
        #     line = vedo.Line([n1, n2], c='black', lw=4)
        #     line.lighting('off')
        #     line.linewidth(4)
        #     line.pattern('- -')
        #     objects.append(line)

        vedo.show(objects, axes=0, **kwargs)


    def plot_stl(self, output_filename='mesh.stl', radius=None, sphere_cap=False, sections=12, strand_id=None, type='strands'):

        def curve_to_tube(points, radius, sections=sections):
            """
            Create a lightweight tube by sweeping a circular profile along the curve.
            
            Parameters:
            - points: List of [x, y, z] points describing the curve.
            - radius: Radius of the tube.
            - sections: Number of sections for the circular cross-section.
            """
            # Create the circular cross-section profile
            theta = np.linspace(0, 2 * np.pi, sections + 1)
            circle = np.column_stack([np.cos(theta) * radius, np.sin(theta) * radius, np.zeros(sections + 1)])
            
            # Create a list to hold the vertices and faces
            vertices = []
            faces = []
            vertex_count = 0
            old_normal = [0, 0, 1]
            old_direction = [0, 0, 1]
            old_binormal = [0, 1, 0]

            # Generate vertices for the tube by sweeping the circle along the curve
            for i in range(len(points)):
                # Align the circle to the tangent direction of the curve
                if i < len(points) - 1:
                    # direction = np.array(points[i + 1]) - np.array(points[i])
                    if i > 0:
                        if np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) > np.linalg.norm(np.array(points[i]) - np.array(points[i - 1])):
                            direction = np.array(points[i]) - np.array(points[i-1])
                        else:
                            direction = np.array(points[i+1]) - np.array(points[i])
                    else:
                        direction = np.array(points[i+1]) - np.array(points[i])
                else:
                    direction = np.array(points[i]) - np.array(points[i - 1])
                    # If chain is a closed loop
                    if np.linalg.norm((np.array(points[0]) - np.array(points[-1]))) < 1.e-12:
                        continue
                
                direction /= np.linalg.norm(direction)
                # normal = np.cross([0, 0, 1], direction)  # Perpendicular vector
                normal = np.cross(old_binormal, direction)  # Perpendicular vector
                if np.linalg.norm(normal) == 0:  # Handle edge case where direction aligns with Z-axis
                    normal = [1, 0, 0]
                normal /= np.linalg.norm(normal)
                binormal = np.cross(direction, normal)
                
                # Rotation matrix to align the circle
                transform = np.array([normal, binormal, direction]).T
                old_binormal = binormal
                
                # Apply the transformation to the circle and translate to the current point
                swept_circle = (circle @ transform.T) + np.array(points[i])
                vertices.extend(swept_circle[:-1])  # Skip the last point to avoid duplication

                # Create faces connecting this section of the circle to the previous one
                if i > 0:
                    for j in range(sections):
                        next_j = (j + 1) % sections
                        faces.append([vertex_count + j, vertex_count + next_j, vertex_count - sections + j])
                        faces.append([vertex_count + next_j, vertex_count - sections + next_j, vertex_count - sections + j])
                        # Ensure the normals are outward-facing
                        face_normal = np.cross(
                            vertices[vertex_count + next_j] - vertices[vertex_count + j],
                            vertices[vertex_count - sections + j] - vertices[vertex_count + j]
                        )
                        center_to_section = vertices[vertex_count + j] - np.array(points[i])
                        if np.dot(face_normal, center_to_section) < 0:
                            faces[-1] = faces[-1][::-1]
                            faces[-2] = faces[-2][::-1]
                
                vertex_count += sections

            # Cap the ends to make the tube watertight
            # Cap the first end
            for j in range(1, sections - 1):
                faces.append([0, j + 1, j])
                
            # # Cap the last end
            start_of_last_circle = vertex_count - sections
            for j in range(1, sections - 1):
                faces.append([start_of_last_circle, start_of_last_circle + j, start_of_last_circle + j + 1])

            # Create the mesh
            meshes = [trimesh.Trimesh(vertices=vertices, faces=faces)]

            # If chain is a closed loop
            if np.linalg.norm((np.array(points[0]) - np.array(points[-1]))) < 1.e-12:
                # Create a cylinder to close the loop
                cylinder_mesh = trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(direction), sections=sections)
                transform_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
                cylinder_mesh.apply_transform(transform_matrix)
                cylinder_mesh.apply_translation((np.array(points[0]) + np.array(points[-2])) / 2)
                meshes.append(cylinder_mesh)
                # Create a sphere to close the loop
                sphere_mesh = trimesh.primitives.Sphere(radius=radius, center=points[0], subdivisions=sections//12+1)
                meshes.append(sphere_mesh)
                sphere_mesh = trimesh.primitives.Sphere(radius=radius, center=points[-2], subdivisions=sections//12+1)
                meshes.append(sphere_mesh)

            return meshes

        if radius is None:
            radius = self.strand_radius
        meshes = []
        if type == 'strands':
            curves = self.strands.strands
        elif type == 'beziers':
            curves = [bezier[1].strands for bezier in self.beziers]
        for i, strand in enumerate(curves):
            if strand_id is not None:
                if i not in strand_id:
                    continue
            meshes.extend(curve_to_tube(strand, radius))
            if sphere_cap:
                meshes.append(trimesh.primitives.Sphere(radius, center=strand[0], subdivisions=sections//12+1))
                meshes.append(trimesh.primitives.Sphere(radius, center=strand[-1], subdivisions=sections//12+1))
        # combined_mesh = trimesh.boolean.union(meshes, engine='manifold', check_volume=True)
        combined_mesh = trimesh.util.concatenate(meshes)

        combined_mesh.export(output_filename, 'stl')

    def export_strands(self, output_filename='strands.dat', N=1):
        """
        Export the strands to a dat file in the form of coordinates of nodes and connectivities.

        Parameters:
            output_filename (str): The name of the output file.
            N (int): Not used, kept for compatibility.
        """
        # Collect all unique nodes and assign indices
        node_list = []
        node_indices = {}
        connectivities = []
        for strand in self.strands.strands:
            strand_indices = []
            for node in strand:
                key = tuple(np.round(node, 8))  # Use rounded coordinates for uniqueness
                if key not in node_indices:
                    node_indices[key] = len(node_list)
                    node_list.append(node)
                    strand_indices.append(node_indices[key])
            # Create connectivity for this strand
            if np.linalg.norm(strand[0] - strand[-1]) < 1.e-12 and len(strand) > 2:
                # Closed loop
                connectivities.extend([[strand_indices[i], strand_indices[i+1]] for i in range(len(strand_indices)-1)])
                connectivities.append([strand_indices[-1], strand_indices[0]])
            else:
                connectivities.extend([[strand_indices[i], strand_indices[i+1]] for i in range(len(strand_indices)-1)])

        points = np.array(np.round(node_list, 8))
        connectivities = np.array(connectivities)
        vectors = self.topology.graph.translations

        # Find minimum number of linearly independent vectors
        independent_vectors = []
        for v in vectors.values():
            # Check if v is linearly independent from existing vectors
            if not independent_vectors:
                independent_vectors.append(v)
            else:
                # Stack existing vectors and check rank
                test_matrix = np.vstack(independent_vectors + [v])
                if np.linalg.matrix_rank(test_matrix) > len(independent_vectors):
                    independent_vectors.append(v)

        # Create vectors along each axis with the range of the lattice
        x_range = np.max(self.nodes[:, 0]) - np.min(self.nodes[:, 0])
        y_range = np.max(self.nodes[:, 1]) - np.min(self.nodes[:, 1])
        z_range = np.max(self.nodes[:, 2]) - np.min(self.nodes[:, 2])
        
        vectors = [np.array([x_range, 0, 0]), np.array([0, y_range, 0]), np.array([0, 0, z_range])]


        tessellated_points = points.tolist()
        tessellated_connectivities = []

        # Generate all possible multiplicities
        multiplicities = list(product(range(N), repeat=len(vectors)))
        point_index_map = {tuple(np.round(p, 8)): i for i, p in enumerate(tessellated_points)}
        
        for mult in multiplicities:
            v = np.zeros(3)
            for i, m in enumerate(mult):
                v += m * vectors[i]
            translated_points = np.round(points + v, 8)
            
            for tp in translated_points:
                tp_tuple = tuple(tp)
                if tp_tuple not in point_index_map:
                    point_index_map[tp_tuple] = len(tessellated_points)
                    tessellated_points.append(tp.tolist())
            
            base_index = len(tessellated_points) - len(translated_points)
            tessellated_connectivities.extend(
                [[point_index_map[tuple(np.round(points[start] + v, 8))], point_index_map[tuple(np.round(points[end] + v, 8))]]
                for start, end in connectivities]
            )
        with open(output_filename, 'w') as f:
            for pt in tessellated_points:
                f.write(f"{pt[0]}, {pt[1]}, {pt[2]}\n")
            f.write("connectivity\n")
            for conn in tessellated_connectivities:
                f.write(f"{conn[0]}, {conn[1]}\n")

class StrandsCollection:
    """
    A class representing a collection of strands in 3D space.

    Attributes:
        strands (list): A list of arrays, each containing the (x, y, z) coordinates of a strand.
    """

    def __init__(self, strands=None, radii=None):
        self.strands = strands if strands is not None else []
        self.radii = radii if radii is not None else []
        self.bounding_box = None

    def add_strand(self, strand, radius=None):
        self.strands.append(strand)
        if radius is not None:
            self.radii.append(radius)

    def compute_bounding_box(self):
        """
        Compute the bounding box of all strands in the collection.

        Returns:
            tuple: A tuple containing the minimum and maximum coordinates of the bounding box.
        """
        if not self.strands:
            return None
        min_coords = np.min([strand - np.expand_dims(self.radii[i], axis=1) for i, strand in enumerate(self.strands)], axis=(0, 1))
        max_coords = np.max([strand + np.expand_dims(self.radii[i], axis=1) for i, strand in enumerate(self.strands)], axis=(0, 1))
        self.bounding_box = (min_coords, max_coords)
        return self.bounding_box

    def rotate(self, source_frame, target_frame):
        """
        Rotate all strands to align with a target frame.

        Parameters:
            source_frame (np.ndarray): Array of shape (3, 3) representing the source frame (orthonormal basis vectors).
            target_frame (np.ndarray): Array of shape (3, 3) representing the target frame (orthonormal basis vectors).
        """
        rotation_matrix = target_frame @ np.linalg.inv(source_frame)
        self.strands = [strand @ rotation_matrix.T for strand in self.strands]

    def translate(self, vector):
        """
        Translate all strands by a given vector.

        Parameters:
            vector (np.ndarray): Array of shape (3,) representing the translation vector.
        """
        self.strands = [strand + vector for strand in self.strands]

    def plot(self, ax=None, color=None, linewidth=1, label=False):
        """
        Plot the strands in 3D space.

        Parameters:
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The axes to plot on. If None, a new figure and axes are created.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if color is None:
            colors = plt.cm.jet(np.linspace(0, 1, len(self.strands)))  # Generate a darker colormap
        else:
            colors = [color] * len(self.strands)
        for i, (strand, strand_color) in enumerate(zip(self.strands, colors)):
            np.random.seed(i*1000)
            if i > 4:
                np.random.seed(i*1000+1)
            strand_color = np.random.rand(3,)
            ax.plot(strand[:, 0], strand[:, 1], strand[:, 2], color=strand_color, linewidth=linewidth)
            if label:
                ax.text(strand[0, 0], strand[0, 1], strand[0, 2], str(i), color=strand_color, fontsize=8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

        if ax.figure is not plt.gcf():  # Show the plot only if a new figure was created
            plt.show()

class Bezier:
    """
    A class representing a Bezier curve in 3D space.

    Attributes:
        control_points (np.ndarray): Array of shape (n, 3) representing the control points of the Bezier curve.
        degree (int): Degree of the Bezier curve.
        points_per_curve (int): Number of points to generate along the curve.
    """

    def __init__(self, control_points=None, points_per_curve=1000, points_tangents=None, params=None, radius=None):
        self.control_points = np.array(control_points) if control_points is not None else np.zeros((4, 3))
        if points_tangents is not None and params is not None:
            self.control_points = self._pttocontrol(points_tangents, params)
        self.degree = len(self.control_points) - 1
        self.points_per_curve = points_per_curve
        self.strands = self._generate_bezier_curve()
        self.radii = np.array([radius] * self.points_per_curve) if radius is not None else np.zeros(self.points_per_curve)
        self.bounding_box = self._compute_bounding_box()

    def _compute_bounding_box(self):
        """
        Compute the bounding box of all strands in the collection.

        Returns:
            tuple: A tuple containing the minimum and maximum coordinates of the bounding box.
        """
        if not self.strands.size:
            return None
        min_coords = np.min(self.strands - self.radii[:, np.newaxis], axis=0)
        max_coords = np.max(self.strands + self.radii[:, np.newaxis], axis=0)
        return (min_coords, max_coords)

    def _bernstein_polynomial(self, i, n, t):
        """
        Compute the Bernstein polynomial value.

        Parameters:
            i (int): Index of the control point.
            n (int): Degree of the Bezier curve.
            t (float or np.ndarray): Parameter value(s) in the range [0, 1].

        Returns:
            float or np.ndarray: Value of the Bernstein polynomial.
        """
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def _generate_bezier_curve(self):
        """
        Generate the Bezier curve using the control points.

        Returns:
            np.ndarray: Array of shape (points_per_curve, 3) representing the points along the curve.
        """
        t = np.linspace(0, 1, self.points_per_curve)
        curve = np.zeros((self.points_per_curve, 3))

        for i in range(self.degree + 1):
            bernstein = self._bernstein_polynomial(i, self.degree, t)
            curve += np.outer(bernstein, self.control_points[i])

        return curve
    
    def _pttocontrol(self, points_tangents, params):
        # Create a Bezier curve given the control points (-> extremes and tangents) and the parameters (-> degree)
        p1, t1, p2, t2 = points_tangents
        distance = np.linalg.norm(p1 - p2)
        if len(params) == 2:
            control_points = [p1, p1 + params[0] * t1 * distance, p2 + params[1] * t2 * distance, p2]
        elif len(params) == 4:
            p3 = (p1 + p2) / 2
            t12 = (t1 + t2) / 2
            if np.linalg.norm(t12) < 1e-12: # If t1 // t2
                t12 = t1                    # Take t1
            if np.linalg.norm(np.cross((p2 - p1) / np.linalg.norm(p2 - p1), t12)) < 1e-12: # If t12 // (p2 - p1)
                t12 = np.random.rand(3) - 0.5  # Generate a random vector
            t12 = np.cross(np.cross(p2 - p1, t12), p2 - p1) # Take the orthogonal to p2 - p1 in the plane of t12
            t12 = t12 / np.linalg.norm(t12)
            # Normalize the axis of rotation
            axis = (p2 - p1) / np.linalg.norm(p2 - p1)
            # Compute the rotation matrix using Rodrigues' rotation formula
            angle = params[3]
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            # Rotate t3
            t3 = rotation_matrix @ t12
            control_points = [p1, p1 + params[0] * t1 * distance / 2, p3 + params[2] * t3 * distance / 2, p2 + params[1] * t2 * distance / 2, p2]
        elif len(params) == 6:
            p3 = p1 + (p2 - p1) / 3
            p4 = p1 + 2 * (p2 - p1) / 3
            t12 = (t1 + t2) / 2
            if np.linalg.norm(t12) < 1e-12: # If t1 // t2
                t12 = t1                    # Take t1
            if np.linalg.norm(np.cross((p2 - p1) / np.linalg.norm(p2 - p1), t12)) < 1e-12: # If t12 // (p2 - p1)
                t12 = np.random.rand(3) - 0.5  # Generate a random vector
            t12 = np.cross(np.cross(p2 - p1, t12), p2 - p1) # Take the orthogonal to p2 - p1 in the plane of t12
            t12 = t12 / np.linalg.norm(t12)
            # Normalize the axis of rotation
            axis = (p2 - p1) / np.linalg.norm(p2 - p1)
            # Compute the rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            angle = params[3]
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            # Rotate t3
            t3 = rotation_matrix @ t12
            angle = params[5]
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            # Rotate t4
            t4 = rotation_matrix @ t12
            control_points = [p1, p1 + params[0] * t1 * distance / 3, p3 + params[2] * t3 * distance / 2, p4 + params[4] * t4 * distance / 2, p2 + params[1] * t2 * distance / 3, p2]
        
        return control_points

class Helix(StrandsCollection):
    """
    A derived class representing a helix in 3D space.

    Attributes:
        N (int): Number of strands in the helix.
        radius (float): Radius of the helix.
        wavelength (float): Wavelength of the helix (distance for one full turn).
        height (float): Total height of the helix.
        points_per_strand (int): Number of points per strand.
    """

    def __init__(self, N, radius, wavelength, height, radius_field=None, wavelength_field=None, points_per_strand=15, newOxyz=None, strand_radius=None):
        super().__init__(radii=[[strand_radius] * points_per_strand] * N)
        self.N = N
        self.radius = radius
        self.radius_field = radius_field
        self.wavelength = wavelength
        self.wavelength_field = wavelength_field
        self.height = height
        self.points_per_strand = points_per_strand
        self.newOxyz = newOxyz
        self.strands = self._generate_helix()

    def _generate_helix(self):
        """
        Generate the strands of the helix.

        Returns:
            list: A list of arrays, each containing the (x, y, z) coordinates of a strand.
        """
        strands = []
        z = np.linspace(0, self.height, self.points_per_strand)

        if self.newOxyz is not None:
            new_centerline = np.column_stack(np.zeros((3, self.points_per_strand)))
            new_centerline[:, 2] = z
            new_centerline = StrandsCollection(new_centerline)
            new_centerline.rotate(np.eye(3), self.newOxyz.basis_vectors)
            new_centerline.translate(self.newOxyz.origin)

        wavelength = self.wavelength if self.wavelength_field is None else np.array([self.wavelength_field(new_centerline.strands[j][0], new_centerline.strands[j][1], new_centerline.strands[j][2], center=np.array([0,0,0])) for j in range(self.points_per_strand)])
        theta = 2 * np.pi * z / self.wavelength if self.wavelength_field is None else 2 * np.pi * z / abs(wavelength.dot(self.newOxyz.basis_vectors[:, 2]))

        for i in range(self.N):
            phase_shift = 2 * np.pi * i / self.N  # Phase shift for each strand
            radius = np.array([self.radius_field(new_centerline.strands[j][0], new_centerline.strands[j][1], new_centerline.strands[j][2], center=np.array([0,0,0])) for j in range(self.points_per_strand)]) if self.radius_field is not None else self.radius
            r = abs(radius.dot(self.newOxyz.basis_vectors[:, 2])) if self.radius_field is not None else self.radius
            x = r * np.cos(theta + phase_shift)
            y = r * np.sin(theta + phase_shift)
            strands.append(np.column_stack((x, y, z)))

        return strands
    
class ReferenceFrame(StrandsCollection):
    """
    A derived class representing a reference frame in 3D space.

    Attributes:
        origin (np.ndarray): The origin of the reference frame.
        basis_vectors (np.ndarray): A 3x3 matrix representing the basis vectors of the reference frame.
    """

    def __init__(self, origin=None, basis_vectors=None):
        super().__init__()
        self.origin = origin if origin is not None else np.zeros(3)
        self.basis_vectors = basis_vectors if basis_vectors is not None else np.eye(3)
        self.strands = self._generate_reference_frame()

    def _generate_reference_frame(self):
        """
        Generate the strands representing the reference frame (axes).

        Returns:
            list: A list of arrays, each containing the (x, y, z) coordinates of an axis.
        """
        strands = []
        axis_length = 1.0  # Default length of the axes

        for i in range(3):
            start = self.origin
            end = self.origin + axis_length * self.basis_vectors[:, i]
            strands.append(np.array([start, end]))

        return strands

    def set_origin(self, origin):
        self.origin = origin
        self.strands = self._generate_reference_frame()

    def set_basis_vectors(self, basis_vectors):
        self.basis_vectors = basis_vectors
        self.strands = self._generate_reference_frame()

    def align(self, point1, point2, e1 = None):
        """
        Generate a reference frame aligned with the direction given by two points.

        Parameters:
            point1 (np.ndarray): Coordinates of the first point (1D array of size 3).
            point2 (np.ndarray): Coordinates of the second point (1D array of size 3).
            e1 (np.ndarray, optional): An arbitrary vector (1D array of size 3) not parallel to the direction vector.

        Returns:
            ReferenceFrame: A ReferenceFrame object with the third axis aligned to the direction.
        """
        # Compute the vector from point1 to point2
        e3 = point2 - point1
        e3 = e3 / np.linalg.norm(e3)  # Normalize the direction vector

        # If no arbitrary vector is provided, choose one that is not parallel to the direction
        if e1 is None:
            if np.allclose(e3, [1, 0, 0]) or np.allclose(e3, [-1, 0, 0]):
                e1 = np.array([0, 1, 0])
            else:
                e1 = np.array([1, 0, 0])
        else:
            e1 = e1 / np.linalg.norm(e1)

        # Ensure the arbitrary vector is not parallel to the direction
        if np.allclose(np.cross(e3, e1), 0):
            raise ValueError("The arbitrary vector must not be parallel to the direction vector.")

        # Compute the second basis vector orthogonal to the direction
        e2 = np.cross(e3, e1)
        e2 = e2 / np.linalg.norm(e2)
        e1 = np.cross(e2, e3)

        # Combine the basis vectors into a 3x3 matrix
        target_frame = np.column_stack((e1, e2, e3))

        self.set_origin(point1)
        self.set_basis_vectors(target_frame)

    def plot(self, ax=None, scale=1.0, linewidth=1):
        """
        Override the base plot function to plot the axes in three different colors.

        Parameters:
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The axes to plot on. If None, a new figure and axes are created.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        colors = ['r', 'g', 'b']  # Colors for x, y, z axes

        for strand, color in zip(self.strands, colors):
            scaled_strand = strand.copy()
            scaled_strand[1] = scaled_strand[0] + (scaled_strand[1] - scaled_strand[0]) * scale  # Scale only the second row
            ax.plot(scaled_strand[:, 0], scaled_strand[:, 1], scaled_strand[:, 2], color=color, linewidth=linewidth)

        if ax.figure is not plt.gcf():  # Show the plot only if a new figure was created
            plt.show()


def generate_helix(N, radius, wavelength, height, points_per_strand=1000):
    """
    Generate a helix in 3D space with N strands.

    Parameters:
        N (int): Number of strands.
        radius (float): Radius of the helix.
        wavelength (float): Wavelength of the helix (distance for one full turn).
        height (float): Total height of the helix.
        points_per_strand (int): Number of points per strand.

    Returns:
        list: A list of N arrays, each containing the (x, y, z) coordinates of a strand.
    """
    strands = []
    z = np.linspace(0, height, points_per_strand)
    theta = 2 * np.pi * z / wavelength  # Angle as a function of height

    for i in range(N):
        phase_shift = 2 * np.pi * i / N  # Phase shift for each strand
        x = radius * np.cos(theta + phase_shift)
        y = radius * np.sin(theta + phase_shift)
        strands.append(np.column_stack((x, y, z)))

    # Add three more strands representing the x, y, z axes
    axis_length = height  # Use the height of the helix as the length of the axes
    x_axis = np.column_stack((np.linspace(0, 2 * axis_length, points_per_strand), np.zeros(points_per_strand), np.zeros(points_per_strand)))
    y_axis = np.column_stack((np.zeros(points_per_strand), np.linspace(0, 0.5 * axis_length, points_per_strand), np.zeros(points_per_strand)))
    z_axis = np.column_stack((np.zeros(points_per_strand), np.zeros(points_per_strand), np.linspace(0, axis_length, points_per_strand)))

    strands.extend([x_axis, y_axis, z_axis])

    return strands

def rotate_to_align(points, source_frame, target_frame):
    """
    Rotate points from a source frame to align with a target frame.

    Parameters:
        points (np.ndarray): Array of shape (n, 3) representing the points to rotate.
        source_frame (np.ndarray): Array of shape (3, 3) representing the source frame (orthonormal basis vectors).
        target_frame (np.ndarray): Array of shape (3, 3) representing the target frame (orthonormal basis vectors).

    Returns:
        np.ndarray: Rotated points in the target frame.
    """
    # Compute the rotation matrix to align source_frame with target_frame
    rotation_matrix = target_frame @ np.linalg.inv(source_frame)
    
    # Apply the rotation to the points
    rotated_points = points @ rotation_matrix.T
    
    return rotated_points

def translate(points, vector):
    """
    Translate points by a given vector.

    Parameters:
        points (np.ndarray): Array of shape (n, 3) representing the points to translate.
        vector (np.ndarray): Array of shape (3,) representing the translation vector.

    Returns:
        np.ndarray: Translated points.
    """
    return points + vector

def plot_helix(strands, ax=None):
    """
    Plot the helix strands in 3D space.

    Parameters:
        strands (list): A list of arrays, each containing the (x, y, z) coordinates of a strand.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The axes to plot on. If None, a new figure and axes are created.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for strand in strands:
        ax.plot(strand[:, 0], strand[:, 1], strand[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Set equal aspect ratio for all axes

    if ax.figure is not plt.gcf():  # Show the plot only if a new figure was created
        plt.show()

def compute_target_frame(point1, point2, arbitrary = None):
    """
    Compute a target frame (orthonormal basis) aligned with the vector between two points.

    Parameters:
        point1 (np.ndarray): Coordinates of the first point (1D array of size 3).
        point2 (np.ndarray): Coordinates of the second point (1D array of size 3).
        arbitrary (np.ndarray): An arbitrary vector (1D array of size 3) not parallel to the direction vector.

    Returns:
        np.ndarray: A 3x3 matrix representing the target frame (orthonormal basis vectors).
    """
    # Compute the vector from point1 to point2
    direction = point2 - point1
    direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

    # If no arbitrary vector is provided, choose one that is not parallel to the direction
    if arbitrary is None:
        if np.allclose(direction, [1, 0, 0]) or np.allclose(direction, [-1, 0, 0]):
            arbitrary = np.array([0, 1, 0])
        else:
            arbitrary = np.array([1, 0, 0])
    else:
        arbitrary = arbitrary / np.linalg.norm(arbitrary)

    # Ensure the arbitrary vector is not parallel to the direction
    if np.allclose(np.cross(direction, arbitrary), 0):
        raise ValueError("The arbitrary vector must not be parallel to the direction vector.")

    # Compute the second basis vector orthogonal to the direction
    orthogonal2 = np.cross(direction, arbitrary)
    orthogonal2 = orthogonal2 / np.linalg.norm(orthogonal2)
    orthogonal1 = np.cross(orthogonal2, direction)

    # Combine the basis vectors into a 3x3 matrix
    target_frame = np.column_stack((orthogonal1, orthogonal2, direction))

    return target_frame

def compute_target_frames(points, connectivities, types):
    """
    Compute target frames for each connectivity.

    Parameters:
        points (np.ndarray): Array of shape (n, 3) representing the points.
        connectivities (list of tuples): List of tuples where each tuple contains indices of connected points.
        types (list): List of types corresponding to each point.

    Returns:
        list: A list of 3x3 matrices representing the target frames for each connectivity.
    """
    target_frames = []
    arbitrary = np.array([1,1,1])

    for conn in connectivities:
        # Determine the first point based on type (port or counterport)
        if types[conn[0]] in ["port", "counterport"]:
            point1 = points[conn[0]]
            point2 = points[conn[1]]
        else:
            point1 = points[conn[1]]
            point2 = points[conn[0]]

        # Compute the target frame for the connectivity
        target_frame = compute_target_frame(point1, point2, arbitrary=arbitrary if types[conn[0]] == "port" or types[conn[1]] == "port" else arbitrary)
        target_frames.append(target_frame)

    return target_frames

def compute_translation_vectors(points, connectivities, types):
    """
    Compute translation vectors for each connectivity.

    Parameters:
        points (np.ndarray): Array of shape (n, 3) representing the points.
        connectivities (list of tuples): List of tuples where each tuple contains indices of connected points.
        types (list): List of types corresponding to each point.

    Returns:
        list: A list of translation vectors for each connectivity.
    """
    translation_vectors = []

    for conn in connectivities:
        # Determine the point to use as the translation vector based on type (port or counterport)
        if types[conn[0]] in ["port", "counterport"]:
            translation_vector = points[conn[0]]
        else:
            translation_vector = points[conn[1]]

        translation_vectors.append(translation_vector)

    return translation_vectors

# Example usage
if __name__ == "__main__":

    solution = 0
    permuted_solution = 0

    lattice = WovenLattice('lattice_cubic.dat', N=2, strand_radius=0.025, helix_radius=0.25, helix_wavelength=2, height=0.25)
    lattice.generate_topology(conjugacy=True, permuted_solutions=permuted_solution, max_multiplicity=None, stop_early=False, discard_threshold=None)
    lattice.generate_geometry(solution=solution, permuted_solution=permuted_solution, expanded=False, bezier_min_degree=3, bezier_max_degree=5, sequential=1)

    lattice.plot_vedo(color_map='color', viewup='z')