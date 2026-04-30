import time
import trimesh
import vtk
import numpy as np
from itertools import product
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys
import os

def read_network_data(filename):
    points = []
    connectivities = []
    with open(filename, 'r') as file:
        # Read points
        line = file.readline().strip()
        while line and not line.startswith("connectivity"):
            points.append([float(x) for x in line.split(',')])
            line = file.readline().strip()

        # Read connectivity
        if line.startswith("connectivity"):
            line = file.readline().strip()
            while line:
                connectivities.append([int(x) for x in line.split(',')])
                line = file.readline().strip()
    return np.array(points), connectivities

def write_dat_file(points, lines, output_filename):
    with open(output_filename, 'w') as file:
        # Write points
        for point in points:
            file.write(','.join(map(str, point)) + '\n')
        
        # Write connectivity
        file.write("connectivity\n")
        for line in lines:
            file.write(','.join(map(str, line)) + '\n')

def mesh2dat(file_path, nodes, connectivities):
    with open(file_path, 'w') as file:
        # Write nodes
        for node in nodes:
            file.write(f"{node[0]:.8f},{node[1]:.8f},{node[2]:.8f}\n")
        
        # Write connectivity header
        file.write("connectivity\n")
        
        # Write connectivities
        for conn in connectivities:
            file.write(f"{conn[0]},{conn[1]}\n")

# Coarsen the mesh (tested for intertwined networks)
def simplify_network(points, lines, distance_threshold=0.1, curvature_threshold=0.4):
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def curvature(p1, p2, p3):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

    simplified_lines = []
    used_points = set()

    for i in range(len(lines) - 1):
        if i % 2 == 1:
            continue
        p1 = points[lines[i][0]]
        p2 = points[lines[i][1]]
        if np.array_equal(p2, points[lines[i + 1][0]]):
            p3 = points[lines[i + 1][1]]
            if distance(p2, p3) < distance_threshold and curvature(p1, p2, p3) < curvature_threshold:
                simplified_lines.append([lines[i][0], lines[i + 1][1]])
                used_points.update([lines[i][0], lines[i + 1][1]])
            else:
                simplified_lines.append(lines[i])
                simplified_lines.append(lines[i + 1])
                used_points.update([lines[i][0], lines[i][1], lines[i + 1][0], lines[i + 1][1]])
        else:
            simplified_lines.append(lines[i])
            simplified_lines.append(lines[i + 1])
            used_points.update([lines[i][0], lines[i][1], lines[i + 1][0], lines[i + 1][1]])
    if len(lines) % 2 == 1:
        simplified_lines.append(lines[-1])
        used_points.update([lines[-1][0], lines[-1][1]])

    # Reindex points
    used_points = sorted(used_points)
    point_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_points)}
    simplified_points = [points[idx] for idx in used_points]
    reindexed_lines = [[point_map[start], point_map[end]] for start, end in simplified_lines]

    return simplified_points, reindexed_lines

def refine_mesh(input_file, output_file, refinement_number, threshold, radii_input_file=None, radii_output_file=None):
    """
    Refines a beam mesh based on refinement number and length threshold.

    Parameters:
    - input_file: str, path to the input mesh file.
    - output_file: str, path to save the refined mesh file.
    - refinement_number: int, number of subdivisions for each beam.
    - threshold: float, minimum beam length for refinement.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse nodes
    nodes = []
    connectivity = []
    radii = []
    reading_connectivity = False
    for line in lines:
        line = line.strip()
        if line == "connectivity":
            reading_connectivity = True
            continue
        if reading_connectivity:
            connectivity.append(tuple(map(int, line.split(','))))
        else:
            nodes.append(list(map(float, line.split(','))))
    # Read radii from file
    if radii_input_file:
        with open(radii_input_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            radii.append(float(line.strip()))
    
    nodes = np.array(nodes)
    new_nodes = nodes.tolist()
    new_connectivity = []
    node_count = len(nodes)
    new_radii = []

    # Process each beam
    for element_id, conn in enumerate(connectivity):
        start_idx, end_idx = conn
        start_node = nodes[start_idx]
        end_node = nodes[end_idx]

        # Compute the length of the beam
        length = np.linalg.norm(end_node - start_node)

        if length > threshold and refinement_number > 1:
            if np.abs(np.dot((start_node - end_node)/length, np.array([0, 0, 1]))) > 0.95:
                refinement_number_ = 4
            elif length > 5 and np.abs(np.dot((start_node - end_node)/length, np.array([0, 0, 1]))) > 0.1:
                refinement_number_ = 4
            else:
                refinement_number_ = refinement_number
            # Subdivide the beam
            for i in range(1, refinement_number_):
                t = i / refinement_number_
                new_node = (1 - t) * start_node + t * end_node
                new_nodes.append(new_node.tolist())
                if i == 1:
                    new_connectivity.append((start_idx, node_count))
                else:
                    new_connectivity.append((node_count - 1, node_count))
                if radii_input_file:
                    new_radii.append(radii[element_id])
                node_count += 1
            new_connectivity.append((node_count - 1, end_idx))
            if radii_input_file:
                new_radii.append(radii[element_id])
        else:
            # Keep the beam unchanged
            new_connectivity.append(conn)
            if radii_input_file:
                new_radii.append(radii[element_id])

    # Write the refined mesh to output file
    with open(output_file, 'w') as f:
        for node in new_nodes:
            f.write(','.join(map(str, node)) + '\n')
        f.write('connectivity\n')
        for conn in new_connectivity:
            f.write(','.join(map(str, conn)) + '\n')
    # Write the radii to output file
    if radii_output_file:
        with open(radii_output_file, 'w') as f:
            for r in new_radii:
                f.write(str(r) + '\n')

def tessellate_space(points, connectivities, N=(1, 1, 1), vectors=None):
    tessellated_points = points.tolist()
    tessellated_connectivities = []
    vectors = vectors if vectors is not None else np.eye(3)
    
    # Generate all possible multiplicities
    multiplicities = list(product(range(N[0]), range(N[1]), range(N[2])))
    print(multiplicities)
    point_index_map = {tuple(p): i for i, p in enumerate(tessellated_points)}
    
    for mult in multiplicities:
        v = np.zeros(3)
        for i, m in enumerate(mult):
            v += m * vectors[i]
        translated_points = points + v
        
        for tp in translated_points:
            tp_tuple = tuple(tp)
            if tp_tuple not in point_index_map:
                point_index_map[tp_tuple] = len(tessellated_points)
                tessellated_points.append(tp.tolist())
        
        tessellated_connectivities.extend(
            [[point_index_map[tuple(points[start] + v)], point_index_map[tuple(points[end] + v)]]
             for start, end in connectivities]
        )
    
    return np.array(tessellated_points), tessellated_connectivities

# Function to write VTK file with node and element IDs
def write_vtk_file(points, lines, output_filename):
    # Create a vtkPoints object to store the points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)

    # Create a vtkCellArray to store the lines (edges)
    vtk_lines = vtk.vtkCellArray()
    for line in lines:
        vtk_line = vtk.vtkLine()
        vtk_line.GetPointIds().SetId(0, line[0])
        vtk_line.GetPointIds().SetId(1, line[1])
        vtk_lines.InsertNextCell(vtk_line)

    # Create a vtkPolyData object to store the data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    # Add node IDs (point data)
    node_ids = vtk.vtkIntArray()
    node_ids.SetName("NodeID")  # Set the name for the point data
    for i in range(len(points)):
        node_ids.InsertNextValue(i)
    poly_data.GetPointData().AddArray(node_ids)

    # Add element IDs (line data)
    element_ids = vtk.vtkIntArray()
    element_ids.SetName("ElementID")  # Set the name for the cell data
    for i in range(len(lines)):
        element_ids.InsertNextValue(i)
    poly_data.GetCellData().AddArray(element_ids)

    # Write the VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()

# Function to write VTK file with node and element IDs
def write_vtk_file_intertwined(points, lines, output_filename, strand_id=None):

    def extract_chains(connectivities):
        """
        Extract ordered chains from a list of connectivity pairs.

        Parameters:
        - connectivities: List of pairs of IDs representing the connections.

        Returns:
        - List of lists, where each sublist represents an ordered chain of IDs.
        """
        from collections import defaultdict

        # Step 1: Build a dictionary of connections
        connections = defaultdict(list)
        for a, b in connectivities:
            connections[a].append(b)
            connections[b].append(a)

        # Step 2: Initialize variables
        chains = []
        visited = set()

        # Step 3: Traverse to form chains
        def build_chain(start_id):
            chain = []
            current = start_id
            prev = None

            while True:
                chain.append(current)
                visited.add(current)

                # Find the next ID in the chain
                neighbors = [n for n in connections[current] if n != prev]
                if not neighbors:  # No more neighbors, chain ends
                    break
                next_id = neighbors[0]
                prev, current = current, next_id

                # Check for a closed chain
                if current == start_id:
                    chain.append(current)  # Close the loop
                    break

            return chain

        # Iterate over all IDs to find chains
        for id_ in connections:
            if id_ not in visited:
                chain = build_chain(id_)
                chains.append(chain)

        # Sort chains by length
        chains.sort(key=len)

        return chains

    # Extract chains from the connectivity data
    chains = extract_chains(lines)

    # Filter chains based on strand_ids if provided
    if strand_id is not None:
        chains = [chains[i] for i in strand_id if i < len(chains)]

    # Collect unique points involved in the chains
    involved_points = set()
    for chain in chains:
        involved_points.update(chain)

    # Create a vtkPoints object to store the points
    vtk_points = vtk.vtkPoints()
    point_map = {}
    for new_id, old_id in enumerate(sorted(involved_points)):
        vtk_points.InsertNextPoint(points[old_id])
        point_map[old_id] = new_id

    # Flatten the chains into lines
    filtered_lines = []
    for chain in chains:
        for i in range(len(chain) - 1):
            filtered_lines.append([point_map[chain[i]], point_map[chain[i + 1]]])

    # Create a vtkCellArray to store the lines (edges)
    vtk_lines = vtk.vtkCellArray()
    for line in filtered_lines:
        vtk_line = vtk.vtkLine()
        vtk_line.GetPointIds().SetId(0, line[0])
        vtk_line.GetPointIds().SetId(1, line[1])
        vtk_lines.InsertNextCell(vtk_line)

    # Create a vtkPolyData object to store the data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    # Add node IDs (point data)
    node_ids = vtk.vtkIntArray()
    node_ids.SetName("NodeID")  # Set the name for the point data
    for i in range(vtk_points.GetNumberOfPoints()):
        node_ids.InsertNextValue(i)
    poly_data.GetPointData().AddArray(node_ids)

    # Add element IDs (line data)
    element_ids = vtk.vtkIntArray()
    element_ids.SetName("ElementID")  # Set the name for the cell data
    for i in range(len(filtered_lines)):
        element_ids.InsertNextValue(i)
    poly_data.GetCellData().AddArray(element_ids)

    # Write the VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()


def union_subset(subset):
    """Helper function for computing union on a subset of meshes."""
    return trimesh.boolean.union(subset)

# For general truss networks
def dat2stl(input_file, sphere_radius=0.025, cylinder_radius=0.025, output_file="merged_geometry.stl", percent=0.3, sections=12, num_threads=os.cpu_count()):
    # Read the points and connectivity from the file
    points = []
    connectivity = []
    
    # Read input data
    with open(input_file, 'r') as file:
        lines = file.readlines()
        reading_points = True
        for line in lines:
            if line.strip() == "connectivity":
                reading_points = False
                continue
            if reading_points:
                points.append([float(coord) for coord in line.split(',')])
            else:
                connectivity.append([int(index) for index in line.split(',')])
    
    points = np.array(points)
    
    # Create a list to hold the individual meshes (spheres and cylinders)
    meshes = []

    def add_sphere(point):
        sphere_mesh = trimesh.primitives.Sphere(radius=sphere_radius, center=point, subdivisions=sections//12+1)
        return sphere_mesh

    def add_cylinder(conn):
        point1 = points[conn[0]]
        point2 = points[conn[1]]
        
        # Compute the direction vector and length of the cylinder
        direction = point2 - point1
        length = np.linalg.norm(direction)
        direction /= length  # Normalize the direction vector
        
        # Create the cylinder between the two points
        cylinder_mesh = trimesh.creation.cylinder(radius=cylinder_radius, height=length, sections=sections)
        
        # Align the cylinder along the direction vector
        transform_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
        
        # Apply rotation and translation to place the cylinder
        cylinder_mesh.apply_transform(transform_matrix)
        cylinder_mesh.apply_translation((point1 + point2) / 2)  # Move to midpoint

        return cylinder_mesh

    # Use ThreadPoolExecutor to parallelize the creation of spheres and cylinders
    start_time = time.time()
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     # Print the number of threads being used
    #     print(f"Using {num_threads} threads")

    #     sphere_futures = [executor.submit(add_sphere, point) for point in points]
    #     cylinder_futures = [executor.submit(add_cylinder, conn) for conn in connectivity]

    #     for future in sphere_futures:
    #         meshes.append(future.result())

    #     for future in cylinder_futures:
    #         meshes.append(future.result())
    for point in points:
        meshes.append(add_sphere(point))
    for conn in connectivity:
        meshes.append(add_cylinder(conn))
    end_time = time.time()
    print(f"Time taken to create spheres and cylinders: {end_time - start_time} seconds")



    def parallel_union(meshes, num_processes=4):
        """Parallelize the union operation over a large number of meshes."""
        # Split the meshes into subsets
        chunk_size = len(meshes) // num_processes
        subsets = [meshes[i:i + chunk_size] for i in range(0, len(meshes), chunk_size)]
        
        # Use ProcessPoolExecutor to compute unions in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(union_subset, subsets))

        final_result = trimesh.boolean.union(results)

        return final_result

    # Attempt to merge all watertight meshes
    start_time = time.time()
    combined_mesh = trimesh.boolean.union(meshes, engine='manifold')
    # combined_mesh = parallel_union(meshes, num_processes=num_threads)
    end_time = time.time()
    print(f"Time taken to merge meshes: {end_time - start_time} seconds")

    # Export the merged mesh to STL
    start_time = time.time()

    # combined_mesh = combined_mesh.simplify_quadric_decimation(percent=percent)
    print('volume =',combined_mesh.volume)

    combined_mesh.export(output_file)
    
    # stl_data = trimesh.exchange.stl.export_stl(combined_mesh)
    # # Save the STL to a file
    # with open(output_file, "wb") as f:
    #     f.write(stl_data)

    end_time = time.time()
    print(f"Time taken to export STL: {end_time - start_time} seconds")
    print(f"Merged STL file saved as {output_file}")

# For intertwined networks
def dat2stl_light(input_file, sphere_cap=False, sphere_radius=0.025, cylinder_radius=0.025, sections=12, output_file="merged_geometry.stl", strand_id=None):
    # Function to read points and connectivity from file
    def read_network_data(filename):
        points = []
        connectivities = []
        with open(filename, 'r') as file:
            # Read points
            line = file.readline().strip()
            while line and not line.startswith("connectivity"):
                points.append([float(x) for x in line.split(',')])
                line = file.readline().strip()

            # Read connectivity
            if line.startswith("connectivity"):
                line = file.readline().strip()
                while line:
                    connectivities.append([int(x) for x in line.split(',')])
                    line = file.readline().strip()

        return points, connectivities
    
    def extract_chains(connectivities):
        """
        Extract ordered chains from a list of connectivity pairs.

        Parameters:
        - connectivities: List of pairs of IDs representing the connections.

        Returns:
        - List of lists, where each sublist represents an ordered chain of IDs.
        """
        from collections import defaultdict

        # Step 1: Build a dictionary of connections
        connections = defaultdict(list)
        for a, b in connectivities:
            connections[a].append(b)
            connections[b].append(a)

        # Step 2: Initialize variables
        chains = []
        visited = set()

        # Step 3: Traverse to form chains
        def build_chain(start_id):
            chain = []
            current = start_id
            prev = None

            while True:
                chain.append(current)
                visited.add(current)

                # Find the next ID in the chain
                neighbors = [n for n in connections[current] if n != prev]
                if not neighbors:  # No more neighbors, chain ends
                    break
                next_id = neighbors[0]
                prev, current = current, next_id

                # Check for a closed chain
                if current == start_id:
                    chain.append(current)  # Close the loop
                    break

            return chain

        # Iterate over all IDs to find chains
        for id_ in connections:
            if id_ not in visited:
                chain = build_chain(id_)
                chains.append(chain)

        # Sort chains by length
        chains.sort(key=len)

        return chains

    def extract_strands(points, connectivities):
        chains = extract_chains(connectivities)
        strands = [[points[i] for i in chain] for chain in chains]
        # for strand in strands:
        #     strand.pop(-1)
        return strands
        # def find_chain(start, visited):
        #     chain = []
        #     stack = [start]
        #     while stack:
        #         current = stack.pop()
        #         if current not in visited:
        #             visited.add(current)
        #             chain.append(points[current])
        #             for conn in connectivities:
        #                 if conn[0] == current and conn[1] not in visited:
        #                     stack.append(conn[1])
        #                 elif conn[1] == current and conn[0] not in visited:
        #                     stack.append(conn[0])
        #     return chain
        
        # visited = set()
        # strands = []
        # for i in range(len(points)):
        #     if i not in visited:
        #         strand = find_chain(i, visited)
        #         if strand:
        #             strands.append(strand)
        # return strands
    
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
        # vertices = np.array(vertices)
        # print(len(vertices))
        # print(vertices)
        # print(len(faces))
        # faces = np.array(faces)
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

    # Read the points and connectivity from the file
    start_time = time.time()
    points, connectivities = read_network_data(input_file)
    strands = extract_strands(points, connectivities)
    end_time = time.time()
    print(f"Time taken to read and extract strands: {end_time - start_time} seconds")
    start_time = time.time()
    meshes = []
    for i, strand in enumerate(strands):
        if strand_id is not None:
            if i not in strand_id:
                continue
        meshes.extend(curve_to_tube(strand, radius=cylinder_radius))
        if sphere_cap:
            meshes.append(trimesh.primitives.Sphere(radius=sphere_radius, center=strand[0], subdivisions=sections//12+1))
            meshes.append(trimesh.primitives.Sphere(radius=sphere_radius, center=strand[-1], subdivisions=sections//12+1))
        # print("created strand")
    end_time = time.time()
    print(f"Time taken to create mesh: {end_time - start_time} seconds")
    if meshes == []:
        print("No strand found")
        sys.exit()
    
    # Export to STL
    start_time = time.time()
    # combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh = trimesh.boolean.union(meshes, engine='manifold', check_volume=True) # turn off check_volume to speed up
    end_time = time.time()
    print(f"Time taken to merge mesh: {end_time - start_time} seconds")
    start_time = time.time()
    combined_mesh.export(output_file)
    print('volume =',combined_mesh.volume)
    end_time = time.time()
    print(f"Time taken to export STL: {end_time - start_time} seconds")
    print(f"Lightweight tube STL file saved as {output_file}")

def plot_lattice(points, connectivity, point_types=False):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    s_p = 50
    s_cp = 50
    j_num = 0
    p_num = 0
    cp_num = 0
    shift = 0.1
    if point_types is True:
        for point, point_type in zip(points, point_types):
            if point_type == 'port':
                ax.scatter(point[0], point[1], point[2], c='r', marker='o', s=s_p)
                ax.text(point[0] + shift, point[1] + shift, point[2], str(p_num), color='black')
                p_num += 1
                s_p *= 2
            elif point_type == 'counterport':
                ax.scatter(point[0], point[1], point[2], c='b', marker='o', s=s_cp)
                ax.text(point[0] + shift, point[1] + shift, point[2], str(cp_num), color='black')
                cp_num += 1
                s_cp *= 2
            elif point_type == 'junction':
                ax.scatter(point[0], point[1], point[2], c='g', marker='s', s=200)
                ax.text(point[0] + shift, point[1] + shift, point[2], str(j_num), color='black')
                j_num += 1
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='o')

    # Plot lines based on connectivity
    lines = [(points[start], points[end]) for start, end in connectivity]
    line_collection = Line3DCollection(lines, colors='k', linewidths=2)
    ax.add_collection3d(line_collection)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio

    # ax.set_box_aspect([np.ptp(points[:, axis]) for axis in range(3)])
    ax.set_box_aspect([1, 1, 1])

    # Show plot
    plt.show()

if __name__ == "__main__":
    # dat2stl("cuboctahedron.dat")
    points, connections = read_network_data("lattice_cubic.dat")
    refine_mesh("lattice_cubic.dat", "lattice_cubic_refined.dat", refinement_number=4, threshold=0.01)
    points, connections = read_network_data("lattice_cubic_refined.dat")
    # points, connections = simplify_network(points, connections, distance_threshold=0.3, curvature_threshold=1)
    # points, connections = tessellate_space(points, connections, N=(2, 2, 2), vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    # write_vtk_file(points, connections, "temp.vtk")
    plot_lattice(points, connections)
    # write_dat_file(points, connections, "lattice_cubic_tessellated.dat")