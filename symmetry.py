import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from itertools import product
from collections import defaultdict
from export import *


def read_lattice_file(filename, types_and_status=True):
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract points and connectivity
    points = []
    connectivity = []
    reading_points = True
    reading_counterports = False
    counterports = []

    for line in lines:
        if line.strip() == 'connectivity':
            reading_points = False
            continue
        if line.strip() == 'counterports':
            reading_counterports = True
            continue
        if reading_points:
            points.append(list(map(float, line.strip().split(','))))
        elif reading_counterports:
            counterports.append(list(map(int, line.strip().split(','))))
        else:
            connectivity.append(list(map(int, line.strip().split(','))))

    points = np.array(points)

    if not types_and_status:
        return points, connectivity

    # Determine the type of each point
    point_types = ['port'] * len(points)
    point_status = ['active'] * len(points)  # Initialize all points as active
    if reading_counterports:
        for counterport in counterports:
            point_types[counterport[1]] = 'counterport'
    connection_count = np.zeros(len(points), dtype=int)

    for start, end in connectivity:
        connection_count[start] += 1
        connection_count[end] += 1

    for i, count in enumerate(connection_count):
        if count > 1:
            point_types[i] = 'junction'

    # This is another definition based on central symmetry for ports-counterports
    center = np.mean(points, axis=0)
    def are_counterports(p,q):
        center = np.mean(points, axis=0)
        return np.linalg.norm(p - center + q - center) < 1e-6
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    # def are_counterports(p,q):
    #     positive_delta = [np.abs(p[i] - q[i]) for i in range(3)]
    #     for i in range(3):
    #         if max_coords[i] - min_coords[i] < 1e-6:
    #             # Dimension is degenerate, skip check
    #             continue
    #         if positive_delta[i] < 1e-6:
    #             if p[i] < min_coords[i] + 1e-6 or p[i] > max_coords[i] - 1e-6:
    #                 return False
    #             # No need to check on q since delta is zero
    #         else:
    #             if np.abs(positive_delta[i] - (max_coords[i] - min_coords[i])) > 1e-6:
    #                 return False
    #     return True

        

    P = point_types.count('port') + point_types.count('counterport')
    if P % 2 != 0:
        PP = P + 1
    else:
        PP = P
    ports = []
    if not reading_counterports:
        counterports = [np.zeros(3) for _ in range(PP // 2)]
    junctions = []
    idxs = []


    for i, point in enumerate(points):
        if point_types[i] == 'junction':
            junctions.append(point)
            idxs.append(P + len(junctions) - 1)
        if point_types[i] == 'port':
            found = False
            if not reading_counterports:
                for j, port in enumerate(ports):
                    if are_counterports(port, point):
                        counterports[j] = point
                        idxs.append(PP // 2 + j)
                        found = True
                        break
            if not found:
                ports.append(point)
                idxs.append(len(ports) - 1)
        if point_types[i] == 'counterport':
            idxs.append(-1)

    if reading_counterports:
        reading = 0
        for i, point in enumerate(points):
            if point_types[i] == 'counterport':
                idxs[i] = (PP // 2 + idxs[counterports[reading][0]])
                reading += 1    


    connectivity = [[idxs[start], idxs[end]] for start, end in connectivity]
    if reading_counterports:
        counterports = [points[i] for i in range (P + len(junctions)) if idxs[i] >= PP // 2 and idxs[i] < P]

    points = np.concatenate((ports, counterports, junctions))
    point_types = ['port'] * (PP // 2) + ['counterport'] * (P // 2) + ['junction'] * (len(points) - P)

    # Check if each point is contained in connectivity, if not, change its point_type to 'suppressed'
    connected_points = set([idx for edge in connectivity for idx in edge])
    for i in range(len(point_types)):
        if i not in connected_points:
            point_status[i] = 'suppressed'

    return points, connectivity, point_types, point_status

def compute_bounding_planes(points, shift=np.array([0.0, 0.0, 0.0])):
    """
    Compute the six axis-aligned planes that bound the point cloud.
    Returns a list of plane equations (normal vector, offset) for each face:
    (normal, d) where the plane equation is normal . x = d.
    If the extent along a dimension is zero, do not append planes for that dimension.
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    planes = []
    # x planes
    if not np.isclose(max_coords[0] - min_coords[0], 0):
        planes.append((np.array([1, 0, 0]), min_coords[0] + shift[0]))
        planes.append((np.array([-1, 0, 0]), -max_coords[0] - shift[0]))
    # y planes
    if not np.isclose(max_coords[1] - min_coords[1], 0):
        planes.append((np.array([0, 1, 0]), min_coords[1] + shift[1]))
        planes.append((np.array([0, -1, 0]), -max_coords[1] - shift[1]))
    # z planes
    if not np.isclose(max_coords[2] - min_coords[2], 0):
        planes.append((np.array([0, 0, 1]), min_coords[2] + shift[2]))
        planes.append((np.array([0, 0, -1]), -max_coords[2] - shift[2]))
    return planes

def compute_translation_vectors(min_coords, max_coords):
    """
    Compute the translation vectors based on the bounding cuboid.
    Returns the nonzero vectors along x, y, and z axes.
    """
    translation_vectors = [
        np.array([max_coords[0] - min_coords[0], 0, 0]),
        np.array([max_coords[0] - min_coords[0], 0, 0]),
        np.array([0, max_coords[1] - min_coords[1], 0]),
        np.array([0, max_coords[1] - min_coords[1], 0]),
        np.array([0, 0, max_coords[2] - min_coords[2]]),
        np.array([0, 0, max_coords[2] - min_coords[2]])
    ]
    # Only keep nonzero vectors
    nonzero_vectors = [vec for vec in translation_vectors if not np.allclose(vec, 0)]
    return nonzero_vectors

def reframe_lattice(points, connectivity, translation_vector, tolerance=1e-8):
    """
    """
    def process_connectivity(ps, conns, plane, translation):
        normal, d = plane
        new_conns = []
        for conn in conns:
            p1, p2 = ps[conn[0]], ps[conn[1]]
            if np.dot(normal, p1) < d:
                # p1 is outside the bounding box on this side
                if np.dot(normal, p2) < d:
                    # Both points are outside the bounding box on this side
                    ps[conn[0]] = p1 + translation
                    ps[conn[1]] = p2 + translation
                    new_conns.append(conn)
                else:
                    # Only p1 is outside
                    # Find intersection of segment with plane
                    direction = p2 - p1
                    denom = np.dot(normal, direction)
                    t = (d - np.dot(normal, p1)) / denom
                    intersection = p1 + t * direction
                    if np.allclose(intersection, p2, atol=tolerance):
                        ps[conn[0]] = None # Remove p1
                        ps[conn[1]] = None # Remove p2
                        ps.append(p1 + translation) # Add translated p1
                        ps.append(intersection + translation) # Add translated intersection
                        new_conns.append([len(ps) - 2, len(ps) - 1]) # Connection from translated p1 to translated intersection
                    else:
                        ps[conn[0]] = intersection # Update p1 to intersection
                        ps.append(p1 + translation) # Add translated p1
                        ps.append(intersection + translation) # Add translated intersection
                        new_conns.append(conn) # Connection from intersection to p2
                        new_conns.append([len(ps) - 2, len(ps) - 1]) # Connection from translated p1 to translated intersection
            else:
                # p1 is inside the bounding box on this side
                if np.dot(normal, p2) < d:
                    # Only p2 is outside
                    direction = p1 - p2
                    denom = np.dot(normal, direction)
                    t = (d - np.dot(normal, p2)) / denom
                    intersection = p2 + t * direction
                    if np.allclose(intersection, p1, atol=tolerance):
                        ps[conn[0]] = None # Remove p1
                        ps[conn[1]] = None # Remove p2
                        ps.append(p2 + translation) # Add translated p2
                        ps.append(intersection + translation) # Add translated intersection
                        new_conns.append([len(ps) - 2, len(ps) - 1]) # Connection from translated p2 to translated intersection
                    else:
                        ps[conn[1]] = intersection
                        ps.append(p2 + translation)
                        ps.append(intersection + translation)
                        new_conns.append(conn)
                        new_conns.append([len(ps) - 2, len(ps) - 1])
                else:
                    # Both points are inside
                    new_conns.append(conn)
        return ps, new_conns
    
    shifted_bounding_planes = compute_bounding_planes(points, shift=translation_vector)
    temp_connectivity = [connectivity[0]]
    temp_points = [points[connectivity[0][0]], points[connectivity[0][1]]]
    translations = compute_translation_vectors(np.min(points, axis=0), np.max(points, axis=0))
    
    new_points = []
    new_connectivity = []

    for conn in connectivity:
        temp_conn = [[0,1]]
        temp_points = [points[conn[0]], points[conn[1]]]
        for plane, translation in zip(shifted_bounding_planes, translations):
            temp_points, temp_conn = process_connectivity(temp_points, temp_conn, plane, translation)

        # Add temp_points to new_points if not already present, and update temp_conn indices accordingly
        indices = []
        for pt in temp_points:
            found = False
            if pt is None:
                found = True
                indices.append(-1)  # Mark for removal
                continue
            for i, npt in enumerate(new_points):
                if np.allclose(pt, npt, atol=tolerance):
                    indices.append(i)
                    found = True
                    break
            if not found:
                new_points.append(pt)
                indices.append(len(new_points) - 1)
        # Add temp_conn (which are pairs of indices) to new_connectivity
        for conn_pair in temp_conn:
            # conn_pair may be a tuple or list of two indices
            idx0 = indices[conn_pair[0]]
            idx1 = indices[conn_pair[1]]
            if [idx0, idx1] not in new_connectivity and [idx1, idx0] not in new_connectivity:
                new_connectivity.append([idx0, idx1])
    
    return np.array(new_points), new_connectivity


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
    if point_types:
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

def find_symmetries(points, point_status=None):

    if point_status is None:
        point_status = ['active'] * len(points)

    def compute_angle(p1, p2, p3):
        """Compute angle at p2 formed by vectors p1->p2 and p2->p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return round(np.arccos(np.clip(cos_theta, -1, 1)), 4)  # Clip for numerical stability

    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    # G.add_edges_from(connectivity)

    # Connect all nodes to ensure the graph is fully connected
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if point_status[i] == 'suppressed' or point_status[j] == 'suppressed':
                G.add_edge(i, j, weight=0)  # Suppressed points have no edges
            else:
                G.add_edge(i, j, weight=np.linalg.norm(points[i] - points[j]))

    # Custom edge comparison function: ensures same length
    def edge_match(e1, e2):
        return np.isclose(e1['weight'], e2['weight'])

    # Custom node match function (must match angles)
    def node_match(n1, n2):
        return n1.get("angles", {}) == n2.get("angles", {})
    
    # # checking what g is
    # print("Graph G:", G.nodes(data=True))
    # print("Graph G edges:", G.edges(data=True))

    # Find isomorphisms preserving edge lengths and angles
    GM = GraphMatcher(G, G, edge_match=edge_match)

   
    valid_permutations = list(GM.isomorphisms_iter())
    
    # # checking what valid_permutations has
    # print("Valid permutations (as node mappings):", valid_permutations)

    # Convert valid permutations to the desired format
    formatted_permutations = []
    for perm in valid_permutations:
        formatted_perm = [0] * len(points)
        for key, value in perm.items():
            formatted_perm[key] = value
        formatted_permutations.append(formatted_perm)

    return formatted_permutations

def find_symmetries_v1(points, connectivity, point_status=None):

    if point_status is None:
        point_status = ['active'] * len(points)

    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))

    # Find all rotations and mirrors for the whole cloud of points
    # that preserve the spatial structure of points

    # Connect all nodes to ensure the graph is fully connected
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if point_status[i] == 'suppressed' or point_status[j] == 'suppressed':
                G.add_edge(i, j, weight=0)  # Suppressed points have no edges
            else:
                G.add_edge(i, j, weight=np.linalg.norm(points[i] - points[j]))

    # Custom edge comparison function: ensures same length
    def edge_match(e1, e2):
        return np.isclose(e1['weight'], e2['weight'])

    # Find isomorphisms preserving edge lengths
    GM = GraphMatcher(G, G, edge_match=edge_match)
    valid_permutations = list(GM.isomorphisms_iter())

    # Amongst the valid permutations find those that preserve the connectivity
    edges_set = set(tuple(sorted(edge)) for edge in connectivity)
    valid_permutations = [
        perm for perm in valid_permutations
        if set(tuple(sorted((perm[start], perm[end]))) for start, end in connectivity) == edges_set
    ]

    # Convert valid permutations to the desired format
    formatted_permutations = []
    for perm in valid_permutations:
        formatted_perm = [0] * len(points)
        for key, value in perm.items():
            formatted_perm[key] = value
        formatted_permutations.append(formatted_perm)

    return formatted_permutations

def generate_counterports(types):
    counterports = {}
    P = sum([1 for t in types if t == 'port'])
    for i, t in enumerate(types):
        if t == 'port':
            counterports[i] = i + P
        elif t == 'counterport':
            counterports[i] = i - P
        # Junctions are not considered counterports (for conjugacy type 0)
        # elif t == 'junction':
        #     counterports[i] = i
    return counterports

def tessellate_space(points, connectivities, point_types, N):
    tessellated_points = points.tolist()
    tessellated_connectivities = []
    vectors = []
    counterports = generate_counterports(point_types)
    
    for ip, (p, t) in enumerate(zip(points, point_types)):
        if t == 'port':
            icp = counterports[ip]
            cp = points[icp]
            vectors.append(cp - p)
    
    # Generate all possible multiplicities
    multiplicities = list(product(range(N), repeat=len(vectors)))
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
        
        base_index = len(tessellated_points) - len(translated_points)
        tessellated_connectivities.extend(
            [[point_index_map[tuple(points[start] + v)], point_index_map[tuple(points[end] + v)]]
             for start, end in connectivities]
        )
    
    return np.array(tessellated_points), tessellated_connectivities


def compute_anglesG1(G):
    G.nodes[4]['angles'] = {frozenset([0, 1]): 1.5708, frozenset([0, 2]): 3.1416, frozenset([0, 3]): 1.5708, frozenset([1, 2]): 1.5708, frozenset([1, 3]): 3.1416, frozenset([2, 3]): 1.5708}
    # G.nodes[4]['angles'] = {(0, 1): 1.5708, (0, 2): 3.1416, (0, 3): 1.5708, (1, 2): 1.5708, (1, 3): 3.1416, (2, 3): 1.5708}
    G.nodes[0]['angles'] = {}
    G.nodes[1]['angles'] = {}
    G.nodes[2]['angles'] = {}
    G.nodes[3]['angles'] = {}

def compute_anglesG2(G):
    G.nodes['e']['angles'] = {('a', 'b'): 1.5708, ('a', 'c'): 3.1416, ('a', 'd'): 1.5708, ('b', 'c'): 1.5708, ('b', 'd'): 3.1416, ('c', 'd'): 1.5708}
    G.nodes['a']['angles'] = {}
    G.nodes['b']['angles'] = {}
    G.nodes['c']['angles'] = {}
    G.nodes['d']['angles'] = {}

def edge_match(e1, e2):
    return np.isclose(e1['weight'], e2['weight'])
    # return nx.algorithms.isomorphism.numerical_edge_match('weight', 0.01)(e1, e2)

def compute_angle(p1, p2, p3):
    """Compute angle at p2 formed by vectors p1->p2 and p2->p3"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return round(np.arccos(np.clip(cos_theta, -1, 1)), 4)  # Clip for numerical stability

def node_match(n1, n2, mapping):
    # Issue here is that when 2 nodes are checked, the mapping of all the other nodes is not known yet
    # for computational efficiency in the GraphMatcher
    if mapping == {}:
        return n1.get("angles", {}) == n2.get("angles", {})
    angles1 = n1.get("angles")
    angles2 = {}
    for key, value in n2.get("angles", {}).items():
        mapped_key = frozenset([mapping[node] for node in key])
        angles2[mapped_key] = value

    if angles1 == {}:
        if angles2 == {}:
            return True
        else:
            return False
    else:
        if angles2 == {}:
            return False
        else:
            if len(angles1) != len(angles2):
                return False
            else:
                for key in angles1.keys():
                    if not np.isclose(angles1[key], angles2[key]):
                        return False
    return True


class Graph:
    def __init__(self, points, connections, types, status):
        self.points = points
        self.connections = connections
        self.types = types
        self.status = status
        self.ports = [i for i, t in enumerate(types) if t == 'port' or t == 'counterport']
        self.junctions = [i for i, t in enumerate(types) if t == 'junction']
        self.edges = self.generate_edges()
        self.counterports = self.generate_counterports()
        # self.counterjunctions = self.generate_counterjunctions() # Still figuring out if it makes sense...
        self.translations = {}
        self.counteredges = self.generate_counteredges()
        self.symmetries = find_symmetries(points, status)

    def generate_edges(self):
        # Prepare initial edge set (edges between ports and junctions, junctions and junctions)
        edges = defaultdict(int)
        for i, (n1, n2) in enumerate(self.connections):
            edges[i] = tuple(sorted((n1, n2)))
        return edges

    def generate_counterports(self):
        counterports = {}
        P = sum([1 for t in self.types if t == 'port'])
        for i, t in enumerate(self.types):
            if t == 'port':
                counterports[i] = i + P
            elif t == 'counterport':
                counterports[i] = i - P
            # Junctions are not considered counterports (for conjugacy type 0)
            # elif t == 'junction':
            #     counterports[i] = i
        return counterports
    
    def generate_counteredges(self):
        edge2id = {v: k for k, v in self.edges.items()}
        counteredges = {}
        translations = {}
        for edge_id, (n1, n2) in self.edges.items():
            if n1 in self.ports:
                if tuple(sorted((self.counterports[n1], n2))) in edge2id:
                    counteredges[edge_id] = edge2id[tuple(sorted((self.counterports[n1], n2)))]
                    translations[edge_id] = self.points[n1] - self.points[self.counterports[n1]]
                else:
                    counteredges[edge_id] = edge_id
                    translations[edge_id] = np.zeros(len(self.points[0]))
            elif n2 in self.ports:
                if tuple(sorted((n1, self.counterports[n2]))) in edge2id:
                    counteredges[edge_id] = edge2id[tuple(sorted((n1, self.counterports[n2])))]
                    translations[edge_id] = self.points[n2] - self.points[self.counterports[n2]]
                else:
                    counteredges[edge_id] = edge_id
                    translations[edge_id] = np.zeros(len(self.points[0]))
            else:
                counteredges[edge_id] = edge_id
                translations[edge_id] = np.zeros(len(self.points[0]))

        self.counteredges = counteredges
        self.translations = translations
        return counteredges
    
    def find_minimum_internal_nodes(self, edges, start_node, end_node):
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
    
    def plot_lattice_graph(self):
        all_nodes = self.ports + self.junctions

        # Compute positions for ports and junctions
        num_ports = len(self.ports)
        num_junctions = len(self.junctions)
        if num_junctions == 1:
            junction_positions = {
                next(iter(self.junctions)): (0, 0)  # Place the single junction at the center
            }
        else:
            junction_positions = {
                junction: (0.5 * np.cos(2 * np.pi * i / num_junctions), 0.5 * np.sin(2 * np.pi * i / num_junctions))
                for i, junction in enumerate(self.junctions)
            }
        port_positions = {}
        i = 0
        for j, junction in enumerate(self.junctions):
            print(self.junction2ports()[junction])
            num_ports_per_junction = len(self.junction2ports()[junction])
            # i = 0
            for _, port in enumerate(self.junction2ports()[junction]):
                if self.counterports[port] not in port_positions.keys():
                    angle = 2 * np.pi * (i / num_ports)
                    # angle = 2 * np.pi * (i / (num_ports_per_junction + num_junctions - 1)) - np.pi/3
                    i += 1
                    # angle = angle - np.pi / (num_ports / len(junctions) -1)
                    port_positions[port] = (junction_positions[junction][0] + np.cos(angle), junction_positions[junction][1] + np.sin(angle))
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
        N=1
        subnode_positions = {}
        for node in all_nodes:
            num_edges = sum(1 for edge in self.edges.values() if node in edge)
            if node in self.ports:
                num_subnodes = N * num_edges
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
                num_subnodes = N * num_edges // 2
                base_x, base_y = positions[node]
                radius = 0.  # Inner circumference for junctions
                angle_offset = np.pi / num_subnodes / 2 + np.arctan2(base_y, base_x) + np.pi / len(self.junctions)  # Orient opposite with respect to the center
                angle_section = 2 * np.pi / len(self.junctions)  # Section of the circle based on N
                subnode_positions[node] = [
                    (
                    base_x + radius * np.cos(angle_offset + angle_section * (j / num_subnodes)),
                    base_y + radius * np.sin(angle_offset + angle_section * (j / num_subnodes))
                    )
                    for j in range(num_subnodes)
                ]
        
        plt.figure(figsize=(10, 10))
        # Latex is slow to render
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Computer Modern Roman"],
        # })

        # Plot edges
        for e, (n1, n2) in self.edges.items():
            plt.plot(
                [positions[n1][0], positions[n2][0]],
                [positions[n1][1], positions[n2][1]],
                color='black', linewidth=4, zorder=1
            )
        
        # Plot nodes and sub-nodes
        ax = plt.gca()
        zorders = [artist.get_zorder() for artist in ax.get_children()]
        max_zorder = max(zorders)
        for node, (x, y) in positions.items():
            if node in self.ports and node < num_ports // 2:
                color = 'red'
                node_type = 'p'
                node_id = node + 1
            elif node in self.counterports.values():
                color = 'blue'
                node_type = 'p'
                node_id = node + 1
            else:
                color = 'green'
                node_type = 'j'
                node_id = node - num_ports + 1
            color = 'black'
            if self.status[node] == 'suppressed':
                color = 'lightgrey'
            if node_type == 'p':
                plt.scatter(x, y, color=color, s=2000, zorder=2)
            else:
                plt.scatter(x, y, color=color, s=2000, zorder=0)
            # if node_type == 'p':
            #     if node_id == 1:
            #         node_id = '4'
            #     elif node_id == 4:
            #         node_id = '1'
            if node_type == 'j':
                x, y  = x + 0.2, y + 0.2  # Adjust position for text
                plt.text(x, y, fr"${node_type}_{{{node_id}}}$", fontsize=50, ha='center', va='center', zorder=3, color=color)
            if node_type == 'p':
                x, y = x*1.225, y*1.225  # Adjust position for text
                plt.text(x, y, fr"${node_type}_{{{node_id}}}$", fontsize=50, ha='center', va='center', zorder=3, color=color)
                if abs(y)<1e-2:
                    y += -0.15
                    if x>0:
                        x += -0.25
                    else:
                        x += 0.35
                else:
                    x += 0.5
                # plt.text(x*3/4, y*3/4, fr"$m_{{\{{{node_type}_{node_id},j_1\}}}}$", fontsize=50, ha='center', va='center', zorder=3, color='black')

        # Plot outer circumference
        outer_radius = 1
        theta = np.linspace(0, 2 * np.pi, 100)
        # plt.plot(outer_radius * np.cos(theta), outer_radius * np.sin(theta), color='black', linestyle='--', alpha=0.5)



        plt.axis('equal')
        plt.axis('off')
        plt.xlim(-1.2,1.2)
        plt.ylim(-1.2,1.2)
        plt.show()

    def plot_graph(self, N):
        """
        Plot the graph with node IDs and edge labels, coloring nodes based on their type.

        Args:
            graph: An instance of the Graph class.
            N: A number to be displayed on the edges of the graph.
        """
        G = nx.Graph()

        # Add nodes with their IDs
        for node in self.points:
            G.add_node(tuple(node) if isinstance(node, np.ndarray) else node)

        # Add edges with labels
        edge_labels = {}
        for edge_id, (n1, n2) in self.edges.items():
            G.add_edge(n1, n2)
            edge_labels[(n1, n2)] = f"{N}"

        # # Determine node colors based on their type
        # node_colors = []
        # for node in self.points:
        #     if any((node == port).all() if isinstance(node, np.ndarray) else node == port for port in self.ports):
        #         node_colors.append('green')  # Ports
        #     elif any((node == counterport).all() if isinstance(node, np.ndarray) else node == counterport for counterport in self.counterports.values()):
        #         node_colors.append('orange')  # Counterports
        #     else:
        #         node_colors.append('blue')  # Junctions

        # Draw the graph
        pos = nx.spring_layout(G)  # Layout for better visualization
        nx.draw(G, pos, with_labels=True, node_color='grey', node_size=500, font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        plt.title("Graph Visualization")
        plt.show()


if __name__ == '__main__':

    # Read lattice and extract points and connectivity
    points, conn = read_lattice_file('lattice_cubic.dat', types_and_status=False)

    # checking what points and conn has
    print("Points:", points)
    print("Connectivity:", conn)
    
    # Visualize the lattice
    plot_lattice(points, conn)

    # Find symmetries
    symm = find_symmetries(points)
    
    print(f"Found {len(symm)} symmetries")

    # # checking what symm has
    # print("Symmetries:", symm)
