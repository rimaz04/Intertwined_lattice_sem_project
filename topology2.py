from collections import defaultdict, Counter, deque
from itertools import chain, combinations, permutations, combinations_with_replacement, product
import sys
import copy
from symmetry import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from plotting import *
import math
import pickle
import random
from matplotlib import animation

def all_unique_sequences(elements, min_length=2, max_length=None):
        """
        Compute all possible sequences (ordered arrangements) of subsets of the given set,
        treating a sequence and its reverse as the same.
        
        Args:
            elements: A set of elements.

        Returns:
            A list of tuples, where each tuple represents a unique sequence (order matters, but reverses are treated as the same).
        """
        elements = list(elements)  # Convert to list for indexing
        sequences = set()  # Use a set to store unique sequences
        if max_length is None:
            max_length = len(elements)
        
        # Generate all subsets, excluding the empty set
        for subset_size in range(min_length, max_length + 1):
        # for subset_size in [2,4,6]:
            for subset in combinations(elements, subset_size):
            # Generate all permutations of the current subset
                for perm in permutations(subset):
                    # Always store the lexicographically smaller of (perm, perm[::-1])
                    sequences.add(min(perm, perm[::-1]))
        
        return sorted(sequences, key=lambda x: (len(x), x))  # Sort by length first, then lexicographically

def condition_fn(collection, values, N):
        """Check if each value appears exactly N times across all sequences in the collection."""
        total_counts = Counter({value: 0 for value in values})
        # print(collection)
        
        # Sum the occurrences of each value considering multiplicities
        for seq, mult in collection.items():
            for value in seq:
                total_counts[value] += mult

        # Check if all values appear exactly N times
        return all(count == N for count in total_counts.values())

def condition_fn_loops(collection, edges):
    """Check if the given collection of paths has at least one loop."""
    lentgh_ok = True
    for seq, mult in collection.items():
        if len(seq) < 3:
            lentgh_ok = True
            break
    if not lentgh_ok:
        return False
    loop_ok = False
    for seq, mult in collection.items():
        if len(seq) > 2 and set(edges[seq[0]]) & set(edges[seq[-1]]):
            loop_ok = True
            break
    if not loop_ok:
        return False
    return True

def are_conjugate_paths(path1, path2, edges, counterports, conjugacy_type):
    """Check if the given paths are conjugate."""
    # Conjugacy type 0: two paths are conjugate if their first and last edges contain the same ports
    if conjugacy_type == 0:
        s1 = set(edges[path1[0]]) & set(counterports[p] for p in edges[path2[0]] if p in counterports)
        s2 = set(edges[path1[-1]]) & set(counterports[p] for p in edges[path2[-1]] if p in counterports)
        s3 = set(edges[path1[0]]) & set(counterports[p] for p in edges[path2[-1]] if p in counterports)
        s4 = set(edges[path1[-1]]) & set(counterports[p] for p in edges[path2[0]] if p in counterports)
        if path1 == path2:
            return s3 and next(iter(s3)) in counterports.keys()
        # The first and last edges must contain the same ports or counterports
        return (s1 and next(iter(s1)) in counterports.keys() and \
                s2 and next(iter(s2)) in counterports.keys()) or \
                (s3 and next(iter(s3)) in counterports.keys() and \
                s4 and next(iter(s4)) in counterports.keys())
    else:
        raise ValueError("Invalid conjugacy type")

def is_conjugate(set_of_paths, edges, counterports, conjugacy_type):
    """Check if the given set of paths is conjugate (i.e., for each path theres a conjugate one)."""
    remaining_paths = set(set_of_paths)  # Create a copy to avoid modifying the input
    while remaining_paths:
        path1, multiplicity1 = remaining_paths.pop()
        if multiplicity1 > 1:
            remaining_paths.add((path1, multiplicity1 - 1))
        # Check conjugacy on the path itself
        if are_conjugate_paths(path1, path1, edges, counterports, conjugacy_type):
            continue
        found_conjugate = False
        # Check conjugacy on the remaining paths
        for path2, multiplicity2 in list(remaining_paths):
            if are_conjugate_paths(path1, path2, edges, counterports, conjugacy_type):
                remaining_paths.remove((path2, multiplicity2))
                if multiplicity2 > 1:
                    remaining_paths.add((path2, multiplicity2 - 1))
                found_conjugate = True
                break
        if not found_conjugate:
            return False
    return True

def generate_valid_multiplicities(sequences, max_multiplicity_edge, values, max_multiplicity_path=None):
    """
    Generate all possible multiplicities for the given sequences, ensuring that the total multiplicity
    for each edge involved in the sequences is exactly max_multiplicity_edge.
    
    Args:
        sequences: A list of sequences (each sequence is a list of edges), gets reshuffled by reference to speed up pruning.
        max_multiplicity_edge: Maximum allowed multiplicity for each edge (corresponds to N of strands).
        max_multiplicity_path: Maximum allowed multiplicity for each sequence (optional, equal to max_multiplicity_edge by default).
        
    Returns:
        A generator yielding tuples of multiplicities corresponding to the sequence.
    """
    if max_multiplicity_path is None:
        max_multiplicity_path = max_multiplicity_edge
    num_elements = len(sequences)
    element_counts = {v: 0 for v in values}
    # shuffle the sequences to speed up pruning during search (otherwise pairs are ordered)
    # np.random.shuffle(sequences)
    print('check 1.2.1')
    print('sequences:', sequences)

    def backtrack(index, current_multiplicity, element_counts):
        """Recursive function to construct valid multiplicities"""
        if index == num_elements:
            print('max mult edge' , max_multiplicity_edge)
            if all(count == max_multiplicity_edge for count in element_counts.values()):
                print('check 1.4')
                yield tuple(current_multiplicity)
            return
        
        # Get unique values in the current sequence
        sequence_values = set(sequences[index])
        
        # Find max multiplicity allowed for this sequence
        max_allowed = max_multiplicity_path
        for v in sequence_values:
            max_allowed = min(max_allowed, max_multiplicity_edge - element_counts[v])
        
        for m in range(max_allowed + 1):
            new_element_counts = element_counts.copy()
            for v in sequence_values:
                new_element_counts[v] += m
                
            yield from backtrack(index + 1, current_multiplicity + [m], new_element_counts)
    
    yield from backtrack(0, [], element_counts)

def generate_valid_multiplicities_fast(sequences, max_multiplicity, values, max_multiplicity_path, discard_threshold=1):
    elements_counts = {v: 0 for v in values}
    new_sequences = []
    for index, sequence in sorted(enumerate(sequences), key=lambda _: np.random.random()):
        if all([elements_counts[i] >= discard_threshold for i in range(len((elements_counts)))]):
            break
        if any([elements_counts[v] >= discard_threshold for v in sequence]):
            continue
        for v in sequence:
            elements_counts[v] += 1
        new_sequences.append(sequence)

    sequences[:]= new_sequences
    print(len(new_sequences),'picked paths:', new_sequences)
    
    yield from generate_valid_multiplicities(sequences, max_multiplicity, values, max_multiplicity_path)

def generate_multiplicity_collections(sequences, values, max_multiplicity, condition_fn, stop_early=False, max_multiplicity_path=None, discard_threshold=None, restart=1, edges=None, counterports=None, conjugacy_type=0):
    """
    Generate all possible collections of sequences with attached multiplicities,
    ensuring the total collection satisfies the given condition.
    
    Args:
        sequences: A list of sequences.
        max_multiplicity: Maximum allowed multiplicity for each sequence.
        condition_fn: A function that takes a collection (Counter) and returns True if valid.

    Returns:
        A set of valid collections (each collection is a Counter mapping sequences to multiplicities).
    """
    if max_multiplicity_path is None:
        max_multiplicity_path = max_multiplicity

    tic = time.time()

    valid_collections = set()
    
    mult = 0
    valid = 0
    if discard_threshold is None:
        # Generate multiplicities from all possible sequences

        print('Check 1.1.1')
        print('sequences:', sequences)
        print('max_multiplicity:', max_multiplicity)
        print('values:', values)
        print('max_multiplicity_path:', max_multiplicity_path)

        # Check 1.1.1
        # sequences: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # max_multiplicity: 2
        # values: [0, 1, 2, 3]
        # max_multiplicity_path: 2
        # multiplicity: (0, 0, 2, 2, 0, 0)
        # multiplicity: (0, 1, 1, 1, 1, 0)
        # multiplicity: (0, 2, 0, 0, 2, 0)
        # multiplicity: (1, 0, 1, 1, 0, 1)
        # multiplicity: (1, 1, 0, 0, 1, 1)
        # multiplicity: (2, 0, 0, 0, 0, 2)

        valid_multiplicities = generate_valid_multiplicities(sequences, max_multiplicity, values, max_multiplicity_path)

        for multiplicities in valid_multiplicities:
                print('multiplicity:', multiplicities)
                toc = time.time()
                if toc - tic > 1:
                    tic = toc
                    print(f"Processed {mult} multiplicities, found {valid} valid collections", end="\r")

                collection = Counter({seq: count for seq, count in zip(sequences, multiplicities) if count > 0})
                mult+=1
                if condition_fn(collection, values, max_multiplicity):
                    valid_collection = frozenset(collection.items())
                    # Check conjugacy
                    if conjugacy_type is not None:
                        if is_conjugate(valid_collection, edges, counterports, conjugacy_type):
                            valid_collections.add(valid_collection)
                        else:
                            continue # Skip this collection if conjugacy is not satisfied
                    else:
                        valid_collections.add(valid_collection)
                    valid+=1
                    if stop_early and valid >= stop_early:
                        print(f"Processed {mult} multiplicities, found {valid} valid collections")
                        return valid_collections
                    
    else:
        r = 0 # Restart counter
        while r < restart and (valid < stop_early if stop_early is not False else False):

            tic_r = time.time()

            # Generate multiplicities from subset of sequences
            new_sequences = sequences.copy()
            valid_multiplicities = generate_valid_multiplicities_fast(new_sequences, max_multiplicity, values, max_multiplicity_path, discard_threshold=discard_threshold)
            r += 1

            for multiplicities in valid_multiplicities:

                toc = time.time()
                # if toc - tic_r > 2:
                #     break
                if toc - tic > 1:
                    tic = toc
                    print(f"Processed {mult} multiplicities, found {valid} valid collections", end="\r")

                collection = Counter({seq: count for seq, count in zip(new_sequences, multiplicities) if count > 0})
                mult+=1
                if condition_fn(collection, values, max_multiplicity):
                    valid_collection = frozenset(collection.items())
                    # Check conjugacy
                    if conjugacy_type is not None:
                        if is_conjugate(valid_collection, edges, counterports, conjugacy_type):
                            valid_collections.add(valid_collection)
                        else:
                            continue # Skip this collection if conjugacy is not satisfied
                    else:
                        valid_collections.add(valid_collection)
                    valid+=1
                    if stop_early and valid >= stop_early:
                        print(f"Processed {mult} multiplicities, found {valid} valid collections")
                        return valid_collections
            print(f"Processed {mult} multiplicities, found {valid} valid collections")

    return valid_collections

def compute_color(color_map, path, unpaired_paths, edges, counterports, seed=None, conjugacy_type=0):
    """
    Compute the color for a given path based on conjugacy and unpaired paths.
    
    """
    if seed == None:
        seed = 0
    if color_map == 'black':
        return 'black', seed, unpaired_paths
    elif color_map == 'lightgrey':
        return 'lightgrey', seed, unpaired_paths
    elif color_map == 'conjugate':
        found_unpaired_path = False
        if path not in unpaired_paths:
            for unpaired_path in unpaired_paths:
                if are_conjugate_paths(path, unpaired_path, edges, counterports, conjugacy_type):
                    found_unpaired_path = True
                    color = unpaired_paths[unpaired_path].pop()
                    if not unpaired_paths[unpaired_path]:
                        del unpaired_paths[unpaired_path]
                    break
        if not found_unpaired_path:
            np.random.seed(seed)
            seed += 1
            color = np.random.rand(3,)
            if path in unpaired_paths:
                unpaired_paths[path].append(color)
            else:
                unpaired_paths[path] = [color]
    else:
        np.random.seed(seed)
        seed += 1
        color = np.random.rand(3,)
    
    return color, seed, unpaired_paths

class Graph:
    def __init__(self, points, connections, types, status):
        self.points = points
        self.connections = connections
        self.types = types
        self.status = status
        self.ports = [i for i, t in enumerate(types) if t == 'port' or t == 'counterport']
        self.junctions = [i for i, t in enumerate(types) if t == 'junction']
        self.translations = {}
        self.edges = self.generate_edges()
        self.counterports = self.generate_counterports()
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
        return counterports
    
    def generate_counteredges(self):
        edge2id = {v: k for k, v in self.edges.items()}
        counteredges = {}
        translations = {}
        for edge_id, (n1, n2) in self.edges.items():
            if n1 in self.ports:
                # n1 port and n2 junction
                if self.status[self.counterports[n1]] == 'suppressed':
                    counteredges[edge_id] = -1
                    translations[edge_id] = np.zeros(len(self.points[0]))
                else:
                    for n in self.junctions:
                        if tuple(sorted((self.counterports[n1], n))) in edge2id:
                            counteredges[edge_id] = edge2id[tuple(sorted((self.counterports[n1], n)))]
                            translations[edge_id] = self.points[n1] - self.points[self.counterports[n1]]
                            break
            elif n2 in self.ports:
                # n2 port and n1 junction
                if self.status[self.counterports[n2]] == 'suppressed':
                    counteredges[edge_id] = -1
                    translations[edge_id] = np.zeros(len(self.points[0]))
                else:
                    for n in self.junctions:
                        if tuple(sorted((self.counterports[n2], n))) in edge2id:
                            counteredges[edge_id] = edge2id[tuple(sorted((self.counterports[n2], n)))]
                            translations[edge_id] = self.points[n2] - self.points[self.counterports[n2]]
                            break
            else:
                # both n1 and n2 junctions -> internal edge
                counteredges[edge_id] = edge_id
                translations[edge_id] = np.zeros(len(self.points[0]))

        self.counteredges = counteredges
        self.translations = translations
        return counteredges

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

        # Draw the graph
        pos = nx.spring_layout(G)  # Layout for better visualization
        nx.draw(G, pos, with_labels=True, node_color='grey', node_size=500, font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        plt.title("Graph Visualization")
        plt.show()

    def plot_lattice(self, point_types=True, edges=True):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        s_p = 50
        s_cp = 50
        shift = 0.02
        if point_types is True:
            for i, (point, point_type) in enumerate(zip(self.points, self.types)):
                if point_type == 'port':
                    ax.scatter(point[0], point[1], point[2], c='r', marker='o', s=s_p)
                    ax.text(point[0] + shift, point[1] + shift, point[2], str(i), color='black')
                elif point_type == 'counterport':
                    ax.scatter(point[0], point[1], point[2], c='b', marker='o', s=s_cp)
                    ax.text(point[0] + shift, point[1] + shift, point[2], str(i), color='black')
                elif point_type == 'junction':
                    ax.scatter(point[0], point[1], point[2], c='g', marker='s', s=200)
                    ax.text(point[0] + shift, point[1] + shift, point[2], str(i), color='black')
        else:
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='k', marker='o')

        for edge_id, (n1, n2) in self.edges.items():
            if edges is True:
                mid_point = (self.points[n1] + self.points[n2]) / 2
                ax.text(mid_point[0] + shift, mid_point[1] + shift, mid_point[2], str(edge_id), color='purple')
            ax.plot([self.points[n1][0], self.points[n2][0]],
                    [self.points[n1][1], self.points[n2][1]],
                    [self.points[n1][2], self.points[n2][2]], c='black')
            
        ax.set_aspect('equal')
        plt.show()
    

class Topology:
    def __init__(self, filename, N, conjugacy = True, max_multiplicity=None):
        self.graph = Graph(*read_lattice_file(filename))
        self.ports = self.graph.ports
        self.junctions = self.graph.junctions
        self.edges = self.graph.edges
        self.status = self.graph.status
        self.N = N
        self.max_multiplicity = N if max_multiplicity is None else max_multiplicity
        self.counterports = self.graph.counterports
        self.counteredges = self.graph.counteredges
        self.translations = self.graph.translations
        self.conjugacy = conjugacy
        self.symmetries = self.graph.symmetries
        self.solutions = []
        self.permuted_solutions = {}
        self.expanded_solutions = {}
        self.loops = {}
        self.expanded_open_strands = {}
        self.expanded_loops = {}
        self.expanded_strands2strands = {}
        # Possibly try to find a way to compute these two automatically
        self.power_set_min_length = 2
        self.power_set_max_length = 4



    def generate(self, stop_early=False, discard_threshold=None, restart=1, permuted_solutions=None, shuffle_seed=None):
        self.generate_solutions(stop_early=stop_early, discard_threshold=discard_threshold, restart=restart)
        
        if permuted_solutions is not None:
            if not isinstance(permuted_solutions, list):
                permuted_solutions = [permuted_solutions]
            for solution in range(len(self.solutions)):
                for permuted_solution in permuted_solutions:
                    if self.instantiate_strands(solution, permuted_solution, shuffle_seed=shuffle_seed):
                        # If instantiation was successful, compute loops
                        self.compute_loops(solution, permuted_solution=permuted_solution)
        elif permuted_solutions == 'all':
            # Compute all permutations for each solution
            for solution in range(len(self.solutions)):
                start = time.time()
                # Finds them all, computing from index 0 until end
                self.instantiate_strands_v1(solution)
                # Finds them all, computing from end until index 0
                # self.instantiate_strands_v2(solution, 1000)
                end = time.time()
                print(f"Instantiated solution {solution} in {end - start:.2f} seconds.")
                for permuted_solution in self.permuted_solutions[solution]:
                    self.compute_loops(solution, permuted_solution=permuted_solution)


    def instantiate_strands(self, solution, permuted_solution=None, shuffle_seed=None):
        """
        Assign unique IDs (from 0 to self.N) to each edge in the solution such that
        each occurrence of the same edge in the set appears with a different ID.

        Args:
            solution: A set of tuples (paths), where each tuple is made of a tuple of edges and a multiplicity.

        Returns:
            A dictionary mapping each edge to a list of assigned IDs.
        """
        strand_solution = []
        if permuted_solution is None:
            edge_id_map = {e: list(range(self.N)) for e in self.edges.keys()}
        else:
            cardinality_permutations = self.compute_cardinality_strands_permutations(solution)[solution]
            if permuted_solution >= cardinality_permutations:
                print(f"Warning: Permuted solution {permuted_solution} exceeds the maximum number of permutations {cardinality_permutations} for {len(self.edges)} edges.")
                return False
            available_permuted_ids = list(permutations(range(self.N)))
            if shuffle_seed: # Shuffle the available permutations if a seed is provided
                random.seed(shuffle_seed)
                random.shuffle(available_permuted_ids)
            # The permuted_solution required is always an index in the range of available_permuted_ids for now
            # So strands_instantiations with close permuted_solution indices will still be similar
            
            # Compute 'edge_permutations' only if not already cached, or if self.edges.keys() or self.N has changed
            if not hasattr(self, '_cached_edge_permutations') or \
               not hasattr(self, '_cached_edge_permutations_edges_keys') or \
               not hasattr(self, '_cached_edge_permutations_N') or \
               list(self.edges.keys()) != self._cached_edge_permutations_edges_keys or \
               self.N != self._cached_edge_permutations_N:
                edge_permutations = list(combinations_with_replacement(available_permuted_ids, len(self.edges.keys())))
                self._cached_edge_permutations = edge_permutations
                self._cached_edge_permutations_edges_keys = list(self.edges.keys())
                self._cached_edge_permutations_N = self.N
            else:
                edge_permutations = self._cached_edge_permutations
            edge_id_map = {e: list(permuted_ids) for e, permuted_ids in zip(self.edges.keys(), edge_permutations[permuted_solution])}

        for path, multiplicity in list(self.solutions)[solution]:
            for _ in range(multiplicity):
                path_with_strands = []
                for edge in path:
                    strand_id = edge_id_map[edge].pop(0)
                    path_with_strands.append((edge, strand_id))
                strand_solution.append(path_with_strands)

        if solution not in self.permuted_solutions:
            self.permuted_solutions[solution] = {}
        permuted_solution_idx = permuted_solution if permuted_solution is not None else 0
        self.permuted_solutions[solution][permuted_solution_idx] = strand_solution
        
        return True
    
    def instantiate_strands_v1(self, solution, break_early=None):

        def assign_strand_ids(available_edge_ids, strand_solution, paths_multiplicities, break_early):
            if break_early is not None and len(strand_solutions) >= break_early:
                return
            if available_edge_ids == {}:
                # All edges have been assigned IDs, append strand_solution
                strand_solutions.append(strand_solution)
                return
            path_multiplicity = paths_multiplicities.pop(0)
            available_edge_ids_path = [available_edge_ids[e][:] for e in path_multiplicity[0]]
            possible_assignments_path = list(product(*available_edge_ids_path))
            assignments_path_multiplicity = list(combinations(possible_assignments_path, path_multiplicity[1]))
            valid_assignments = []
            for assignment in assignments_path_multiplicity:
                compatible = True
                for e in range(len(path_multiplicity[0])):
                    used_ids = set()
                    for m in range(path_multiplicity[1]):
                        used_ids.add(assignment[m][e])
                    if len(used_ids) < path_multiplicity[1]:
                        # Non compatible assignment, skip
                        compatible = False
                        break
                if compatible:
                    valid_assignments.append(assignment)
            if not valid_assignments:
                # This should never happen
                raise ValueError(f"No valid assignments found for path {path_multiplicity[0]} with multiplicity {path_multiplicity[1]}")
            else:
                for assignment in valid_assignments:
                    available_edge_ids_updated = copy.deepcopy(available_edge_ids)
                    for i, e in enumerate(path_multiplicity[0]):
                        for m in range(path_multiplicity[1]):
                            available_edge_ids_updated[e].remove(assignment[m][i])
                    available_edge_ids_updated = {e: ids for e, ids in available_edge_ids_updated.items() if ids}  # Remove edges with no available IDs
                    strand_solution_updated = strand_solution.copy()
                    for m in range(path_multiplicity[1]):
                        strand_solution_updated.append([(e, assignment[m][i]) for i, e in enumerate(path_multiplicity[0])])
                    assign_strand_ids(available_edge_ids_updated, strand_solution_updated, paths_multiplicities.copy(), break_early)
            
        strand_solutions = []
        available_edge_ids = {e: list(range(self.N)) for e in self.edges.keys()}

        # # check what edge ids there are
        # print("Available edge ids before assignment:", available_edge_ids)
        print('this is done 1')
        print('length of solution rn', len(list(self.solutions)[solution]))
        assign_strand_ids(available_edge_ids, [], list(list(self.solutions)[solution]), break_early)
        print('this is done 2')
        # print(len(strand_solutions), 'strand solutions found for solution', solution)

        # # check what edge ids are now
        # print("Available edge ids after assignment :", available_edge_ids)

        if solution not in self.permuted_solutions:
            self.permuted_solutions[solution] = {}
        for permuted_solution in range(len(strand_solutions)):
            self.permuted_solutions[solution][permuted_solution] = strand_solutions[permuted_solution]

        return strand_solutions

    def instantiate_strands_v2(self, solution, break_early=None):
        """
        Efficiently generate all possible strand assignments for a given solution.
        Each edge can be used self.N times, and each path's multiplicity must be respected.
        This version uses a more efficient recursive backtracking with pruning.
        """

        paths_multiplicities = list(list(self.solutions)[solution])
        num_paths = len(paths_multiplicities)
        edge_ids = list(self.edges.keys())
        N = self.N
        
        # Precompute all possible assignments for each path/multiplicity
        path_assignments = []
        for path, multiplicity in paths_multiplicities:
            # For each edge in path, assign multiplicity distinct strand ids (from 0 to N-1)
            # For each path, generate all possible assignments of strand ids to edges, such that
            # for each edge, the strand ids are unique within the path
            edge_ranges = [range(N) for _ in path]
            # All possible assignments for one instance of the path
            single_assignments = list(product(*edge_ranges))
            # For multiplicity > 1, choose multiplicity distinct assignments with no repeated strand id for any edge
            if multiplicity == 1:
                path_assignments.append([[a] for a in single_assignments])
            else:
                # Only keep combinations where for each edge, the strand ids are all different
                valid_multi = []
                for combo in combinations(single_assignments, multiplicity):
                    valid = True
                    for edge_idx in range(len(path)):
                        ids = [combo[m][edge_idx] for m in range(multiplicity)]
                        if len(set(ids)) < multiplicity:
                            valid = False
                            break
                    if valid:
                        valid_multi.append(list(combo))
                path_assignments.append(valid_multi)

        strand_solutions = []
        # Use an explicit stack for DFS to avoid recursion limit
        stack = deque()
        # Each stack entry: (path_idx, current_solution, used_ids_per_edge)
        # used_ids_per_edge: {edge: set(used_ids)}
        stack.append((0, [], {e: set() for e in edge_ids}))

        while stack:
            if break_early is not None and len(strand_solutions) >= break_early:
                break
            path_idx, current_solution, used_ids_per_edge = stack.pop()
            if path_idx == num_paths:
                strand_solutions.append(current_solution)
                continue
            path, multiplicity = paths_multiplicities[path_idx]
            for assignment in path_assignments[path_idx]:
                # Check if assignment is compatible with used_ids_per_edge
                compatible = True
                for m in range(multiplicity):
                    for edge_idx, e in enumerate(path):
                        strand_id = assignment[m][edge_idx]
                        if strand_id in used_ids_per_edge[e]:
                            compatible = False
                            break
                    if not compatible:
                        break
                if not compatible:
                    continue
                # Update used_ids_per_edge
                new_used = {e: used_ids_per_edge[e].copy() for e in edge_ids}
                for m in range(multiplicity):
                    for edge_idx, e in enumerate(path):
                        strand_id = assignment[m][edge_idx]
                        new_used[e].add(strand_id)
                # Prune if any edge exceeds N
                if any(len(new_used[e]) > N for e in edge_ids):
                    continue
                # Build new solution
                new_solution = current_solution + [
                    [(e, assignment[m][edge_idx]) for edge_idx, e in enumerate(path)]
                    for m in range(multiplicity)
                ]
                stack.append((path_idx + 1, new_solution, new_used))

        max_perms = self.compute_cardinality_strands_permutations(solution)[solution]
        found_perms = len(strand_solutions)
        diff = max_perms - found_perms

        if solution not in self.permuted_solutions:
            self.permuted_solutions[solution] = {}
        for permuted_solution in range(len(strand_solutions)):
            self.permuted_solutions[solution][int(max_perms-1-permuted_solution)] = strand_solutions[permuted_solution]
        
        # Reorder the keys in increasing order (they are currently in decreasing order)
        self.permuted_solutions[solution] = {k: self.permuted_solutions[solution][k] for k in sorted(self.permuted_solutions[solution].keys())}

        # print(len(strand_solutions), 'strand solutions found for solution', solution)
        return strand_solutions
    
    def compute_cardinality_strands_permutations(self, solution=None):
        # Now handles solutions where N is constant along all edges
        N = self.N
        if solution is not None:
            solutions = [solution]
        else:
            solutions = range(len(self.solutions))
        cardinalities = {}
        for solution in solutions:
            paths_multiplicities = list(list(self.solutions)[solution])
            cardinality = math.factorial(N) ** len(self.edges)
            for path, multiplicity in paths_multiplicities:
                cardinality /= math.factorial(multiplicity)
            cardinalities[solution] = int(cardinality)
        return cardinalities

    def compute_loops(self, solution, permuted_solution=0, threshold = None):
        threshold = threshold if threshold is not None else 10*max([np.linalg.norm(self.translations[e]) for e in self.edges.keys()])
        loops = []
        expanded_open_strands  = []
        expanded_strands2strands = []
        visited_paths_locations = []
        expanded_loops = []
        visited_path_ids_uc = set()
        base_uc_location = np.zeros(list(self.translations.values())[0].shape)
        for path_id_uc, path_uc in enumerate(self.permuted_solutions[solution][permuted_solution]):

            # Skip paths in UCs already processed
            if path_id_uc in visited_path_ids_uc:
                continue
            
            visited_path_ids_uc.add(path_id_uc)
            visited_path_ids = {path_id_uc}
            loop = [path_id_uc]
            visited_paths = [(path_uc, base_uc_location)]
            e_start, s_start = path_uc[0]
            e_old, s_old = path_uc[-1]
            translation = base_uc_location
            def find_loop(e_old, s_old, translation_old, loop):
                if e_old not in self.edges:
                    return False
                for path_id, path in enumerate(self.permuted_solutions[solution][permuted_solution]):
                    
                    # Check if current path is successor
                    e1, s1 = path[0]
                    e2, s2 = path[-1]
                    if (e1 == self.counteredges[e_old] and s1 == s_old):
                        e_new, s_new = e2, s2
                    elif (e2 == self.counteredges[e_old] and s2 == s_old):
                        e_new, s_new = e1, s1
                    else:
                        continue
                    # Successor found

                    if path_id in visited_path_ids:
                        # Periodicity is achieved
                        # Update translation
                        new_translation = translation_old + self.translations[e_old]
                        # If the path closes the loop
                        if np.linalg.norm(new_translation - base_uc_location) < 1e-12:
                            loops.append(loop)
                            expanded_loops.append(expanded_loop)
                            expanded_strands2strands.append(loop)
                            return True
                        # If path is an open curve
                        else:
                            expanded_open_strands.append(expanded_loop)
                            expanded_strands2strands.append(loop)
                            return False

                    # Update translation
                    new_translation = translation_old + self.translations[e_old]
                    # Add current path
                    visited_paths.append((path, new_translation))
                    visited_path_ids.add(path_id)
                    # IF path is in the original UC, add it to the visited paths of the UC as well
                    if np.linalg.norm(new_translation - base_uc_location) < 1e-12:
                        visited_path_ids_uc.add(path_id)
                        # Add path to the loop
                        loop.append(path_id)

                    if find_loop(e_new, s_new, new_translation, loop):
                        return True
                    else:
                        return False
                    

            expanded_loop = len(visited_paths_locations)
            find_loop(e_old, s_old, translation, loop)
            visited_paths_locations.append(visited_paths[::])

        self.loops[solution, permuted_solution] = loops
        self.expanded_open_strands[solution, permuted_solution] = expanded_open_strands
        self.expanded_solutions[solution, permuted_solution] = visited_paths_locations
        self.expanded_loops[solution, permuted_solution] = expanded_loops
        self.expanded_strands2strands[solution, permuted_solution] = expanded_strands2strands

        return loops, visited_paths_locations






    def generate_solutions(self, stop_early=False, discard_threshold=None, restart=1):
        # Generate all possible sequences (paths) of edges
        all_seqs = all_unique_sequences(self.edges.keys(), min_length=self.power_set_min_length, max_length=self.power_set_max_length)

        # # printing the min and max length of the sequences generated:
        # print('power set min length:', self.power_set_min_length) # is manually set to 2 ad 3 respectively
        # print('power set max length:', self.power_set_max_length)

        # # printing self.egde
        print('edges:', self.edges)


        # # checking what all_seqs are:
        # print('all sequences:', all_seqs)

        # Filter out invalid paths
        valid_paths = [s for s in all_seqs if self.is_valid_path(s)]

        # # checking valid paths
        # print('valid paths before pruning:', valid_paths)


        # Prune redundant loops (e.g., (0,1,2) and (1,2,0) are the same loop)
        valid_paths = self.prune_redundant_loops(valid_paths)
        # print(len(valid_paths),'possible paths:', valid_paths)

        # # checking valid paths
        print('valid paths after pruning:', valid_paths)

        # values = list(set(e for path in valid_paths for e in path))
        values = list(self.edges.keys())

        # # checking what values are:
        # print('values:', values)

        # Generate all possible collections of paths with attached multiplicities
        if self.conjugacy:
            valid_collections = generate_multiplicity_collections(valid_paths, values, self.N, condition_fn, stop_early=stop_early, max_multiplicity_path=self.max_multiplicity, discard_threshold=discard_threshold, restart=restart, edges=self.edges, counterports=self.counterports, conjugacy_type=0)
        else:
            valid_collections = generate_multiplicity_collections(valid_paths, values, self.N, condition_fn, stop_early=stop_early, max_multiplicity_path=self.max_multiplicity, discard_threshold=discard_threshold, restart=restart, edges=self.edges, conjugacy_type=None)

        # # check what vali dcollections are
        print('valid collections:', valid_collections)
        print(len(valid_collections), "valid collections found")
        print('solution symmetris:', self.symmetries)
        # print('edges:', self.edges)

        # Store valid collections before filtering (needed for multiplicity computation)
        self.valid_collections = valid_collections

        # Filter out isomorphic solutions based on symmetry
        self.solutions = self.find_unique_sets(valid_collections, self.symmetries, self.edges)
        print(len(self.solutions), "unique solutions found")
        
        # checking what solution is:
        if p_sol == True:
            print('Checking solution', self.solutions)


        # finished looking at it on 26th march 
        # understood what number of solutions are 

    def generate_new_edge_mapping(self, node_mapping, edges):
        """Generate a new edge mapping based on a given node permutation."""
        # new_edges_ids = {tuple(sorted((node_mapping[a], node_mapping[b]))): id for id, (a, b) in edges.items()}

        # #printing edge ids causing issue in 4 junctions case
        # print('checking edge ids for edge mapping generation:', new_edges_ids)
        # print('edge items:', edges.items())
        # edge_mapping = {id: new_edges_ids[edge] for id, edge in edges.items()}   
        # 
        edge_to_id = {tuple(sorted(edge)): edge_id for edge_id, edge in edges.items()}
        edge_mapping = {}
        for edge_id, (a, b) in edges.items():
            mapped_edge = tuple(sorted((node_mapping[a], node_mapping[b])))
            if mapped_edge not in edge_to_id:
                raise KeyError(mapped_edge)
            edge_mapping[edge_id] = edge_to_id[mapped_edge]     
        return edge_mapping

    def apply_permutation(self, path_set, permutation, edges):
        """Transform path_set based on a given node permutation."""
        mapping_nodes = {orig: perm for orig, perm in enumerate(permutation)}
        # mapping = self.generate_new_edge_mapping(mapping_nodes, edges)
        try:
            mapping = self.generate_new_edge_mapping(mapping_nodes, edges)
        except KeyError:
            return None
        transformed = frozenset({
            ((tuple(mapping[node] for node in path), count) if mapping[path[0]] < mapping[path[-1]] else (tuple(mapping[node] for node in reversed(path)), count))
            for path, count in path_set
        })
        return transformed

    def find_unique_sets(self, path_sets, equivalences, edges):
        """Remove equivalent sets under node permutations."""
        unique_sets = set()
        
        for path_set in path_sets:
            # equivalent_representations = {self.apply_permutation(path_set, eq, edges) for eq in equivalences}
            

            equivalent_representations = {
                transformed
                for eq in equivalences
                for transformed in [self.apply_permutation(path_set, eq, edges)]
                if transformed is not None
            }
            if not equivalent_representations:
                equivalent_representations = {frozenset(path_set)}

            if not unique_sets.intersection(equivalent_representations):  # No equivalent already stored
                unique_sets.add(frozenset(path_set))  # Store as canonical representation
    
        return unique_sets

    def compute_solution_multiplicities(self):
        """
        Compute the multiplicity of each unique solution.
        
        Multiplicity = number of valid (non-unique) solutions that map to each unique solution under symmetries.
        
        Example: if there are 4 valid solutions but 3 are equivalent under symmetry,
        the unique solution representing those 3 has multiplicity 3.
        
        Returns:
            A dictionary where keys are unique solutions (as frozensets) and values are their multiplicities (int).
        """
        if not hasattr(self, 'valid_collections'):
            print("Warning: valid_collections not stored. Run generate_solutions() first.")
            return {}
        
        multiplicities = {}
        
        for unique_sol in self.solutions:
            count = 0
            for valid_col in self.valid_collections:
                # Generate all equivalent representations of valid_col under symmetries
                # equivalent_reps = {self.apply_permutation(valid_col, eq, self.edges) for eq in self.symmetries}
                equivalent_reps = {
                    transformed
                    for eq in self.symmetries
                    for transformed in [self.apply_permutation(valid_col, eq, self.edges)]
                    if transformed is not None
                }


                # Check if unique_sol matches any equivalent representation
                if unique_sol in equivalent_reps:
                    count += 1
            multiplicities[unique_sol] = count
        
        return multiplicities

    def is_valid_path(self, path):
        """Check if the given path is valid."""
        # Loop
        if len(path) > 2 and set(self.edges[path[0]]) & set(self.edges[path[-1]]):
            # The path must be made by junctions only
            if any(n in self.ports for e in path for n in self.edges[e]):
                return False
            # Ensure edges in the path are contiguous by checking for a common node
            if not all(set(self.edges[path[i]]) & set(self.edges[path[i + 1]]) for i in range(len(path) - 1)):
                return False
        else:
            # Open path
            # The path must contain at least two edges
            if len(path) < 2:
                return False
            # Ensure edges in the path are contiguous by checking for a common node
            if not all(set(self.edges[path[i]]) & set(self.edges[path[i + 1]]) for i in range(len(path) - 1)):
                return False
            # Ensure no edge in the path shares a node with the two edges before it
            if len(path) > 2 and any(set(self.edges[path[i]]) & set(self.edges[path[i+2]]) for i in range(len(path) - 2)):
                return False
            # The first and last edges in the path must contain ports
            if not (any(n in self.ports for n in self.edges[path[0]]) and any(n in self.ports for n in self.edges[path[-1]])):
                return False
        return True
    
    def prune_redundant_loops(self, paths):
        """Prune redundant loops from the given paths."""
        pruned_paths = []
        seen_loops = set()
        for path in paths:
            # Check if the path is a loop (first and last edge share a node, length > 2)
            is_loop = len(path) > 2 and set(self.edges[path[0]]) & set(self.edges[path[-1]])
            if not is_loop:
                pruned_paths.append(path)
                continue
            # If any edge in the path involves a port, keep it
            if any(n in self.ports for e in path for n in self.edges[e]):
                AssertionError("Loops with ports should have been filtered out earlier.")
            # Loop with only junctions: check for duplicates (cyclic or reversed)
            # Generate all cyclic permutations and their reverses
            loop_variants = set()
            plen = len(path)
            for offset in range(plen):
                rotated = tuple(path[offset:] + path[:offset])
                reversed_rotated = tuple(reversed(rotated))
                loop_variants.add(rotated)
                loop_variants.add(reversed_rotated)
            # If any variant has been seen, skip this path
            if seen_loops & loop_variants:
                continue
            # Otherwise, add all variants to seen_loops and keep this path
            seen_loops.update(loop_variants)
            pruned_paths.append(path)
        return pruned_paths    
    
    def are_conjugate_paths(self, path1, path2, conjugacy_type=0):
        """Check if the given paths are conjugate."""
        # Conjugacy type 0: two paths are conjugate if their first and last edges contain the same ports
        if conjugacy_type == 0:
            s1 = set(self.edges[path1[0]]) & set(self.counterports[p] for p in self.edges[path2[0]] if p in self.counterports)
            s2 = set(self.edges[path1[-1]]) & set(self.counterports[p] for p in self.edges[path2[-1]] if p in self.counterports)
            s3 = set(self.edges[path1[0]]) & set(self.counterports[p] for p in self.edges[path2[-1]] if p in self.counterports)
            s4 = set(self.edges[path1[-1]]) & set(self.counterports[p] for p in self.edges[path2[0]] if p in self.counterports)
            if path1 == path2:
                return s3 and next(iter(s3)) in self.counterports.keys()
            # The first and last edges must contain the same ports or counterports
            return (s1 and next(iter(s1)) in self.counterports.keys() and \
                    s2 and next(iter(s2)) in self.counterports.keys()) or \
                    (s3 and next(iter(s3)) in self.counterports.keys() and \
                    s4 and next(iter(s4)) in self.counterports.keys())
        else:
            raise ValueError("Invalid conjugacy type")
    
    def is_conjugate(self, set_of_paths):
        """Check if the given set of paths is conjugate (i.e., for each path theres a conjugate one)."""
        remaining_paths = set(set_of_paths)  # Create a copy to avoid modifying the input
        while remaining_paths:
            path1, multiplicity1 = remaining_paths.pop()
            if multiplicity1 > 1:
                remaining_paths.add((path1, multiplicity1 - 1))
            # Check conjugacy on the path itself
            if self.are_conjugate_paths(path1, path1):
                continue
            found_conjugate = False
            # Check conjugacy on the remaining paths
            for path2, multiplicity2 in list(remaining_paths):
                if self.are_conjugate_paths(path1, path2):
                    remaining_paths.remove((path2, multiplicity2))
                    if multiplicity2 > 1:
                        remaining_paths.add((path2, multiplicity2 - 1))
                    found_conjugate = True
                    break
            if not found_conjugate:
                return False
        return True

    def enforce_conjugacy(self, set_of_sets_paths):
        """Enforce conjugacy on the given set of sets of paths."""
        return set([s for s in set_of_sets_paths if self.is_conjugate(s)])
    
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
        # all_nodes = set(chain(*[edge for edge in self.edges.values()]))
        all_nodes = set(self.ports) | set(self.junctions)
        junctions = set(self.junctions)

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
            if num_edges == 0:
                num_edges = 1  # For suppressed ports
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

    def plot_solution_v1(self, solution, permutation=None, loops=False, starting_seed=0, color_map=None):
        """
        Plot a 2D representation of a solution, where nodes are placed on circumferences
        and edges are drawn based on the paths in the solution. Each node (port or junction)
        has sub-nodes (strands) equal to self.N times the number of edges involving that node.

        Args:
            solution: A set of edges representing a path and their multiplicities.
        """
        if color_map is None:
            if self.conjugacy:
                color_map = 'conjugate'
        elif color_map == 'conjugate':
            if not self.conjugacy:
                raise ValueError("Color map 'conjugate' is only available for conjugate solutions.")

        # Separate ports and junctions
        # all_nodes = set(chain(*[edge for edge in self.edges.values()]))
        all_nodes = set(self.ports) | set(self.junctions)
        junctions = set(self.junctions)

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
            # print(self.junction2ports()[junction])
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
            if num_edges == 0:
                num_edges = 1  # For suppressed ports
            if node in self.ports:
                num_subnodes = self.N * num_edges
                base_x, base_y = positions[node]
                radius = 0.1  # Outer circumference for ports
                if num_subnodes == 1:
                    # If only one subnode, its position coincides with the node
                    subnode_positions[node] = [(base_x, base_y)]
                else:
                    # Place sub-nodes symmetrically on a tangent line at the port location
                    if permutation is not None:
                        # tangent_length = 0.25  # Total length of the tangent segment
                        tangent_length = 0.4  # Total length of the tangent segment
                        # tangent_length = 0.5  # Total length of the tangent segment

                    else:
                        tangent_length = 0 # Port-level solution
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
                radius = 0.  # Inner circumference for junctions
                angle_offset = np.pi / num_subnodes / 2 + np.arctan2(base_y, base_x) + np.pi / len(junctions)  # Orient opposite with respect to the center
                angle_section = 2 * np.pi / len(junctions)  # Section of the circle based on N
                subnode_positions[node] = [
                    (
                    base_x + radius * np.cos(angle_offset + angle_section * (j / num_subnodes)),
                    base_y + radius * np.sin(angle_offset + angle_section * (j / num_subnodes))
                    )
                    for j in range(num_subnodes)
                ]
        
        # plt.figure(figsize=(10, 10))
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Computer Modern Roman"],
        # })
        center = (0, 0)

        # Plot edges using sub-nodes
        used_subnodes = {node: [0] * len(subnodes) for node, subnodes in subnode_positions.items()}
        radius = 1.35
        s_all = []
        unpaired_paths = {}
        seed = starting_seed
        if permutation is None:
            for p, (path, multiplicity) in enumerate(list(self.solutions)[solution]):
                for _ in range(multiplicity):
                    current_path = []
                    for i, e in enumerate(path):
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
                        # plt.plot([x1, x2], [y1, y2], color='black', alpha=0.7, zorder=0)
                        if i == 0:
                            current_path.append((x1, y1))
                            current_path.append((x2, y2))
                        else:
                            if current_path[-2] == (x1, y1):
                                current_path = current_path[:-1]
                                current_path.append((x2, y2))
                            elif current_path[-2] == (x2, y2):
                                current_path = current_path[:-1]
                                current_path.append((x1, y1))
                            elif current_path[-1] == (x1, y1):
                                current_path.append((x2, y2))
                            elif current_path[-1] == (x2, y2):
                                current_path.append((x1, y1))
                    # radius += 0.
                    if multiplicity > 1:
                        vec = np.array(current_path[2]) - np.array(current_path[0])
                        orthogonal_vec = np.array([-vec[1], vec[0]])
                        scale = 0.125
                        fraction = -1 + 2 * (_ / (multiplicity - 1))
                        current_path[1] = tuple(np.array(current_path[1]) + scale * fraction * orthogonal_vec)
                    # if multiplicity > 1:
                    #     loops = False
                    # if p == 2:
                    #     current_path[1] = tuple(np.array(center) - 0.2*((np.array(current_path[0]) + np.array(current_path[2])) / 2)/np.linalg.norm(np.array(current_path[0]) + np.array(current_path[2])/2))
                    # else:
                    #     current_path[1] = tuple(np.array(center) - 0.5*((np.array(current_path[0]) + np.array(current_path[2])) / 2)/np.linalg.norm(np.array(current_path[0]) + np.array(current_path[2])/2))
                    # if p == 3:
                    #     radius -= 0.0
                    #     current_path[1] = tuple(np.array(center) - (0.8,0.1))
                    # if p == 2:
                    #     radius += 0.1
                    #     current_path[1] = tuple(np.array(center) + (0.8,0.1))
                    s = composite_points([(current_path, (center, radius))], type='bezier', loops=False)

                    color, seed, unpaired_paths = compute_color(color_map, path, unpaired_paths, self.edges, self.counterports, seed=seed, conjugacy_type=0)

                    # Pick a different color every time
                    # if p == 1:
                    #     p = 3
                    # elif p == 3:
                    #     p = 1
                    # np.random.seed(seed)
                    # seed += 1
                    # color = np.random.rand(3,)
                    # color = 'k'
                    # if p in [8]:
                    #     color = 'r'
                    # if p==0:
                    #     color = 'blue'
                    # elif p==3:
                    #     color = 'blue'
                    # elif p==1:
                    #     color = 'orange'
                    # elif p==2:
                    #     color = 'orange'
                    plot_curve_points(s, color=color, linewidth=4)
        else:
            for p, path in enumerate(self.permuted_solutions[solution][permutation]):
                current_path = []
                port_path = ()
                for i, (e, strand_id) in enumerate(path):
                    port_path += (e,)
                    n1, n2 = self.edges[e]
                    subnodes1 = subnode_positions[n1]
                    subnodes2 = subnode_positions[n2]

                    # Find the first available sub-nodes for each node
                    if n1 in self.ports:
                        i1 = strand_id
                    else:
                        i1 = next(i for i, count in enumerate(used_subnodes[n1]) if count < (2 if n1 not in self.ports else 1))
                    if n2 in self.ports:
                        i2 = strand_id
                    else:
                        i2 = next(i for i, count in enumerate(used_subnodes[n2]) if count < (2 if n2 not in self.ports else 1))

                    # Mark these sub-nodes as used
                    used_subnodes[n1][i1] += 1
                    used_subnodes[n2][i2] += 1

                    # Connect the selected sub-nodes
                    x1, y1 = subnodes1[i1]
                    x2, y2 = subnodes2[i2]
                    # plt.plot([x1, x2], [y1, y2], color='black', alpha=0.7, zorder=0)
                    if i == 0:
                        current_path.append((x1, y1))
                        current_path.append((x2, y2))
                    else:
                        if current_path[-2] == (x1, y1):
                            current_path = current_path[:-1]
                            current_path.append((x2, y2))
                        elif current_path[-2] == (x2, y2):
                            current_path = current_path[:-1]
                            current_path.append((x1, y1))
                        elif current_path[-1] == (x1, y1):
                            current_path.append((x2, y2))
                        elif current_path[-1] == (x2, y2):
                            current_path.append((x1, y1))
                # radius += 0.
                # if multiplicity > 1:
                #     if _ > 0:
                #         current_path[1] = (0.2,-0.2)
                #     else:
                #         current_path[1] = (-0.2,0.2)
                # if multiplicity > 1:
                #     loops = False
                # if p == 2:
                #     current_path[1] = tuple(np.array(center) - 0.2*((np.array(current_path[0]) + np.array(current_path[2])) / 2)/np.linalg.norm(np.array(current_path[0]) + np.array(current_path[2])/2))
                # else:
                #     current_path[1] = tuple(np.array(center) - 0.5*((np.array(current_path[0]) + np.array(current_path[2])) / 2)/np.linalg.norm(np.array(current_path[0]) + np.array(current_path[2])/2))
                # if p == 3:
                #     radius -= 0.0
                #     current_path[1] = tuple(np.array(center) - (0.8,0.1))
                # if p == 2:
                #     radius += 0.1
                #     current_path[1] = tuple(np.array(center) + (0.8,0.1))
                
                loop = False
                if loops:
                    if [p] in self.loops[(solution, permutation)]: # Here I am not considering loops with reentrant paths yet
                        loop = True
                # radius += 0.025
                radius += 0.03
                s = composite_points([(current_path, (center, radius))], type='bezier', loops=loop)
                
                inter_all = []
                for s_i in s_all:
                    inter_i = find_curve_intersections(s_i, s)
                    # if inter_i:
                    #     plt.scatter(*zip(*inter_i), color='black', s=500, zorder=100)
                    inter_all.extend(inter_i)
                if inter_all:
                    s = resample_curve_points(s, 250, must_include=inter_all)
                    # s = resample_curve_points(s, 250)
                else:
                    s = resample_curve_points(s, 250)
                s_all.append(s)

                color, seed, unpaired_paths = compute_color(color_map, port_path, unpaired_paths, self.edges, self.counterports, seed=seed, conjugacy_type=0)

                
                # Pick a different color every time
                # if p == 1:
                #     p = 3
                # elif p == 3:
                #     p = 1
                # np.random.seed(starting_seed+p)
                # color = np.random.rand(3,)
                # if p==0:
                #     color = 'blue'
                # elif p==3:
                #     color = 'blue'
                # elif p==1:
                #     color = 'orange'
                # elif p==2:
                #     color = 'orange'
                if not loops:
                    plot_curve_points(s, color=color, linewidth=4)
                else:
                    plot_curve_points(s, color=color, linewidth=4)
                    # plot_curve_with_linking_highlights(s, [(s_i, 1) for s_i in s_all], color=color, linewidth=4.33)

        # Plot nodes and sub-nodes
        fontsize = 25
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
                center = (x, y)
                node_type = 'j'
                node_id = node - num_ports + 1
            if self.status[node] == 'suppressed':
                color = 'lightgrey'
            else:
                color = 'black'
            if permutation is None:
                # Plot the nodes
                if node_type == 'p':
                    plt.scatter(x, y, color=color, s=1000, zorder=2)
                else:
                    plt.scatter(x, y, color='white', s=10000, zorder=1, linewidths=4, edgecolor='black')

                x, y = x*1.225, y*1.225  # Adjust position for text

                if node_type =='p':
                    x, y = x*1.1, y*1.1  # Adjust position for text
                    plt.text(x, y, f"${node_type}_{node_id}$", fontsize=fontsize, ha='center', va='center', zorder=3, color=color)
                else:
                    plt.text(x, y, f"${node_type}_{node_id}$", fontsize=fontsize, ha='center', va='center', zorder=3, color=color)

            if permutation is not None:
                # Plot sub-nodes
                if node_type == 'p':
                    for i, (sx, sy) in enumerate(subnode_positions[node]):
                        plt.scatter(sx, sy, color=color, s=250, zorder=max_zorder+1)
                        sx, sy = sx*1.35, sy*1.35  # Adjust position for text
                        angle = np.arctan2(sy, sx)
                        plt.text(sx, sy, fr"$s_{{{node_type}_{node_id},{i+1}}}$", fontsize=fontsize, ha='center', va='center', zorder=max_zorder+2, color=color, rotation=np.degrees(angle))
                else:
                    plt.scatter(x, y, color='white', s=10000, zorder=1, linewidths=3.33, edgecolor='black')
                #     # x, y = x - 0.3, y + 0.25
                    plt.text(x, y, fr"$s_{{{node_type}_{node_id},r}}$", fontsize=fontsize, ha='center', va='center', zorder=max_zorder+2, color=color)

        plt.axis('equal')
        plt.axis('off')
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        # print(f"xlim: {xlim}, ylim: {ylim}")
        plt.xlim(xlim[0]*1.25, xlim[1]*1.25)
        plt.ylim(ylim[0]*1.25, ylim[1]*1.25)
        plt.show()

    def save_solution_animation(self, filename='animation1.gif', solutions=None, permutations=None, loops=False, starting_seed=0, color_map=None, interval=1000, dpi=150):
        """
        Save an animation (mp4/gif) of frames generated by plot_solution_v1 for different solutions and permutations.

        Args:
            filename (str): Output file name (should end with .mp4 or .gif).
            solutions (list): List of solution indices to animate. If None, animate all.
            permutations (list): List of permutation indices for each solution (or None).
            loops (bool): Whether to highlight loops.
            starting_seed (int): Seed for color generation.
            color_map (str): Color map to use.
            interval (int): Delay between frames in ms.
            dpi (int): Dots per inch for output.
        """
        import matplotlib.pyplot as plt

        if solutions is None:
            solutions = list(range(len(self.solutions)))
        if permutations is None:
            permutations = [None] * len(solutions)
        frames = list(zip(solutions, permutations))

        fig = plt.figure(figsize=(10, 10))

        def draw_frame(idx):
            plt.clf()
            sol_idx, perm_idx = frames[idx]
            self.plot_solution_v1(sol_idx, permutation=perm_idx, loops=loops, starting_seed=starting_seed, color_map=color_map)
            # plt.title(f"Solution {sol_idx+1}, Permutation {perm_idx if perm_idx is not None else 'None'}", fontsize=18)

        anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=interval, repeat=True)
        anim.save(filename, writer='imagemagick', dpi=dpi)
        # anim.save(filename, writer='ffmpeg', dpi=dpi)

    def plot_solution_v2(self, solution, permutation=None, loops=False, starting_seed=0, ax=None, color_map=None):
        """
        Plot a 2D representation of a solution, where nodes are placed on circumferences
        and edges are drawn based on the paths in the solution. Each node (port or junction)
        has sub-nodes (strands) equal to self.N times the number of edges involving that node.

        Args:
            solution: A set of edges representing a path and their multiplicities.
            ax: Optional matplotlib Axes to plot into (for subplots).
        """
        if color_map is None:
            if self.conjugacy:
                color_map = 'conjugate'
        elif color_map == 'conjugate':
            if not self.conjugacy:
                raise ValueError("Color map 'conjugate' is only available for conjugate solutions.")

        # Use provided axes or get current axes
        if ax is None:
            ax = plt.gca()

        # Separate ports and junctions
        all_nodes = set(self.ports) | set(self.junctions)
        junctions = set(self.junctions)

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
            num_ports_per_junction = len(self.junction2ports()[junction])
            for _, port in enumerate(self.junction2ports()[junction]):
                if self.counterports[port] not in port_positions.keys():
                    angle = 2 * np.pi * (i / num_ports)
                    # angle = 2 * np.pi * (i / (num_ports_per_junction + num_junctions - 1)) - np.pi/3
                    i += 1
                    port_positions[port] = (junction_positions[junction][0] + np.cos(angle), junction_positions[junction][1] + np.sin(angle))
        if num_junctions == 1:
            counterport_positions = {
                self.counterports[port]: (-x, -y) for port, (x, y) in port_positions.items()
            }
        else:
            counterport_positions = {}
            for port, (x, y) in port_positions.items():
                linked_junctions = self.find_minimum_internal_nodes(self.edges, port, self.counterports[port])
                if len(linked_junctions) == 1:
                    counterport_positions[self.counterports[port]] = (x, -y)
                else:
                    counterport_positions[self.counterports[port]] = (-x, -y)
        positions = {**port_positions, **counterport_positions, **junction_positions}

        # Compute sub-node positions for ports and junctions
        subnode_positions = {}
        for node in all_nodes:
            num_edges = sum(1 for edge in self.edges.values() if node in edge)
            if num_edges == 0:
                num_edges = 1  # For suppressed ports
            if node in self.ports:
                num_subnodes = self.N * num_edges
                base_x, base_y = positions[node]
                if num_subnodes == 1:
                    subnode_positions[node] = [(base_x, base_y)]
                else:
                    if permutation is not None:
                        tangent_length = 0.5
                    else:
                        tangent_length = 0
                    angle = np.arctan2(base_y, base_x)
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
                radius = 0.
                angle_offset = np.pi / num_subnodes / 2 + np.arctan2(base_y, base_x) + np.pi / len(junctions)
                angle_section = 2 * np.pi / len(junctions)
                subnode_positions[node] = [
                    (
                        base_x + radius * np.cos(angle_offset + angle_section * (j / num_subnodes)),
                        base_y + radius * np.sin(angle_offset + angle_section * (j / num_subnodes))
                    )
                    for j in range(num_subnodes)
                ]

        center = (0, 0)

        # Plot edges using sub-nodes
        used_subnodes = {node: [0] * len(subnodes) for node, subnodes in subnode_positions.items()}
        radius = 1.35
        s_all = []
        unpaired_paths = {}
        seed = starting_seed
        if permutation is None:
            for p, (path, multiplicity) in enumerate(list(self.solutions)[solution]):
                for _ in range(multiplicity):
                    current_path = []
                    for i, e in enumerate(path):
                        n1, n2 = self.edges[e]
                        subnodes1 = subnode_positions[n1]
                        subnodes2 = subnode_positions[n2]
                        i1 = next(i for i, count in enumerate(used_subnodes[n1]) if count < (2 if n1 not in self.ports else 1))
                        i2 = next(i for i, count in enumerate(used_subnodes[n2]) if count < (2 if n2 not in self.ports else 1))
                        used_subnodes[n1][i1] += 1
                        used_subnodes[n2][i2] += 1
                        x1, y1 = subnodes1[i1]
                        x2, y2 = subnodes2[i2]
                        if i == 0:
                            current_path.append((x1, y1))
                            current_path.append((x2, y2))
                        else:
                            if current_path[-2] == (x1, y1):
                                current_path = current_path[:-1]
                                current_path.append((x2, y2))
                            elif current_path[-2] == (x2, y2):
                                current_path = current_path[:-1]
                                current_path.append((x1, y1))
                            elif current_path[-1] == (x1, y1):
                                current_path.append((x2, y2))
                            elif current_path[-1] == (x2, y2):
                                current_path.append((x1, y1))
                    if multiplicity > 1:
                        vec = np.array(current_path[2]) - np.array(current_path[0])
                        orthogonal_vec = np.array([-vec[1], vec[0]])
                        scale = 0.125
                        fraction = -1 + 2 * (_ / (multiplicity - 1))
                        current_path[1] = tuple(np.array(current_path[1]) + scale * fraction * orthogonal_vec)
                    s = composite_points([(current_path, (center, radius))], type='bezier', loops=False)
                    color, seed, unpaired_paths = compute_color(color_map, path, unpaired_paths, self.edges, self.counterports, seed=seed, conjugacy_type=0)
                    plot_curve_points(s, color=color, linewidth=3)
        else:
            for p, path in enumerate(self.permuted_solutions[solution][permutation]):
                current_path = []
                port_path = ()
                for i, (e, strand_id) in enumerate(path):
                    port_path += (e,)
                    n1, n2 = self.edges[e]
                    subnodes1 = subnode_positions[n1]
                    subnodes2 = subnode_positions[n2]
                    if n1 in self.ports:
                        i1 = strand_id
                    else:
                        i1 = next(i for i, count in enumerate(used_subnodes[n1]) if count < (2 if n1 not in self.ports else 1))
                    if n2 in self.ports:
                        i2 = strand_id
                    else:
                        i2 = next(i for i, count in enumerate(used_subnodes[n2]) if count < (2 if n2 not in self.ports else 1))
                    used_subnodes[n1][i1] += 1
                    used_subnodes[n2][i2] += 1
                    x1, y1 = subnodes1[i1]
                    x2, y2 = subnodes2[i2]
                    if i == 0:
                        current_path.append((x1, y1))
                        current_path.append((x2, y2))
                    else:
                        if current_path[-2] == (x1, y1):
                            current_path = current_path[:-1]
                            current_path.append((x2, y2))
                        elif current_path[-2] == (x2, y2):
                            current_path = current_path[:-1]
                            current_path.append((x1, y1))
                        elif current_path[-1] == (x1, y1):
                            current_path.append((x2, y2))
                        elif current_path[-1] == (x2, y2):
                            current_path.append((x1, y1))
                loop = False
                if loops:
                    if [p] in self.loops[(solution, permutation)]:
                        loop = True
                s = composite_points([(current_path, (center, radius))], type='bezier', loops=loop)
                inter_all = []
                for s_i in s_all:
                    inter_i = find_curve_intersections(s_i, s)
                    inter_all.extend(inter_i)
                if inter_all:
                    s = resample_curve_points(s, 250)
                else:
                    s = resample_curve_points(s, 250)
                s_all.append(s)
                color, seed, unpaired_paths = compute_color(color_map, port_path, unpaired_paths, self.edges, self.counterports, seed=seed, conjugacy_type=0)
                if not loops:
                    plot_curve_points(s, color=color, linewidth=3)
                else:
                    plot_curve_with_linking_highlights(s, [(s_i, 1) for s_i in s_all], color=color, linewidth=3)

        # Plot nodes and sub-nodes
        zorders = [artist.get_zorder() for artist in ax.get_children()]
        max_zorder = max(zorders) if zorders else 1
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
                center = (x, y)
                node_type = 'j'
                node_id = node - num_ports + 1
            if self.status[node] == 'suppressed':
                color = 'lightgrey'
            else:
                color = 'black'
            if permutation is None:
                if node_type == 'p':
                    ax.scatter(x, y, color=color, s=250, zorder=2)
                else:
                    # ax.scatter(x, y, color='white', s=500, zorder=1, linewidths=4, edgecolor='black')
                    x, y = x+0.05, y-0.05
                x, y = x*1.225, y*1.225
                # ax.text(x, y, f"${node_type}_{node_id}$", fontsize=50, ha='center', va='center', zorder=3, color=color)

            if permutation is not None:
                if node_type == 'p':
                    for i, (sx, sy) in enumerate(subnode_positions[node]):
                        ax.scatter(sx, sy, color=color, s=100, zorder=max_zorder+1)
                # else:
                #     ax.scatter(x, y, color='white', s=500, zorder=1, linewidths=4, edgecolor='black')
                        
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        # ax.set_xlim(-1.15, 1.15)
        # ax.set_ylim(-1.15, 1.15)

    def plot_multiple_solutions(self, solutions=None, permutations=None, loops=False, starting_seed=0, nrows=None, ncols=None, figsize=(10, 10), color_map=None):
        """
        Plot multiple solutions, each in a subplot of the same figure.

        Args:
            solutions: List of solution indices to plot. If None, plot all.
            permutations: List of permutation indices for each solution (or None).
            loops: Whether to highlight loops.
            starting_seed: Seed for color generation.
            nrows: Number of subplot rows (optional).
            ncols: Number of subplot columns (optional).
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt

        if solutions is None:
            solutions = list(range(len(self.solutions)))
        if permutations is None:
            permutations = [None] * len(solutions)
        num_plots = len(solutions)

        # Compute grid size if not given
        if nrows is None or ncols is None:
            ncols = math.ceil(math.sqrt(num_plots))
            nrows = math.ceil(num_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        # Latex is slow for plotting
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Computer Modern Roman"],
        # })

        for idx, (sol_idx, perm_idx) in enumerate(zip(solutions, permutations)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            plt.sca(ax)
            self.plot_solution_v2(sol_idx, permutation=perm_idx, loops=loops, starting_seed=starting_seed, color_map=color_map)
            if perm_idx is not None:
                ax.set_title(f"$({perm_idx+1})$", fontsize=32)
            else:
                ax.set_title(f"$({sol_idx+1})$", fontsize=32)
            ax.axis('off')
            

        # Hide unused subplots
        for idx in range(num_plots, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.subplots_adjust(wspace=0.15, hspace=0)

        plt.show()

    def plot_multiple_cardinality_strands_permutations(self, solutions=None, nrows=None, ncols=None, figsize=(10, 10)):
        """
        Plot the cardinalities (just a number) for each solution in a subplot grid,
        matching the layout of plot_multiple_solutions.
        """
        import matplotlib.pyplot as plt

        if solutions is None:
            solutions = list(range(len(self.solutions)))
        num_plots = len(solutions)

        # Compute grid size if not given
        if nrows is None or ncols is None:
            ncols = math.ceil(math.sqrt(num_plots))
            nrows = math.ceil(num_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Computer Modern Roman"],
        # })

        cardinalities = self.compute_cardinality_strands_permutations()
        for idx, sol_idx in enumerate(solutions):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            ax.axis('off')
            ax.text(0.5, 0.5, f"{int(cardinalities[sol_idx])}", fontsize=32, ha='center', va='center')
            ax.set_title(f"$({sol_idx+1})$", fontsize=32)

        # Hide unused subplots
        for idx in range(num_plots, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis('off')

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.5)
        plt.subplots_adjust(wspace=0.15, hspace=1)
        plt.show()
    
    def save(self, filename):
        """
        Save the current instance of the Topology class to a file using pickle.

        Args:
            filename (str): The path to the file where the instance will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load an instance of the Topology class from a file using pickle.

        Args:
            filename (str): The path to the file from which the instance will be loaded.

        Returns:
            Topology: The loaded Topology instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def export_topology(self, filename='topology.txt'):
        """
        Export the topology information to a text file.

        Args:
            filename (str): The path to the file where the topology will be saved.
        """
        with open(filename, 'w') as f:
            # Write node positions (points) of the lattice unit cell
            f.write("Node positions [<node_index>: <x>, <y>, <z>]:\n")
            for idx, point in enumerate(self.graph.points):
                if isinstance(point, np.ndarray):
                    coords = ", ".join(map(str, point.tolist()))
                else:
                    coords = str(point)
                f.write(f"{idx}: {coords}\n")
            f.write("\n")
            # Symmetries
            f.write("Symmetries [<node_index_1>, <node_index_2>, ...]\n")
            for sym in self.graph.symmetries:
                f.write(f"{str(sym).replace('[','').replace(']','')}\n")
            f.write("\n")
            # Write edges with their corresponding node indices
            f.write("Edges [<edge_index>: <node_index_1>, <node_index_2>]\n")
            for edge_id, (n1, n2) in self.edges.items():
                f.write(f"{edge_id}: {n1}, {n2}\n")
            f.write("\n")
            # Write solutions
            f.write("Port-level solutions [P<port_level_realization_index>: <multiplicity>, <edge_index_1>, <edge_index_2>, ...]:\n")
            for sol_idx, sol in enumerate(self.solutions):
                f.write(f"P{sol_idx}:\n")
                for path, multiplicity in sol:
                    f.write(f"{multiplicity}, {str(path).replace('(','').replace(')','')}\n")
            f.write("\n")
            # Write permutations
            f.write("Strand-level permutations [P<port_level_realization_index>S<permutation_index>: <edge_index_1>, <strand_id_1>, <edge_index_2>, <strand_id_2>, ...]:\n")
            for sol_idx in self.permuted_solutions:
                # print(self.permuted_solutions[sol_idx])
                perms = self.permuted_solutions[sol_idx]
                for perm_idx in perms:
                    f.write(f"P{sol_idx}S{perm_idx}:\n")
                    perm = perms[perm_idx]
                    for edge_id_strand_id in perm:
                        f.write(f"{str(edge_id_strand_id).replace('[','').replace('(','').replace(')','').replace(']','')}\n")





if __name__ == '__main__':

    p_sol = True

    # Instantiate Topology object
    topology = Topology('lattice_triangular.dat', N=1, conjugacy=True, max_multiplicity=None)

    # Plot the lattice with nodes and edge labels
    # topology.graph.plot_lattice()

    # Now compute multiplicities

    # First: generate solutions
    topology.generate(stop_early=False)
    mults = topology.compute_solution_multiplicities()


    # Print results
    for unique_solution, multiplicity in mults.items():
        print(f"Unique solution: {unique_solution}")
        print(f"Multiplicity: {multiplicity}\n")

    # Generate topology port-level solutions (no strand-level permutations yet)
    topology.generate(stop_early=False, discard_threshold=None, restart=1)
    print(f"Number of solutions: {len(topology.solutions)}")

    # # check what topology has
    print('topology.ports:', topology.ports)
    # print('topology.counterports:', topology.counterports)
    print('topology.junctions:', topology.junctions)
    # print('topology.edges:', topology.edges)
    print('topology.solutions:', topology.solutions)
    print('length of solutions:', len(topology.solutions))

    # Plot the first port-level solution
    # topology.plot_solution_v1(solution=0)
    # topology.plot_solution_v1(solution=1)

    # Generate strand-level permutations for the port-level solution 5
    # (See B.3 in the paper, indices start from 0 in Python)
    topology.instantiate_strands_v1(0) # <- this might take a while to compute,
                                       #    if you're interested only in the first N permutations
                                       #    you can modify the code to stop after computing those with break_early=N.
    print(f"Number of strand-level permutations for solution 5: {len(topology.permuted_solutions[0])}")
    # print('strand level solutions', topology.permuted_solutions)
    # print('checking function instantiate', topology.instantiate_strands_v1(1))

    # Plot the first strand-level permutation for solution 5
    # topology.plot_solution_v1(solution=0, permutation=0)