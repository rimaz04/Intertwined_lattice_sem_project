from collections import Counter, defaultdict
from itertools import combinations, permutations, product
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from symmetry import find_symmetries, read_lattice_file
from plotting import bspline_curve_points


def all_unique_sequences(elements, min_length=2, max_length=None):
    """Return unique ordered subsets, with reverse paths treated as identical."""
    elements = list(elements)
    sequences = set()
    if max_length is None:
        max_length = len(elements)

    for subset_size in range(min_length, max_length + 1):
        for subset in combinations(elements, subset_size):
            for perm in permutations(subset):
                sequences.add(min(perm, perm[::-1]))

    return sorted(sequences, key=lambda x: (len(x), x))


def are_conjugate_paths(path1, path2, edges, counterports, conjugacy_type=0):
    """Conjugacy check for port-level paths."""
    if conjugacy_type != 0:
        raise ValueError("Invalid conjugacy type")

    s1 = set(edges[path1[0]]) & set(counterports[p] for p in edges[path2[0]] if p in counterports)
    s2 = set(edges[path1[-1]]) & set(counterports[p] for p in edges[path2[-1]] if p in counterports)
    s3 = set(edges[path1[0]]) & set(counterports[p] for p in edges[path2[-1]] if p in counterports)
    s4 = set(edges[path1[-1]]) & set(counterports[p] for p in edges[path2[0]] if p in counterports)

    if path1 == path2:
        return bool(s3 and next(iter(s3)) in counterports.keys())

    return (
        s1 and next(iter(s1)) in counterports.keys() and s2 and next(iter(s2)) in counterports.keys()
    ) or (
        s3 and next(iter(s3)) in counterports.keys() and s4 and next(iter(s4)) in counterports.keys()
    )


def is_conjugate(path_multiset, edges, counterports, conjugacy_type=0):
    """Check if each path has a conjugate partner in the multiset."""
    remaining = set(path_multiset)
    while remaining:
        path1, m1 = remaining.pop()
        if m1 > 1:
            remaining.add((path1, m1 - 1))

        if are_conjugate_paths(path1, path1, edges, counterports, conjugacy_type):
            continue

        found = False
        for path2, m2 in list(remaining):
            if are_conjugate_paths(path1, path2, edges, counterports, conjugacy_type):
                remaining.remove((path2, m2))
                if m2 > 1:
                    remaining.add((path2, m2 - 1))
                found = True
                break

        if not found:
            return False

    return True


def normalize_edge_targets(edge_ids, edge_targets, default_edge_target=0):
    """Normalize int/list/dict target input into {edge_id: target} mapping."""
    edge_ids = list(edge_ids)

    if isinstance(edge_targets, int):
        return {e: edge_targets for e in edge_ids}

    if isinstance(edge_targets, (list, tuple)):
        if len(edge_targets) != len(edge_ids):
            raise ValueError(f"Expected {len(edge_ids)} edge targets, got {len(edge_targets)}")
        return {e: int(edge_targets[i]) for i, e in enumerate(edge_ids)}

    if isinstance(edge_targets, dict):
        return {e: int(edge_targets.get(e, default_edge_target)) for e in edge_ids}

    raise TypeError("edge_targets must be int, list/tuple, or dict")


def condition_fn_per_edge(collection, edge_ids, edge_targets):
    """Validate exact edge usage counts for a collection of paths."""
    counts = Counter({e: 0 for e in edge_ids})
    for path, mult in collection.items():
        for e in path:
            counts[e] += mult
    return all(counts[e] == edge_targets[e] for e in edge_ids)


def generate_valid_multiplicities_v2(sequences, edge_targets, edge_ids, max_multiplicity_path=None):
    """Yield multiplicity vectors that satisfy heterogeneous per-edge targets exactly."""
    edge_ids = list(edge_ids)
    if max_multiplicity_path is None:
        max_multiplicity_path = max(edge_targets.values()) if edge_targets else 0

    residual = {e: edge_targets[e] for e in edge_ids}
    n_seq = len(sequences)

    def backtrack(idx, current):
        if idx == n_seq:
            if all(residual[e] == 0 for e in edge_ids):
                yield tuple(current)
            return

        seq_edges = set(sequences[idx])
        max_allowed = min([max_multiplicity_path] + [residual[e] for e in seq_edges])

        for m in range(max_allowed + 1):
            for e in seq_edges:
                residual[e] -= m

            if all(residual[e] >= 0 for e in edge_ids):
                yield from backtrack(idx + 1, current + [m])

            for e in seq_edges:
                residual[e] += m

    yield from backtrack(0, [])


class Graph:
    def __init__(self, points, connections, types, status):
        self.points = points
        self.connections = connections
        self.types = types
        self.status = status
        self.ports = [i for i, t in enumerate(types) if t == "port" or t == "counterport"]
        self.junctions = [i for i, t in enumerate(types) if t == "junction"]
        self.edges = self.generate_edges()
        self.counterports = self.generate_counterports()
        self.symmetries = find_symmetries(points, status)

    def generate_edges(self):
        edges = defaultdict(int)
        for i, (n1, n2) in enumerate(self.connections):
            edges[i] = tuple(sorted((n1, n2)))
        return edges

    def generate_counterports(self):
        counterports = {}
        p = sum(1 for t in self.types if t == "port")
        for i, t in enumerate(self.types):
            if t == "port":
                counterports[i] = i + p
            elif t == "counterport":
                counterports[i] = i - p
        return counterports


class TopologyMultiEdge:
    """
    Topology generator with heterogeneous edge multiplicities.

    This class keeps the same high-level flow as topology2.py:
    valid paths -> multiplicity search -> optional conjugacy -> symmetry dedup.
    """

    def __init__(
        self,
        filename,
        edge_targets,
        conjugacy=False,
        max_multiplicity_path=None,
        power_set_min_length=2,
        power_set_max_length=2,
        default_edge_target=0,
    ):
        self.graph = Graph(*read_lattice_file(filename))
        self.ports = self.graph.ports
        self.junctions = self.graph.junctions
        self.edges = self.graph.edges
        self.status = self.graph.status
        self.counterports = self.graph.counterports
        self.symmetries = self.graph.symmetries

        self.power_set_min_length = power_set_min_length
        self.power_set_max_length = power_set_max_length
        self.conjugacy = conjugacy

        self.edge_targets = normalize_edge_targets(
            self.edges.keys(), edge_targets, default_edge_target=default_edge_target
        )
        self.max_multiplicity_path = max_multiplicity_path

        self.valid_collections = set()
        self.solutions = []
        self.strand_solutions = {}

    def is_valid_path(self, path):
        """Same path validity logic as original port-level solver."""
        if len(path) > 2 and set(self.edges[path[0]]) & set(self.edges[path[-1]]):
            if any(n in self.ports for e in path for n in self.edges[e]):
                return False
            if not all(set(self.edges[path[i]]) & set(self.edges[path[i + 1]]) for i in range(len(path) - 1)):
                return False
        else:
            if len(path) < 2:
                return False
            if not all(set(self.edges[path[i]]) & set(self.edges[path[i + 1]]) for i in range(len(path) - 1)):
                return False
            if len(path) > 2 and any(
                set(self.edges[path[i]]) & set(self.edges[path[i + 2]]) for i in range(len(path) - 2)
            ):
                return False
            if not (
                any(n in self.ports for n in self.edges[path[0]])
                and any(n in self.ports for n in self.edges[path[-1]])
            ):
                return False
        return True

    def prune_redundant_loops(self, paths):
        """Remove cyclic/reversed duplicates among junction-only loops."""
        pruned_paths = []
        seen_loops = set()
        for path in paths:
            is_loop = len(path) > 2 and set(self.edges[path[0]]) & set(self.edges[path[-1]])
            if not is_loop:
                pruned_paths.append(path)
                continue

            loop_variants = set()
            plen = len(path)
            for offset in range(plen):
                rotated = tuple(path[offset:] + path[:offset])
                reversed_rotated = tuple(reversed(rotated))
                loop_variants.add(rotated)
                loop_variants.add(reversed_rotated)

            if seen_loops & loop_variants:
                continue

            seen_loops.update(loop_variants)
            pruned_paths.append(path)

        return pruned_paths

    def generate_new_edge_mapping(self, node_mapping):
        edge_to_id = {tuple(sorted(edge)): edge_id for edge_id, edge in self.edges.items()}
        edge_mapping = {}
        for edge_id, (a, b) in self.edges.items():
            mapped_edge = tuple(sorted((node_mapping[a], node_mapping[b])))
            if mapped_edge not in edge_to_id:
                raise KeyError(mapped_edge)
            edge_mapping[edge_id] = edge_to_id[mapped_edge]
        return edge_mapping

    def apply_permutation(self, path_set, permutation):
        mapping_nodes = {orig: perm for orig, perm in enumerate(permutation)}
        try:
            mapping = self.generate_new_edge_mapping(mapping_nodes)
        except KeyError:
            return None

        transformed = frozenset(
            {
                (
                    (tuple(mapping[e] for e in path), count)
                    if mapping[path[0]] < mapping[path[-1]]
                    else (tuple(mapping[e] for e in reversed(path)), count)
                )
                for path, count in path_set
            }
        )
        return transformed

    def find_unique_sets(self, path_sets):
        unique_sets = set()
        for path_set in path_sets:
            equivalent_representations = {
                transformed
                for eq in self.symmetries
                for transformed in [self.apply_permutation(path_set, eq)]
                if transformed is not None
            }
            if not equivalent_representations:
                equivalent_representations = {frozenset(path_set)}

            if not unique_sets.intersection(equivalent_representations):
                unique_sets.add(frozenset(path_set))

        return unique_sets

    def _frozen_collection_matches_targets(self, frozen_collection):
        """Return True if the frozen collection satisfies exact per-edge targets."""
        collection = Counter(dict(frozen_collection))
        return condition_fn_per_edge(collection, self.edges.keys(), self.edge_targets)

    def verify_all_solutions(self):
        """Verify each stored solution satisfies exact edge targets."""
        invalid = []
        for idx, sol in enumerate(self.solutions):
            if not self._frozen_collection_matches_targets(sol):
                invalid.append(idx)
        return invalid

    def generate_multiplicity_collections(self, sequences, stop_early=False):
        edge_ids = list(self.edges.keys())
        valid_collections = set()

        multiplicity_iter = generate_valid_multiplicities_v2(
            sequences=sequences,
            edge_targets=self.edge_targets,
            edge_ids=edge_ids,
            max_multiplicity_path=self.max_multiplicity_path,
        )

        valid = 0
        for multiplicities in multiplicity_iter:
            collection = Counter({seq: m for seq, m in zip(sequences, multiplicities) if m > 0})

            if not condition_fn_per_edge(collection, edge_ids, self.edge_targets):
                continue

            frozen = frozenset(collection.items())
            if self.conjugacy and not is_conjugate(frozen, self.edges, self.counterports, 0):
                continue

            valid_collections.add(frozen)
            valid += 1
            if stop_early and valid >= stop_early:
                break

        return valid_collections

    def generate_solutions(self, stop_early=False):
        all_seqs = all_unique_sequences(
            self.edges.keys(),
            min_length=self.power_set_min_length,
            max_length=self.power_set_max_length,
        )
        valid_paths = [s for s in all_seqs if self.is_valid_path(s)]
        valid_paths = self.prune_redundant_loops(valid_paths)

        self.valid_collections = self.generate_multiplicity_collections(valid_paths, stop_early=stop_early)
        unique = self.find_unique_sets(self.valid_collections)

        # Deterministic ordering for stable solution indices when plotting.
        self.solutions = sorted(unique, key=lambda s: str(sorted(list(s))))

        invalid = self.verify_all_solutions()
        if invalid:
            raise ValueError(
                f"Internal consistency error: solutions at indices {invalid} do not match edge targets {self.edge_targets}."
            )

        return self.solutions

    def _path_to_ordered_nodes(self, path):
        """Convert path as edge IDs into an ordered node list for polyline plotting."""
        edge_nodes = [self.edges[e] for e in path]
        if len(edge_nodes) == 1:
            return [edge_nodes[0][0], edge_nodes[0][1]]

        first = edge_nodes[0]
        second = edge_nodes[1]
        if first[0] in second:
            ordered = [first[1], first[0]]
        elif first[1] in second:
            ordered = [first[0], first[1]]
        else:
            ordered = [first[0], first[1]]

        for e in edge_nodes[1:]:
            current = ordered[-1]
            if e[0] == current:
                ordered.append(e[1])
            elif e[1] == current:
                ordered.append(e[0])
            else:
                # Fallback for numerically valid but orientation-ambiguous transitions.
                ordered.append(e[0])
                if e[1] != e[0]:
                    ordered.append(e[1])

        return ordered

    def _get_curved_path_points(self, path, pos2d, num_points=50):
        """
        Generate smooth curved path points between nodes.
        
        Args:
            path: Path as edge IDs
            pos2d: Dict mapping node IDs to (x, y) coordinates
            num_points: Number of points to sample along the curve
            
        Returns:
            List of (x, y) tuples for the curved path
        """
        node_seq = self._path_to_ordered_nodes(path)
        if len(node_seq) < 2:
            return [(pos2d[node_seq[0]][0], pos2d[node_seq[0]][1])]
        
        # For simple 2-node paths, add a control point for a gentle curve
        if len(node_seq) == 2:
            p1 = np.array(pos2d[node_seq[0]])
            p2 = np.array(pos2d[node_seq[1]])
            midpoint = (p1 + p2) / 2
            
            # Add perpendicular offset for curve control point
            direction = p2 - p1
            perp = np.array([-direction[1], direction[0]])
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-6:
                perp = perp / perp_norm
            
            # Offset by a small amount proportional to edge length
            offset_dist = np.linalg.norm(direction) * 0.1
            control_pt = midpoint + perp * offset_dist
            
            control_points = [tuple(p1), tuple(control_pt), tuple(p2)]
        else:
            # For longer paths, use the node sequence as control points
            control_points = [pos2d[node] for node in node_seq]
        
        # Generate smooth B-spline curve through control points
        try:
            curve_pts = bspline_curve_points(control_points, degree=min(3, len(control_points)-1), num_points=num_points)
            return curve_pts
        except (ValueError, RuntimeError):
            # Fallback to linear interpolation if B-spline fails
            all_pts = []
            for i in range(len(node_seq)):
                all_pts.append(pos2d[node_seq[i]])
            return all_pts

    def plot_port_solution(self, solution_index=0, figsize=(8, 8), show_edge_ids=True):
        """Plot one port-level solution (no strand permutations)."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        if solution_index < 0 or solution_index >= len(self.solutions):
            raise IndexError(f"solution_index {solution_index} out of range [0, {len(self.solutions)-1}]")

        fig, ax = plt.subplots(figsize=figsize)

        pos2d = {
            i: (self.graph.points[i][0], self.graph.points[i][1])
            for i in range(len(self.graph.points))
        }

        # Draw base graph lightly for context.
        g = nx.Graph()
        for node_id in range(len(self.graph.points)):
            g.add_node(node_id)
        for _, (n1, n2) in self.edges.items():
            g.add_edge(n1, n2)

        nx.draw_networkx_edges(g, pos=pos2d, ax=ax, width=1.0, edge_color="lightgray", alpha=0.9)

        port_nodes = [i for i, t in enumerate(self.graph.types) if t == "port"]
        counterport_nodes = [i for i, t in enumerate(self.graph.types) if t == "counterport"]
        junction_nodes = [i for i, t in enumerate(self.graph.types) if t == "junction"]

        nx.draw_networkx_nodes(g, pos=pos2d, nodelist=port_nodes, node_color="tab:red", node_size=80, ax=ax)
        nx.draw_networkx_nodes(g, pos=pos2d, nodelist=counterport_nodes, node_color="tab:blue", node_size=80, ax=ax)
        nx.draw_networkx_nodes(g, pos=pos2d, nodelist=junction_nodes, node_color="black", node_size=40, ax=ax)

        if show_edge_ids:
            edge_labels = {}
            for e_id, (n1, n2) in self.edges.items():
                edge_labels[(n1, n2)] = str(e_id)
            nx.draw_networkx_edge_labels(g, pos=pos2d, edge_labels=edge_labels, font_size=8, ax=ax)

        solution = list(self.solutions[solution_index])
        cmap = plt.cm.get_cmap("tab20", max(len(solution), 1))

        for idx, (path, multiplicity) in enumerate(solution):
            for rep in range(multiplicity):
                # Get curved path points
                curve_pts = self._get_curved_path_points(path, pos2d, num_points=50)
                xs = [pt[0] for pt in curve_pts]
                ys = [pt[1] for pt in curve_pts]

                # Slight jitter separates repeated same path multiplicity visually.
                jitter = 0.005 * (rep - (multiplicity - 1) / 2.0)
                ax.plot(
                    xs,
                    [y + jitter for y in ys],
                    color=cmap(idx),
                    linewidth=2.2,
                    alpha=0.95,
                )

        ax.set_title(f"Port-level solution {solution_index}")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def get_solution_edge_counts(self, solution_index):
        """Return per-edge usage counts for one solution."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        if solution_index < 0 or solution_index >= len(self.solutions):
            raise IndexError(f"solution_index {solution_index} out of range [0, {len(self.solutions)-1}]")

        counts = Counter({e: 0 for e in self.edges.keys()})
        for path, multiplicity in self.solutions[solution_index]:
            for e in path:
                counts[e] += multiplicity
        return counts

    def print_solution(self, solution_index):
        """Print one solution as edge paths and multiplicities."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        if solution_index < 0 or solution_index >= len(self.solutions):
            raise IndexError(f"solution_index {solution_index} out of range [0, {len(self.solutions)-1}]")

        print("=" * 70)
        print(f"Solution {solution_index}")
        print("Paths (edge sequence x multiplicity):")

        solution = sorted(list(self.solutions[solution_index]), key=lambda x: (len(x[0]), x[0], x[1]))
        for path, multiplicity in solution:
            path_str = " -> ".join(str(e) for e in path)
            print(f"  ({path_str})  x {multiplicity}")

        counts = self.get_solution_edge_counts(solution_index)
        print("Edge usage in this solution:")
        print(f"  {dict(sorted(counts.items()))}")
        print("Target edge multiplicities:")
        print(f"  {dict(sorted(self.edge_targets.items()))}")
        print(f"Exact match: {dict(sorted(counts.items())) == dict(sorted(self.edge_targets.items()))}")

    def print_all_solutions(self):
        """Print all computed solutions in a readable format."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        for idx in range(len(self.solutions)):
            self.print_solution(idx)

    def plot_all_port_solutions(self, ncols=3, figsize=(15, 5), show_edge_ids=False):
        """Plot all port-level solutions in a subplot grid."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")

        n_solutions = len(self.solutions)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n_solutions / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            if idx >= n_solutions:
                ax.axis("off")
                continue

            pos2d = {i: (self.graph.points[i][0], self.graph.points[i][1]) for i in range(len(self.graph.points))}
            g = nx.Graph()
            for node_id in range(len(self.graph.points)):
                g.add_node(node_id)
            for _, (n1, n2) in self.edges.items():
                g.add_edge(n1, n2)

            nx.draw_networkx_edges(g, pos=pos2d, ax=ax, width=1.0, edge_color="lightgray", alpha=0.9)

            port_nodes = [i for i, t in enumerate(self.graph.types) if t == "port"]
            counterport_nodes = [i for i, t in enumerate(self.graph.types) if t == "counterport"]
            junction_nodes = [i for i, t in enumerate(self.graph.types) if t == "junction"]

            nx.draw_networkx_nodes(g, pos=pos2d, nodelist=port_nodes, node_color="tab:red", node_size=60, ax=ax)
            nx.draw_networkx_nodes(g, pos=pos2d, nodelist=counterport_nodes, node_color="tab:blue", node_size=60, ax=ax)
            nx.draw_networkx_nodes(g, pos=pos2d, nodelist=junction_nodes, node_color="black", node_size=30, ax=ax)

            if show_edge_ids:
                edge_labels = {(n1, n2): str(e_id) for e_id, (n1, n2) in self.edges.items()}
                nx.draw_networkx_edge_labels(g, pos=pos2d, edge_labels=edge_labels, font_size=7, ax=ax)

            solution = list(self.solutions[idx])
            cmap = plt.cm.get_cmap("tab20", max(len(solution), 1))
            for pidx, (path, multiplicity) in enumerate(solution):
                for rep in range(multiplicity):
                    # Get curved path points
                    curve_pts = self._get_curved_path_points(path, pos2d, num_points=50)
                    xs = [pt[0] for pt in curve_pts]
                    ys = [pt[1] for pt in curve_pts]
                    
                    jitter = 0.005 * (rep - (multiplicity - 1) / 2.0)
                    ax.plot(xs, [y + jitter for y in ys], color=cmap(pidx), linewidth=2.0, alpha=0.95)

            ax.set_title(f"Solution {idx}")
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def _expand_solution_instances(self, solution_index, sort_paths=True):
        """Expand one port-level solution into individual path instances."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        if solution_index < 0 or solution_index >= len(self.solutions):
            raise IndexError(f"solution_index {solution_index} out of range [0, {len(self.solutions)-1}]")

        items = list(self.solutions[solution_index])
        if sort_paths:
            # Longer paths first usually prunes strand backtracking faster.
            items = sorted(items, key=lambda x: (-len(x[0]), x[0], x[1]))

        instances = []
        for path, multiplicity in items:
            for _ in range(multiplicity):
                instances.append(tuple(path))
        return instances

    def _canonicalize_strand_solution(self, strand_solution):
        """Canonicalize one strand assignment modulo per-edge strand relabeling."""
        relabel_per_edge = {}
        next_id_per_edge = {}
        normalized_paths = []

        for strand_path in strand_solution:
            normalized = []
            for e, sid in strand_path:
                if e not in relabel_per_edge:
                    relabel_per_edge[e] = {}
                    next_id_per_edge[e] = 0
                if sid not in relabel_per_edge[e]:
                    relabel_per_edge[e][sid] = next_id_per_edge[e]
                    next_id_per_edge[e] += 1
                normalized.append((e, relabel_per_edge[e][sid]))
            normalized_paths.append(tuple(normalized))

        return tuple(sorted(normalized_paths))

    def compute_strand_assignment_cardinality(self, solution_index):
        """Compute factorial estimate for strand instantiations of one port-level solution."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        if solution_index < 0 or solution_index >= len(self.solutions):
            raise IndexError(f"solution_index {solution_index} out of range [0, {len(self.solutions)-1}]")

        numerator = 1
        for e in self.edges.keys():
            numerator *= math.factorial(self.edge_targets[e])

        denominator = 1
        for _, multiplicity in self.solutions[solution_index]:
            denominator *= math.factorial(multiplicity)

        return numerator // denominator

    def generate_strand_solutions(
        self,
        solution_index,
        break_early=None,
        sort_paths=True,
        quotient_by_relabeling=False,
    ):
        """
        Enumerate unique strand-level assignments for one port-level solution.

        Returns a list where each strand solution is a tuple of strand paths:
            ((edge_id, strand_id), (edge_id, strand_id), ...)

        By default, solutions are unique up to path multiplicity ordering only,
        matching the original instantiate_strands_v1 semantics.
        Set quotient_by_relabeling=True only if you explicitly want to merge
        solutions that differ by per-edge strand ID renaming.
        """
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")
        if solution_index < 0 or solution_index >= len(self.solutions):
            raise IndexError(f"solution_index {solution_index} out of range [0, {len(self.solutions)-1}]")

        edge_counts = self.get_solution_edge_counts(solution_index)
        if dict(sorted(edge_counts.items())) != dict(sorted(self.edge_targets.items())):
            raise ValueError(
                "Port-level solution does not match edge targets, cannot instantiate strand-level solution."
            )

        path_terms = list(self.solutions[solution_index])
        if sort_paths:
            # Longer paths first typically improves pruning.
            path_terms = sorted(path_terms, key=lambda x: (-len(x[0]), x[0], x[1]))

        edge_ids = list(self.edges.keys())
        initial_available = {e: list(range(self.edge_targets[e])) for e in edge_ids}
        strand_solutions = []
        seen = set()

        def backtrack(term_idx, current_solution, available_now):
            if break_early is not None and len(strand_solutions) >= break_early:
                return

            if term_idx == len(path_terms):
                frozen = tuple(current_solution)
                if quotient_by_relabeling:
                    frozen = self._canonicalize_strand_solution(frozen)
                if frozen not in seen:
                    seen.add(frozen)
                    strand_solutions.append(frozen)
                return

            path, multiplicity = path_terms[term_idx]
            candidate_ids = [available_now[e][:] for e in path]
            single_assignments = list(product(*candidate_ids))

            if multiplicity == 1:
                assignment_groups = [(a,) for a in single_assignments]
            else:
                # Use combinations so identical path copies are not over-counted by order.
                assignment_groups = []
                for group in combinations(single_assignments, multiplicity):
                    valid = True
                    for edge_pos in range(len(path)):
                        used = {group[m][edge_pos] for m in range(multiplicity)}
                        if len(used) < multiplicity:
                            valid = False
                            break
                    if valid:
                        assignment_groups.append(group)

            for group in assignment_groups:
                available_next = {e: available_now[e][:] for e in edge_ids}
                updated_solution = list(current_solution)
                valid_group = True

                for assign in group:
                    strand_path = []
                    for edge_pos, e in enumerate(path):
                        sid = assign[edge_pos]
                        if sid not in available_next[e]:
                            valid_group = False
                            break
                        available_next[e].remove(sid)
                        strand_path.append((e, sid))
                    if not valid_group:
                        break
                    updated_solution.append(tuple(strand_path))

                if not valid_group:
                    continue

                backtrack(term_idx + 1, updated_solution, available_next)

        backtrack(0, [], initial_available)
        self.strand_solutions[solution_index] = strand_solutions
        return strand_solutions

    def count_unique_strand_solutions(self, solution_index, break_early=None, quotient_by_relabeling=False):
        """Return the number of unique strand-level solutions for one port-level solution."""
        sols = self.generate_strand_solutions(
            solution_index=solution_index,
            break_early=break_early,
            sort_paths=True,
            quotient_by_relabeling=quotient_by_relabeling,
        )
        return len(sols)

    def print_unique_strand_solution_count(self, solution_index, quotient_by_relabeling=False):
        """Print and return the unique strand-level count for a port-level solution."""
        count = self.count_unique_strand_solutions(
            solution_index=solution_index,
            break_early=None,
            quotient_by_relabeling=quotient_by_relabeling,
        )
        print(f"Port-level solution {solution_index}: {count} unique strand-level solutions")
        return count

    def generate_all_strand_solutions(self, break_early_per_solution=None, stop_after_solutions=None):
        """Generate strand-level solutions for all port-level solutions."""
        if not self.solutions:
            raise ValueError("No solutions computed. Run generate_solutions() first.")

        generated = {}
        n = len(self.solutions)
        if stop_after_solutions is not None:
            n = min(n, int(stop_after_solutions))

        for idx in range(n):
            generated[idx] = self.generate_strand_solutions(
                idx,
                break_early=break_early_per_solution,
                sort_paths=True,
            )
        return generated

    def print_strand_solution(self, solution_index, strand_solution_index=0):
        """Print one strand-level solution in readable form."""
        if solution_index not in self.strand_solutions:
            self.generate_strand_solutions(solution_index)

        all_strand = self.strand_solutions[solution_index]
        if not all_strand:
            print(f"No strand-level solutions found for solution {solution_index}.")
            return
        if strand_solution_index < 0 or strand_solution_index >= len(all_strand):
            raise IndexError(
                f"strand_solution_index {strand_solution_index} out of range [0, {len(all_strand)-1}]"
            )

        print("=" * 70)
        print(f"Port-level solution {solution_index}, strand-level solution {strand_solution_index}")
        for i, strand_path in enumerate(all_strand[strand_solution_index]):
            path_str = " -> ".join(f"(e{e}, s{sid})" for e, sid in strand_path)
            print(f"  path_instance {i}: {path_str}")

    def _draw_base_graph_2d(self, ax, show_edge_ids=False, edge_label_font_size=8):
        """Draw graph skeleton and node types for plotting helpers."""
        pos2d = {
            i: (self.graph.points[i][0], self.graph.points[i][1])
            for i in range(len(self.graph.points))
        }

        g = nx.Graph()
        for node_id in range(len(self.graph.points)):
            g.add_node(node_id)
        for _, (n1, n2) in self.edges.items():
            g.add_edge(n1, n2)

        nx.draw_networkx_edges(g, pos=pos2d, ax=ax, width=1.0, edge_color="lightgray", alpha=0.9)

        port_nodes = [i for i, t in enumerate(self.graph.types) if t == "port"]
        counterport_nodes = [i for i, t in enumerate(self.graph.types) if t == "counterport"]
        junction_nodes = [i for i, t in enumerate(self.graph.types) if t == "junction"]

        nx.draw_networkx_nodes(g, pos=pos2d, nodelist=port_nodes, node_color="tab:red", node_size=80, ax=ax)
        nx.draw_networkx_nodes(g, pos=pos2d, nodelist=counterport_nodes, node_color="tab:blue", node_size=80, ax=ax)
        nx.draw_networkx_nodes(g, pos=pos2d, nodelist=junction_nodes, node_color="black", node_size=40, ax=ax)

        if show_edge_ids:
            edge_labels = {(n1, n2): str(e_id) for e_id, (n1, n2) in self.edges.items()}
            nx.draw_networkx_edge_labels(g, pos=pos2d, edge_labels=edge_labels, font_size=edge_label_font_size, ax=ax)

        return pos2d

    def plot_strand_solution(
        self,
        solution_index=0,
        strand_solution_index=0,
        figsize=(8, 8),
        show_edge_ids=False,
        color_by="instance",
    ):
        """Plot one strand-level solution over the lattice geometry."""
        if solution_index not in self.strand_solutions:
            self.generate_strand_solutions(solution_index)

        strand_solutions = self.strand_solutions.get(solution_index, [])
        if not strand_solutions:
            raise ValueError("No strand-level solutions available for this port-level solution.")
        if strand_solution_index < 0 or strand_solution_index >= len(strand_solutions):
            raise IndexError(
                f"strand_solution_index {strand_solution_index} out of range [0, {len(strand_solutions)-1}]"
            )

        strand_solution = strand_solutions[strand_solution_index]
        fig, ax = plt.subplots(figsize=figsize)
        pos2d = self._draw_base_graph_2d(ax=ax, show_edge_ids=show_edge_ids)

        keys = []
        for idx, strand_path in enumerate(strand_solution):
            if color_by == "path":
                key = tuple(e for e, _ in strand_path)
            elif color_by == "strand":
                key = strand_path
            else:
                key = idx
            keys.append(key)

        unique_keys = list(dict.fromkeys(keys))
        cmap = plt.cm.get_cmap("tab20", max(len(unique_keys), 1))
        color_lookup = {k: cmap(i) for i, k in enumerate(unique_keys)}

        for idx, strand_path in enumerate(strand_solution):
            edge_path = tuple(e for e, _ in strand_path)
            curve_pts = self._get_curved_path_points(edge_path, pos2d, num_points=70)
            xs = [pt[0] for pt in curve_pts]
            ys = [pt[1] for pt in curve_pts]

            # Small deterministic jitter separates overlapping strand drawings.
            jitter_seed = sum((i + 1) * (sid + 1) for i, (_, sid) in enumerate(strand_path))
            jitter = 0.004 * ((jitter_seed % 9) - 4)
            color = color_lookup[keys[idx]]
            ax.plot(xs, [y + jitter for y in ys], color=color, linewidth=2.2, alpha=0.95)

        ax.set_title(f"Strand solution {strand_solution_index} for port solution {solution_index}")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_all_strand_solutions(self, solution_index=0, ncols=3, figsize=(15, 5), show_edge_ids=False):
        """Plot all strand-level solutions for one port-level solution."""
        if solution_index not in self.strand_solutions:
            self.generate_strand_solutions(solution_index)

        strand_solutions = self.strand_solutions.get(solution_index, [])
        if not strand_solutions:
            raise ValueError("No strand-level solutions available for this port-level solution.")

        n_solutions = len(strand_solutions)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n_solutions / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]

            if idx >= n_solutions:
                ax.axis("off")
                continue

            pos2d = self._draw_base_graph_2d(ax=ax, show_edge_ids=show_edge_ids, edge_label_font_size=7)
            strand_solution = strand_solutions[idx]
            cmap = plt.cm.get_cmap("tab20", max(len(strand_solution), 1))

            for pidx, strand_path in enumerate(strand_solution):
                edge_path = tuple(e for e, _ in strand_path)
                curve_pts = self._get_curved_path_points(edge_path, pos2d, num_points=60)
                xs = [pt[0] for pt in curve_pts]
                ys = [pt[1] for pt in curve_pts]

                jitter_seed = sum((i + 1) * (sid + 1) for i, (_, sid) in enumerate(strand_path))
                jitter = 0.004 * ((jitter_seed % 9) - 4)
                ax.plot(xs, [y + jitter for y in ys], color=cmap(pidx), linewidth=2.0, alpha=0.95)

            ax.set_title(f"Strand {idx}")
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def debug_trace_enumeration(self):
        """
        Trace all sequences through the three-stage pipeline:
        Stage 1: Enumeration (all_unique_sequences)
        Stage 2: Multiplicity constraint (generate_valid_multiplicities_v2)
        Stage 3: Symmetry dedup (find_unique_sets)
        """
        print("\n" + "="*90)
        print("DEBUG: Full three-stage enumeration trace")
        print("="*90)
        
        # Stage 1: Generate all sequences
        all_seqs = all_unique_sequences(
            self.edges.keys(),
            min_length=self.power_set_min_length,
            max_length=self.power_set_max_length,
        )
        print(f"\n{'STAGE 1 - ENUMERATION':^90}")
        print(f"Total unique edge sequences generated: {len(all_seqs)}")
        for i, seq in enumerate(all_seqs):
            print(f"  {i:3d}: {seq}")
        
        # Stage 1b: Path validity filter
        valid_paths = [s for s in all_seqs if self.is_valid_path(s)]
        valid_paths = self.prune_redundant_loops(valid_paths)
        print(f"\n{'STAGE 1b - PATH VALIDITY FILTER':^90}")
        print(f"Sequences that pass port/junction/connectivity checks: {len(valid_paths)}")
        for i, seq in enumerate(valid_paths):
            print(f"  {i:3d}: {seq}")
        
        # Stage 2: Generate all valid multiplicity tuples
        print(f"\n{'STAGE 2 - MULTIPLICITY CONSTRAINT':^90}")
        print(f"Target edge counts: {self.edge_targets}")
        print(f"Max multiplicity per path: {self.max_multiplicity_path}")
        
        multiplicity_iter = generate_valid_multiplicities_v2(
            sequences=valid_paths,
            edge_targets=self.edge_targets,
            edge_ids=list(self.edges.keys()),
            max_multiplicity_path=self.max_multiplicity_path,
        )
        valid_multiplicities = list(multiplicity_iter)
        print(f"\nAll multiplicity tuples from backtracking solver: {len(valid_multiplicities)}")
        
        collections_stage2 = []
        for i, mults in enumerate(valid_multiplicities):
            collection = Counter({seq: m for seq, m in zip(valid_paths, mults) if m > 0})
            edge_counts = Counter({e: 0 for e in self.edges.keys()})
            for path, mult in collection.items():
                for e in path:
                    edge_counts[e] += mult
            
            matches_target = all(edge_counts[e] == self.edge_targets[e] for e in self.edges.keys())
            
            print(f"\n  Tuple {i}: {mults}")
            print(f"    → Collection: {dict(collection)}")
            print(f"    → Edge counts: {dict(sorted(edge_counts.items()))}")
            print(f"    → Matches target exactly: {matches_target}")
            
            if matches_target:
                collections_stage2.append(collection)
        
        print(f"\n  ✓ {len(collections_stage2)} multiplicity tuples satisfy edge target constraint")
        
        # Stage 2b: Build frozensets for dedup
        valid_collections_frozen = [frozenset(coll.items()) for coll in collections_stage2]
        print(f"\n{'STAGE 2b - CONVERT TO FROZENSETS':^90}")
        print(f"Collections as hashable frozensets:")
        for i, frozen in enumerate(valid_collections_frozen):
            print(f"  {i:3d}: {frozen}")
        
        # Stage 3: Symmetry dedup
        print(f"\n{'STAGE 3 - SYMMETRY DEDUPLICATION':^90}")
        print(f"Lattice symmetry operations: {len(self.symmetries)}")
        for sym_idx, sym in enumerate(self.symmetries):
            print(f"  {sym_idx}: {sym}")
        
        unique = self.find_unique_sets(valid_collections_frozen)
        print(f"\nAfter deduplication: {len(unique)} canonical solutions")
        for i, sol in enumerate(sorted(unique, key=str)):
            print(f"  {i:3d}: {sol}")
        
        # Trace which collections map to which canonical solution
        print(f"\n{'DEDUPLICATION MAPPING':^90}")
        for coll_idx, coll in enumerate(valid_collections_frozen):
            is_canonical = coll in unique
            status = "✓ KEPT (canonical)" if is_canonical else "✗ MERGED (equivalent)"
            print(f"\n  Collection {coll_idx}: {status}")
            print(f"    {coll}")
            
            if not is_canonical:
                # Find which canonical it maps to
                found_equivalent = False
                for eq_perm in self.symmetries:
                    transformed = self.apply_permutation(coll, eq_perm)
                    if transformed is not None and transformed in unique:
                        print(f"    → Maps to canonical via permutation {eq_perm}:")
                        print(f"       {transformed}")
                        found_equivalent = True
                        break
                if not found_equivalent:
                    print(f"    → (mapping not found in canonical set)")
        
        print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    # Example usage: heterogeneous targets for first four edges, rest default to zero.
    topo = TopologyMultiEdge(
        filename="lattice_square.dat",
        edge_targets={0: 3, 1: 3, 2:1, 3: 1},
        default_edge_target=0,
        conjugacy=False,
        max_multiplicity_path=3,
        power_set_min_length=2,
        power_set_max_length=2,
    )

    topo.generate_solutions(stop_early=False)
    
    # Run full debug trace to understand filtering
    # topo.debug_trace_enumeration()
    
    print(f"Found {len(topo.solutions)} unique port-level solutions\n")
    topo.print_all_solutions()

    if topo.solutions:
        # topo.plot_port_solution(solution_index=0)
        topo.plot_all_port_solutions(ncols=3, figsize=(15, 10), show_edge_ids=False)

    topo.generate_strand_solutions(solution_index=2)
    # topo.print_strand_solution(solution_index=0, strand_solution_index=1)
    topo.plot_all_strand_solutions(solution_index=2)