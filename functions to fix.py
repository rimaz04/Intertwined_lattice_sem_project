
# # line 864 
# edge_to_id = {tuple(sorted(edge)): edge_id for edge_id, edge in edges.items()}
#         edge_mapping = {}
#         for edge_id, (a, b) in edges.items():
#             mapped_edge = tuple(sorted((node_mapping[a], node_mapping[b])))
#             if mapped_edge not in edge_to_id:
#                 raise KeyError(mapped_edge)
#             edge_mapping[edge_id] = edge_to_id[mapped_edge]

# # line 876
#         try:
#             mapping = self.generate_new_edge_mapping(mapping_nodes, edges)
#         except KeyError:
#             return None

# # remove mapping line

# # line 891
#             equivalent_representations = {
#                 transformed
#                 for eq in equivalences
#                 for transformed in [self.apply_permutation(path_set, eq, edges)]
#                 if transformed is not None
#             }
#             if not equivalent_representations:
#                 equivalent_representations = {frozenset(path_set)}

# # change equivalent rep


# # line 926, same replace equail_reps

#                 equivalent_reps = {
#                     transformed
#                     for eq in self.symmetries
#                     for transformed in [self.apply_permutation(valid_col, eq, self.edges)]
#                     if transformed is not None
#                 }

