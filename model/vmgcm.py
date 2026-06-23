import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist


class VoronoiGraphBuilder:
    POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS', 'HIS', 'ASP', 'GLU', 'LYS', 'ARG'}
    NONPOLAR_RESIDUES = {'LEU', 'ILE', 'VAL', 'PHE', 'TRP', 'MET', 'ALA', 'GLY', 'PRO'}
    POSITIVE_RESIDUES = {'LYS', 'ARG', 'HIS'}
    NEGATIVE_RESIDUES = {'ASP', 'GLU'}

    def __init__(self, hydrogen_bond_threshold=6.0, hydrophobic_threshold=8.0, salt_bridge_threshold=12.0):
        self.hydrogen_bond_threshold = hydrogen_bond_threshold
        self.hydrophobic_threshold = hydrophobic_threshold
        self.salt_bridge_threshold = salt_bridge_threshold

    def build_voronoi_topology(self, residue_coords, cutoff=12.0):
        n_residues = len(residue_coords)
        voronoi_area_matrix = np.zeros((n_residues, n_residues))
        coords_np = residue_coords.cpu().numpy() if hasattr(residue_coords, 'cpu') else residue_coords
        distance_matrix = cdist(coords_np, coords_np)

        adjacency = (distance_matrix < cutoff) & (distance_matrix > 0)

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                if adjacency[i, j]:
                    voronoi_area_matrix[i, j] = self._estimate_contact_area(
                        distance_matrix[i, j]
                    )
                    voronoi_area_matrix[j, i] = voronoi_area_matrix[i, j]

        return torch.tensor(voronoi_area_matrix, dtype=torch.float32)

    def _estimate_contact_area(self, distance):
        sigma = 4.0
        if distance < 8.0:
            return np.exp(-((distance - 3.5) ** 2) / (2 * sigma ** 2))
        return 0.0

    def build_hydrogen_bond_edges(self, residue_coords, residue_types):
        n_residues = len(residue_coords)
        edge_matrix = np.zeros((n_residues, n_residues))
        coords_np = residue_coords.cpu().numpy() if hasattr(residue_coords, 'cpu') else residue_coords
        distance_matrix = cdist(coords_np, coords_np)

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = distance_matrix[i, j]
                if dist < self.hydrogen_bond_threshold:
                    if residue_types[i] in self.POLAR_RESIDUES and residue_types[j] in self.POLAR_RESIDUES:
                        edge_matrix[i, j] = 1.0
                        edge_matrix[j, i] = 1.0

        return torch.tensor(edge_matrix, dtype=torch.float32)

    def build_hydrophobic_edges(self, residue_coords, residue_types):
        n_residues = len(residue_coords)
        edge_matrix = np.zeros((n_residues, n_residues))
        coords_np = residue_coords.cpu().numpy() if hasattr(residue_coords, 'cpu') else residue_coords
        distance_matrix = cdist(coords_np, coords_np)

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = distance_matrix[i, j]
                if dist < self.hydrophobic_threshold:
                    if residue_types[i] in self.NONPOLAR_RESIDUES and residue_types[j] in self.NONPOLAR_RESIDUES:
                        edge_matrix[i, j] = 1.0
                        edge_matrix[j, i] = 1.0

        return torch.tensor(edge_matrix, dtype=torch.float32)

    def build_salt_bridge_edges(self, residue_coords, residue_types):
        n_residues = len(residue_coords)
        edge_matrix = np.zeros((n_residues, n_residues))
        coords_np = residue_coords.cpu().numpy() if hasattr(residue_coords, 'cpu') else residue_coords
        distance_matrix = cdist(coords_np, coords_np)

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = distance_matrix[i, j]
                if dist < self.salt_bridge_threshold:
                    pos_neg = (
                        (residue_types[i] in self.POSITIVE_RESIDUES and residue_types[j] in self.NEGATIVE_RESIDUES) or
                        (residue_types[i] in self.NEGATIVE_RESIDUES and residue_types[j] in self.POSITIVE_RESIDUES)
                    )
                    if pos_neg:
                        edge_matrix[i, j] = 1.0
                        edge_matrix[j, i] = 1.0

        return torch.tensor(edge_matrix, dtype=torch.float32)


class VMGCM(nn.Module):
    EDGE_TYPES = ['voronoi', 'hydrogen_bond', 'hydrophobic', 'salt_bridge']

    def __init__(self, config=None):
        super().__init__()

        graph_config = config.get('graph', {}) if config else {}
        self.hydrogen_bond_threshold = graph_config.get('hydrogen_bond_threshold', 6.0)
        self.hydrophobic_threshold = graph_config.get('hydrophobic_threshold', 8.0)
        self.salt_bridge_threshold = graph_config.get('salt_bridge_threshold', 12.0)

        self.graph_builder = VoronoiGraphBuilder(
            hydrogen_bond_threshold=self.hydrogen_bond_threshold,
            hydrophobic_threshold=self.hydrophobic_threshold,
            salt_bridge_threshold=self.salt_bridge_threshold
        )

        self.edge_type_embeddings = nn.ModuleDict({
            edge_type: nn.Linear(1, 64) for edge_type in self.EDGE_TYPES
        })

    def build_graph(self, residue_coords, residue_types=None, sequence=None):
        if residue_types is None and sequence is not None:
            residue_types = self._sequence_to_residue_types(sequence)

        voronoi_matrix = self.graph_builder.build_voronoi_topology(residue_coords)

        hydrogen_bond_matrix = self.graph_builder.build_hydrogen_bond_edges(
            residue_coords, residue_types
        )

        hydrophobic_matrix = self.graph_builder.build_hydrophobic_edges(
            residue_coords, residue_types
        )

        salt_bridge_matrix = self.graph_builder.build_salt_bridge_edges(
            residue_coords, residue_types
        )

        edge_matrices = {
            'voronoi': voronoi_matrix,
            'hydrogen_bond': hydrogen_bond_matrix,
            'hydrophobic': hydrophobic_matrix,
            'salt_bridge': salt_bridge_matrix
        }

        normalized_matrices = {}
        for edge_type, matrix in edge_matrices.items():
            normalized_matrices[edge_type] = self._normalize_adjacency(matrix)

        return normalized_matrices

    def _normalize_adjacency(self, adj_matrix):
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        normalized = adj_matrix / degree
        return normalized

    def _sequence_to_residue_types(self, sequence):
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return [three_to_one.get(aa.upper(), 'X') for aa in sequence]

    def forward(self, residue_coords, residue_types=None, sequence=None):
        if residue_coords.dim() == 3:
            residue_coords = residue_coords[0]

        if residue_types is not None and isinstance(residue_types[0], list):
            residue_types = residue_types[0]

        edge_matrices = self.build_graph(residue_coords, residue_types, sequence)

        device = next(self.parameters()).device
        raw_edge_matrices = {}
        processed_edges = {}
        for edge_type, adj in edge_matrices.items():
            adj = adj.to(device).unsqueeze(0)
            raw_edge_matrices[edge_type] = adj
            adj_unsqueezed = adj.unsqueeze(-1)
            processed_edges[edge_type] = self.edge_type_embeddings[edge_type](adj_unsqueezed)

        return raw_edge_matrices, processed_edges
