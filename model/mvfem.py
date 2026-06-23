import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticView(nn.Module):
    def __init__(self, esm2_model_name="esm2_t33_650M_UR50D", embedding_dim=1280):
        super().__init__()
        self.esm2_model_name = esm2_model_name
        self.embedding_dim = embedding_dim
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_model_name)
            self.model.eval()
        except (ImportError, FileNotFoundError):
            self.model = None
            print("Warning: ESM-2 not available. Semantic features will be zeros.")

    def extract(self, sequence):
        if self.model is None:
            return torch.zeros(len(sequence), self.embedding_dim)

        with torch.no_grad():
            batch_tokens = self.alphabet.encode(sequence)
            results = self.model(batch_tokens)
            embeddings = results["representations"][-1]
            return embeddings

    def forward(self, sequence):
        return self.extract(sequence)


class EvolutionaryView(nn.Module):
    def __init__(self, pssm_iterations=3, pssm_evalue=0.001):
        super().__init__()
        self.pssm_iterations = pssm_iterations
        self.pssm_evalue = pssm_evalue
        self.pssm_matrix = None

    def generate_pssm(self, sequence, db_path="swissprot"):
        try:
            from Bio.Blast.Applications import PsiblastCommandline
            import subprocess
            cmd = PsiblastCommandline(
                query=sequence,
                db=db_path,
                num_iterations=self.pssm_iterations,
                evalue=self.pssm_evalue,
                out_ascii_pssm="pssm.txt"
            )
            subprocess.run(str(cmd), shell=True)
            return self._parse_pssm("pssm.txt")
        except Exception:
            return torch.zeros(len(sequence), 20)

    def _parse_pssm(self, pssm_file):
        matrix = []
        with open(pssm_file, 'r') as f:
            for line in f:
                if line.strip().isdigit():
                    continue
                parts = line.split()
                if len(parts) >= 44:
                    row = [float(x) for x in parts[42:62]]
                    matrix.append(row)
        return torch.tensor(matrix, dtype=torch.float32)

    def forward(self, sequence):
        if self.pssm_matrix is None:
            self.pssm_matrix = self.generate_pssm(sequence)
        return self.pssm_matrix


class StructuralView(nn.Module):
    def __init__(self):
        super().__init__()
        self.secondary_structure_dim = 8
        self.torsion_angle_dim = 4
        self.rsa_dim = 1
        self.total_dim = self.secondary_structure_dim + self.torsion_angle_dim + self.rsa_dim

    def extract_dSSP(self, pdb_path):
        try:
            from Bio.PDB import DSSP
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)
            model = structure[0]
            dssp = DSSP(model, pdb_path)
            features = []
            for key in dssp.keys():
                ss_code = self._ss_to_onehot(dssp[key][2])
                phi_psi = self._torsion_to_features(dssp[key][3], dssp[key][4])
                rsa = torch.tensor([dssp[key][3]], dtype=torch.float32)
                features.append(torch.cat([ss_code, phi_psi, rsa]))
            return torch.stack(features) if features else torch.zeros(1, self.total_dim)
        except Exception:
            return torch.zeros(1, self.total_dim)

    def _ss_to_onehot(self, ss):
        ss_map = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
        onehot = torch.zeros(self.secondary_structure_dim)
        if ss in ss_map:
            onehot[ss_map[ss]] = 1.0
        return onehot

    def _torsion_to_features(self, phi, psi):
        import math
        phi_rad = math.radians(phi) if phi else 0
        psi_rad = math.radians(psi) if psi else 0
        return torch.tensor([
            math.sin(phi_rad * math.pi / 180),
            math.cos(phi_rad * math.pi / 180),
            math.sin(psi_rad * math.pi / 180),
            math.cos(psi_rad * math.pi / 180)
        ], dtype=torch.float32)

    def forward(self, pdb_path):
        return self.extract_dSSP(pdb_path)


class AtomicView(nn.Module):
    ATOM_FEATURES_DIM = 7

    def __init__(self):
        super().__init__()
        self.atom_feature_names = [
            'mass', 'b_factor', 'is_sidechain', 'charge',
            'num_hydrogens', 'is_ring', 'van_der_waals_radius'
        ]

    def extract_atomic_features(self, pdb_path):
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)
            residue_features = []

            for chain in structure[0]:
                for residue in chain:
                    if residue.get_id()[0] == ' ':
                        atom_features = []
                        for atom in residue:
                            if atom.element != 'H':
                                feature = self._extract_atom_feature(atom)
                                atom_features.append(feature)
                        if atom_features:
                            residue_features.append(torch.mean(torch.stack(atom_features), dim=0))
                        else:
                            residue_features.append(torch.zeros(self.ATOM_FEATURES_DIM))
            return torch.stack(residue_features) if residue_features else torch.zeros(1, self.ATOM_FEATURES_DIM)
        except Exception:
            return torch.zeros(1, self.ATOM_FEATURES_DIM)

    def _extract_atom_feature(self, atom):
        element = atom.element
        features = [
            self._get_atom_mass(element),
            atom.bfactor / 100.0 if atom.bfactor else 0.0,
            1.0 if self._is_sidechain_atom(atom) else 0.0,
            self._get_partial_charge(atom),
            self._count_hydrogens(atom, element),
            1.0 if self._is_in_ring(element) else 0.0,
            self._get_vdw_radius(element)
        ]
        return torch.tensor(features, dtype=torch.float32)

    def _get_atom_mass(self, element):
        mass_map = {'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'P': 30.97}
        return mass_map.get(element, 1.0)

    def _is_sidechain_atom(self, atom):
        residue_name = atom.parent.resname
        backbone_atoms = {'N', 'CA', 'C', 'O', 'H', 'HA'}
        return atom.name not in backbone_atoms

    def _get_partial_charge(self, atom):
        electronegativity = {'N': 3.04, 'O': 3.44, 'S': 2.58, 'C': 2.55, 'P': 2.19}
        return (electronegativity.get(atom.element, 2.5) - 2.5) * 0.2

    def _count_hydrogens(self, atom, element):
        return 0

    def _is_in_ring(self, element):
        aromatic_elements = {'C', 'N'}
        return 1.0 if element in aromatic_elements else 0.0

    def _get_vdw_radius(self, element):
        vdw_map = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80}
        return vdw_map.get(element, 1.50)

    def forward(self, pdb_path):
        return self.extract_atomic_features(pdb_path)


class MVFEM(nn.Module):
    SEMANTIC_DIM = 1280
    EVOLUTIONARY_DIM = 20
    STRUCTURAL_DIM = 13
    ATOMIC_DIM = 7
    TOTAL_DIM = SEMANTIC_DIM + EVOLUTIONARY_DIM + STRUCTURAL_DIM + ATOMIC_DIM

    def __init__(self, config=None):
        super().__init__()
        self.semantic_view = SemanticView()
        self.evolutionary_view = EvolutionaryView()
        self.structural_view = StructuralView()
        self.atomic_view = AtomicView()

        hidden_dim = config.get('model', {}).get('hidden_dim', 256) if config else 256
        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(self.TOTAL_DIM),
            nn.Linear(self.TOTAL_DIM, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, sequence=None, pdb_path=None, return_all_features=False):
        features = []
        device = next(self.parameters()).device

        seq_len = len(sequence) if sequence else 1

        if sequence is not None:
            semantic_feat = self.semantic_view(sequence).to(device)
            evolutionary_feat = self.evolutionary_view(sequence).to(device)
        else:
            semantic_feat = torch.zeros(seq_len, self.SEMANTIC_DIM, device=device)
            evolutionary_feat = torch.zeros(seq_len, self.EVOLUTIONARY_DIM, device=device)
        features.extend([semantic_feat, evolutionary_feat])

        if pdb_path is not None:
            structural_feat = self.structural_view(pdb_path).to(device)
            atomic_feat = self.atomic_view(pdb_path).to(device)
        else:
            structural_feat = torch.zeros(seq_len, self.STRUCTURAL_DIM, device=device)
            atomic_feat = torch.zeros(seq_len, self.ATOMIC_DIM, device=device)
        features.extend([structural_feat, atomic_feat])

        concatenated = torch.cat(features, dim=-1)
        embedded = self.embedding_layer(concatenated)

        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(0)

        if return_all_features:
            return concatenated, embedded
        return embedded
