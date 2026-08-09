"""Deterministic 2D molecular graph and ECFP4 features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

ATOM_FEATURE_CARDINALITIES = (119, 6, 12, 12, 10, 6, 9, 2, 2)
BOND_FEATURE_CARDINALITIES = (5, 7, 2, 2)

_CHIRALITY = (
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
    "CHI_TETRAHEDRAL",
)
_HYBRIDIZATION = ("S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "OTHER")
_BOND_TYPE = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
_BOND_STEREO = (
    "STEREONONE",
    "STEREOANY",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
)


def _index_or_unknown(value: object, choices: Sequence[object]) -> int:
    try:
        return choices.index(value)
    except ValueError:
        return len(choices)


def atom_features(atom: Chem.Atom) -> list[int]:
    atomic_number = atom.GetAtomicNum()
    atomic_number_index = atomic_number - 1 if 1 <= atomic_number <= 118 else 118
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    total_hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
    radicals = atom.GetNumRadicalElectrons()
    return [
        atomic_number_index,
        _index_or_unknown(str(atom.GetChiralTag()), _CHIRALITY),
        degree if 0 <= degree <= 10 else 11,
        formal_charge + 5 if -5 <= formal_charge <= 5 else 11,
        total_hydrogens if 0 <= total_hydrogens <= 8 else 9,
        radicals if 0 <= radicals <= 4 else 5,
        _index_or_unknown(str(atom.GetHybridization()), _HYBRIDIZATION),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


def bond_features(bond: Chem.Bond) -> list[int]:
    return [
        _index_or_unknown(str(bond.GetBondType()), _BOND_TYPE),
        _index_or_unknown(str(bond.GetStereo()), _BOND_STEREO),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str, labels: Sequence[float] | None = None) -> Data:
    """Convert a canonical SMILES into a bidirectional PyG graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse canonical SMILES: {smiles}")
    node_features = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    edges: list[tuple[int, int]] = []
    edge_features: list[list[int]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        edges.extend([(begin, end), (end, begin)])
        edge_features.extend([features, features])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_FEATURE_CARDINALITIES)), dtype=torch.long)
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        graph.y = torch.from_numpy(np.asarray(labels, dtype=np.float32).copy()).unsqueeze(0)
    return graph


def ecfp_matrix(
    smiles: Sequence[str], radius: int = 2, size: int = 2048, include_chirality: bool = True
) -> np.ndarray:
    """Build a dense binary Morgan fingerprint matrix with stable row order."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=size,
        includeChirality=include_chirality,
    )
    matrix = np.empty((len(smiles), size), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"RDKit cannot parse canonical SMILES: {value}")
        matrix[index] = generator.GetFingerprintAsNumPy(mol)
    return matrix
