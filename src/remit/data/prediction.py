"""Prediction datasets that preserve explicit split row order."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data

from remit.chemistry.features import smiles_to_graph


def label_matrix(frame: pd.DataFrame, endpoints: Sequence[str]) -> np.ndarray:
    return frame[list(endpoints)].astype("float32").to_numpy(na_value=np.nan)


class MoleculeGraphDataset(Dataset[Data]):
    """Eager in-memory graph dataset for a single immutable partition."""

    def __init__(self, frame: pd.DataFrame, endpoints: Sequence[str]) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.endpoints = tuple(endpoints)
        labels = label_matrix(self.frame, self.endpoints)
        self.graphs = [
            smiles_to_graph(smiles, labels[index])
            for index, smiles in enumerate(self.frame["canonical_smiles"].astype(str))
        ]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]
