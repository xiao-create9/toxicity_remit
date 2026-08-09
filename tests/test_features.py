from __future__ import annotations

import numpy as np

from remit.chemistry.features import ecfp_matrix, smiles_to_graph


def test_graph_features_are_bidirectional_and_handle_bondless_molecules() -> None:
    ethanol = smiles_to_graph("CCO", [0.0, 1.0])
    sodium = smiles_to_graph("[Na+]", [1.0, np.nan])
    assert ethanol.x.shape == (3, 9)
    assert ethanol.edge_index.shape == (2, 4)
    assert ethanol.edge_attr.shape == (4, 4)
    assert ethanol.y.shape == (1, 2)
    assert sodium.edge_index.shape == (2, 0)
    assert sodium.edge_attr.shape == (0, 4)


def test_ecfp_is_binary_deterministic_and_order_preserving() -> None:
    first = ecfp_matrix(["CCO", "c1ccccc1"], radius=2, size=128)
    second = ecfp_matrix(["CCO", "c1ccccc1"], radius=2, size=128)
    assert first.shape == (2, 128)
    assert set(np.unique(first)).issubset({0, 1})
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
