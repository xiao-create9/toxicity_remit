from __future__ import annotations

from torch_geometric.data import Batch

from remit.chemistry.features import smiles_to_graph
from remit.models.gnn import AttentiveFPPredictor, GINEPredictor


def _batch() -> Batch:
    return Batch.from_data_list(
        [smiles_to_graph("CCO", [0.0, 1.0]), smiles_to_graph("[Na+]", [1.0, 0.0])]
    )


def test_gine_forward_shape() -> None:
    batch = _batch()
    model = GINEPredictor(num_tasks=2, hidden_dim=32, num_layers=2, dropout=0.0)
    assert model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).shape == (2, 2)


def test_attentivefp_forward_shape() -> None:
    batch = _batch()
    model = AttentiveFPPredictor(
        num_tasks=2,
        hidden_dim=32,
        num_layers=2,
        num_timesteps=2,
        dropout=0.0,
    )
    assert model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).shape == (2, 2)
