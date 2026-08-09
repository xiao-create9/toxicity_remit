"""Edge-aware GINE and AttentiveFP multi-task predictors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

from remit.chemistry.features import ATOM_FEATURE_CARDINALITIES, BOND_FEATURE_CARDINALITIES


class CategoricalFeatureEncoder(nn.Module):
    """Sum independent embeddings without treating category IDs as continuous values."""

    def __init__(self, cardinalities: Sequence[int], hidden_dim: int) -> None:
        super().__init__()
        self.cardinalities = tuple(cardinalities)
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, hidden_dim) for cardinality in self.cardinalities
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.embeddings):
            raise ValueError(
                f"Expected categorical features [N, {len(self.embeddings)}], "
                f"received {tuple(features.shape)}"
            )
        encoded = self.embeddings[0](features[:, 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            encoded = encoded + embedding(features[:, column])
        return encoded


class GINEPredictor(nn.Module):
    """Residual molecular GINE without a virtual-node explanation shortcut."""

    def __init__(
        self,
        num_tasks: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.2,
        train_eps: bool = True,
        pooling: str = "add_mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GINE requires at least one message-passing layer")
        if pooling not in {"add", "mean", "add_mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.atom_encoder = CategoricalFeatureEncoder(ATOM_FEATURE_CARDINALITIES, hidden_dim)
        self.bond_encoder = CategoricalFeatureEncoder(BOND_FEATURE_CARDINALITIES, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=train_eps, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        graph_dim = hidden_dim * 2 if pooling == "add_mean" else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        node_state = self.atom_encoder(x)
        edge_state = self.bond_encoder(edge_attr)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            update = conv(node_state, edge_index, edge_state)
            node_state = norm(node_state + self.dropout(update))
        if self.pooling == "add":
            graph_state = global_add_pool(node_state, batch)
        elif self.pooling == "mean":
            graph_state = global_mean_pool(node_state, batch)
        else:
            graph_state = torch.cat(
                [global_add_pool(node_state, batch), global_mean_pool(node_state, batch)], dim=-1
            )
        return self.head(graph_state)


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP preceded by the same categorical atom/bond encoders as GINE."""

    def __init__(
        self,
        num_tasks: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.atom_encoder = CategoricalFeatureEncoder(ATOM_FEATURE_CARDINALITIES, hidden_dim)
        self.bond_encoder = CategoricalFeatureEncoder(BOND_FEATURE_CARDINALITIES, hidden_dim)
        self.model = AttentiveFP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=num_tasks,
            edge_dim=hidden_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(self.atom_encoder(x), edge_index, self.bond_encoder(edge_attr), batch)


def build_gnn_model(model_config: dict[str, Any], num_tasks: int) -> nn.Module:
    name = model_config.get("name")
    common = {
        "num_tasks": num_tasks,
        "hidden_dim": int(model_config.get("hidden_dim", 256)),
        "num_layers": int(model_config.get("num_layers", 3)),
        "dropout": float(model_config.get("dropout", 0.2)),
    }
    if name == "gine":
        return GINEPredictor(
            **common,
            train_eps=bool(model_config.get("train_eps", True)),
            pooling=str(model_config.get("pooling", "add_mean")),
        )
    if name == "attentivefp":
        return AttentiveFPPredictor(
            **common,
            num_timesteps=int(model_config.get("num_timesteps", 2)),
        )
    raise ValueError(f"Unsupported GNN model: {name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
