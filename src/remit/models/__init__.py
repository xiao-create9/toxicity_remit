"""Prediction model registry."""

from remit.models.gnn import AttentiveFPPredictor, GINEPredictor, build_gnn_model

__all__ = ["AttentiveFPPredictor", "GINEPredictor", "build_gnn_model"]
