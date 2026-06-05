"""
TriGraphX - Multi-dimensional Relational Metric Space

A unified database model replacing tree index + graph DB + vector DB
with a single metric space abstraction.
"""

__version__ = "0.1.0"

from .entity import (
    Entity, DistanceMetric, MetricType,
    SemanticEmbedding, HierarchyEmbedding, AssociationEmbedding, CausalEmbedding,
    HierarchyDistance, SemanticDistance, AssociationDistance, CausalDistance
)
from .space import MetricSpace, QueryResult
from .persistence import PersistenceLayer

__all__ = [
    "Entity",
    "DistanceMetric",
    "MetricType",
    "MetricSpace",
    "QueryResult",
    "PersistenceLayer",
    "SemanticEmbedding",
    "HierarchyEmbedding",
    "AssociationEmbedding",
    "CausalEmbedding",
    "HierarchyDistance",
    "SemanticDistance",
    "AssociationDistance",
    "CausalDistance",
]
