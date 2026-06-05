"""
Entity and Distance Metric definitions for TriGraphX.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import hashlib
from datetime import datetime


class MetricType(Enum):
    """Support distance metric types."""
    HIERARCHY = "hierarchy"           # Tree distances
    SEMANTIC = "semantic"              # Vector similarity
    ASSOCIATION = "association"        # Graph relationships
    CAUSAL = "causal"                 # Cause-effect


@dataclass
class Entity:
    """Base entity in the metric space."""
    
    id: str
    data: Dict[str, Any]
    embeddings: Dict[MetricType, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted: bool = False
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "data": self.data,
            "embeddings": {k.value: v for k, v in self.embeddings.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted": self.deleted,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Entity":
        """Deserialize from dictionary."""
        embeddings = {}
        if d.get("embeddings"):
            for k, v in d["embeddings"].items():
                try:
                    embeddings[MetricType(k)] = v
                except ValueError:
                    pass  # Skip unknown metric types
        
        return Entity(
            id=d["id"],
            data=d.get("data", {}),
            embeddings=embeddings,
            metadata=d.get("metadata", {}),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else None,
            updated_at=datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else None,
            deleted=d.get("deleted", False),
        )
    
    def get_hash(self) -> str:
        """Get content hash for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class HierarchyEmbedding:
    """Tree hierarchy representation."""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    level: int = 0
    path: str = ""  # e.g., "root/branch/leaf"
    
    def to_dict(self):
        return {
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "level": self.level,
            "path": self.path,
        }
    
    @staticmethod
    def from_dict(d):
        return HierarchyEmbedding(**d)


@dataclass
class SemanticEmbedding:
    """Vector embedding representation."""
    vector: List[float]  # Original embeddings
    dimension: int = 0
    quantized: Optional[List[int]] = None  # int8 quantized (for Rust)
    
    def __post_init__(self):
        if self.dimension == 0:
            self.dimension = len(self.vector)
    
    def to_dict(self):
        return {
            "vector": self.vector,
            "dimension": self.dimension,
            "quantized": self.quantized,
        }
    
    @staticmethod
    def from_dict(d):
        return SemanticEmbedding(**d)


@dataclass
class AssociationEmbedding:
    """Graph relationship representation."""
    edges: Dict[str, float] = field(default_factory=dict)  # {entity_id: weight}
    bidirectional: bool = True
    relationship_type: str = "related"
    
    def to_dict(self):
        return {
            "edges": self.edges,
            "bidirectional": self.bidirectional,
            "relationship_type": self.relationship_type,
        }
    
    @staticmethod
    def from_dict(d):
        return AssociationEmbedding(**d)


@dataclass
class CausalEmbedding:
    """Cause-effect relationship representation."""
    causes: List[Tuple[str, float]] = field(default_factory=list)  # (entity_id, strength)
    effects: List[Tuple[str, float]] = field(default_factory=list)
    temporal_order: Optional[List[str]] = None  # temporal sequence of entities
    
    def to_dict(self):
        return {
            "causes": self.causes,
            "effects": self.effects,
            "temporal_order": self.temporal_order,
        }
    
    @staticmethod
    def from_dict(d):
        return CausalEmbedding(**d)


class DistanceMetric:
    """Abstract base for distance computation."""
    
    def __init__(self, metric_type: MetricType):
        self.metric_type = metric_type
    
    def compute(self, embedding1: Any, embedding2: Any) -> float:
        """Compute distance between two embeddings."""
        raise NotImplementedError
    
    def compute_batch(self, embedding1: Any, embeddings2: List[Any]) -> List[float]:
        """Compute distance to multiple embeddings."""
        raise NotImplementedError


class HierarchyDistance(DistanceMetric):
    """Tree distance using lowest common ancestor."""
    
    def __init__(self):
        super().__init__(MetricType.HIERARCHY)
    
    def compute(self, emb1: HierarchyEmbedding, emb2: HierarchyEmbedding) -> float:
        """
        Compute hierarchy distance.
        - Same node: 0
        - Direct parent-child: 1
        - Common ancestor at level n: 2*(depth - n)
        """
        path1 = emb1.path.split("/")
        path2 = emb2.path.split("/")
        
        # Find lowest common ancestor
        lca_depth = 0
        for i, (p1, p2) in enumerate(zip(path1, path2)):
            if p1 == p2:
                lca_depth = i
            else:
                break
        
        # Distance = sum of depths from LCA
        depth1 = len(path1) - 1 - lca_depth
        depth2 = len(path2) - 1 - lca_depth
        
        return float(depth1 + depth2)
    
    def compute_batch(self, emb1: HierarchyEmbedding, embeddings2: List[HierarchyEmbedding]) -> List[float]:
        return [self.compute(emb1, emb2) for emb2 in embeddings2]


class SemanticDistance(DistanceMetric):
    """Vector similarity using cosine/euclidean distance."""
    
    def __init__(self, use_cosine: bool = True):
        super().__init__(MetricType.SEMANTIC)
        self.use_cosine = use_cosine
    
    def compute(self, emb1: SemanticEmbedding, emb2: SemanticEmbedding) -> float:
        """Compute vector distance."""
        v1 = emb1.vector
        v2 = emb2.vector
        
        if len(v1) != len(v2):
            raise ValueError("Vector dimensions must match")
        
        if self.use_cosine:
            return 1.0 - self._cosine_similarity(v1, v2)
        else:
            return self._euclidean_distance(v1, v2)
    
    def compute_batch(self, emb1: SemanticEmbedding, embeddings2: List[SemanticEmbedding]) -> List[float]:
        return [self.compute(emb1, emb2) for emb2 in embeddings2]
    
    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)
    
    @staticmethod
    def _euclidean_distance(v1: List[float], v2: List[float]) -> float:
        """Compute euclidean distance."""
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5


class AssociationDistance(DistanceMetric):
    """Graph distance based on shortest path."""
    
    def __init__(self):
        super().__init__(MetricType.ASSOCIATION)
    
    def compute(self, emb1: AssociationEmbedding, emb2: AssociationEmbedding) -> float:
        """
        Compute association distance.
        - Direct connection: 1 / weight
        - No connection: 1.0 (default)
        
        Note: To properly use edge weights, embeddings should be passed with entity IDs
        tracked at a higher level (in MetricSpace).
        """
        # Check if emb2 has edges (indicates it's connected to something)
        if emb2.edges:
            # For now, return a distance based on edge density
            # In production, pass entity_id to compute function
            avg_weight = sum(emb2.edges.values()) / len(emb2.edges) if emb2.edges else 0
            return 1.0 - avg_weight if avg_weight > 0 else 1.0
        
        return 1.0  # Default distance for unconnected nodes
    
    def compute_batch(self, emb1: AssociationEmbedding, embeddings2: List[AssociationEmbedding]) -> List[float]:
        return [self.compute(emb1, emb2) for emb2 in embeddings2]


class CausalDistance(DistanceMetric):
    """Causal relationship distance."""
    
    def __init__(self):
        super().__init__(MetricType.CAUSAL)
    
    def compute(self, emb1: CausalEmbedding, emb2: CausalEmbedding) -> float:
        """
        Compute causal distance based on relationship strength.
        - Direct cause/effect: strength score (0-1)
        - No relationship: 1.0
        """
        # Check if emb2 is a cause or effect of emb1
        for cause_id, strength in emb1.causes:
            if cause_id == emb2.id:
                return 1.0 - strength  # Lower distance for stronger relationships
        
        for effect_id, strength in emb1.effects:
            if effect_id == emb2.id:
                return 1.0 - strength
        
        return 1.0  # No relationship
    
    def compute_batch(self, emb1: CausalEmbedding, embeddings2: List[CausalEmbedding]) -> List[float]:
        return [self.compute(emb1, emb2) for emb2 in embeddings2]
