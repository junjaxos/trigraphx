"""
Core MetricSpace implementation - the unified data structure for TriGraphX.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import heapq

from .entity import (
    Entity, DistanceMetric, MetricType, 
    HierarchyDistance, SemanticDistance, AssociationDistance, CausalDistance,
    HierarchyEmbedding, SemanticEmbedding, AssociationEmbedding, CausalEmbedding
)


@dataclass
class QueryResult:
    """Result from a metric space query."""
    entity_ids: List[str]
    distances: List[float]
    scores: List[float]  # Normalized scores (0-1, lower is better)
    
    def to_list(self) -> List[Tuple[str, float, float]]:
        return list(zip(self.entity_ids, self.distances, self.scores))


class MetricSpace:
    """
    Unified metric space for all data relationships.
    
    Replaces:
    - Tree index (via hierarchy metric)
    - Graph DB (via association metric)
    - Vector DB (via semantic metric)
    - New: Causal relationships
    """
    
    def __init__(self, max_entities: int = 1_000_000):
        self.max_entities = max_entities
        
        # Main storage
        self.entities: Dict[str, Entity] = {}
        
        # Metrics and distance functions
        self.metrics: Dict[MetricType, DistanceMetric] = {
            MetricType.HIERARCHY: HierarchyDistance(),
            MetricType.SEMANTIC: SemanticDistance(use_cosine=True),
            MetricType.ASSOCIATION: AssociationDistance(),
            MetricType.CAUSAL: CausalDistance(),
        }
        
        # Indices for fast lookups
        self.entity_index: Dict[str, str] = {}  # id -> id (for quick existence check)
        self.hierarchy_index: Dict[str, Set[str]] = defaultdict(set)  # parent_id -> children_ids
        self.embedding_index: Dict[MetricType, Dict[str, Any]] = {
            mt: {} for mt in MetricType
        }
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the space."""
        if len(self.entities) >= self.max_entities:
            raise RuntimeError(f"Metric space full ({self.max_entities} entities)")
        
        if entity.id in self.entities:
            raise ValueError(f"Entity {entity.id} already exists")
        
        self.entities[entity.id] = entity
        self.entity_index[entity.id] = entity.id
        
        # Update indices
        if MetricType.HIERARCHY in entity.embeddings:
            emb = entity.embeddings[MetricType.HIERARCHY]
            if isinstance(emb, dict):
                emb = HierarchyEmbedding.from_dict(emb)
            if emb.parent_id:
                self.hierarchy_index[emb.parent_id].add(entity.id)
            self.embedding_index[MetricType.HIERARCHY][entity.id] = emb
        
        if MetricType.SEMANTIC in entity.embeddings:
            emb = entity.embeddings[MetricType.SEMANTIC]
            if isinstance(emb, dict):
                emb = SemanticEmbedding.from_dict(emb)
            self.embedding_index[MetricType.SEMANTIC][entity.id] = emb
        
        if MetricType.ASSOCIATION in entity.embeddings:
            emb = entity.embeddings[MetricType.ASSOCIATION]
            if isinstance(emb, dict):
                emb = AssociationEmbedding.from_dict(emb)
            self.embedding_index[MetricType.ASSOCIATION][entity.id] = emb
        
        if MetricType.CAUSAL in entity.embeddings:
            emb = entity.embeddings[MetricType.CAUSAL]
            if isinstance(emb, dict):
                emb = CausalEmbedding.from_dict(emb)
            self.embedding_index[MetricType.CAUSAL][entity.id] = emb
        
        return True
    
    def get_entity(self, entity_id: str, include_deleted: bool = False) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        entity = self.entities.get(entity_id)
        if entity and not entity.deleted:
            return entity
        if include_deleted and entity:
            return entity
        return None
    
    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity with partial changes."""
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")
        
        entity = self.entities[entity_id]
        if entity.deleted:
            raise ValueError(f"Entity {entity_id} is deleted")
        
        # Apply updates
        if "data" in updates:
            entity.data.update(updates["data"])
        
        if "embeddings" in updates:
            for metric_type, emb in updates["embeddings"].items():
                entity.embeddings[metric_type] = emb
        
        if "metadata" in updates:
            entity.metadata.update(updates["metadata"])
        
        entity.updated_at = __import__('datetime').datetime.utcnow()
        
        return True
    
    def soft_delete_entity(self, entity_id: str) -> bool:
        """Soft delete (mark as deleted)."""
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")
        
        self.entities[entity_id].deleted = True
        return True
    
    def hard_delete_entity(self, entity_id: str) -> bool:
        """Hard delete (permanent removal)."""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities.pop(entity_id)
        del self.entity_index[entity_id]
        
        # Clean up indices
        for metric_type in self.embedding_index:
            self.embedding_index[metric_type].pop(entity_id, None)
        
        return True
    
    def knn_query(
        self,
        query_entity_id: str,
        k: int = 10,
        metric_type: MetricType = MetricType.SEMANTIC,
    ) -> QueryResult:
        """K-nearest neighbors query."""
        if query_entity_id not in self.entities:
            raise ValueError(f"Query entity {query_entity_id} not found")
        
        query_entity = self.entities[query_entity_id]
        if metric_type not in query_entity.embeddings:
            raise ValueError(f"Entity {query_entity_id} has no {metric_type} embedding")
        
        query_emb = query_entity.embeddings[metric_type]
        if isinstance(query_emb, dict):
            query_emb = self._dict_to_embedding(query_emb, metric_type)
        
        # Compute distances to all entities
        distances = []
        for ent_id, entity in self.entities.items():
            if ent_id == query_entity_id or entity.deleted:
                continue
            
            if metric_type not in entity.embeddings:
                continue
            
            target_emb = entity.embeddings[metric_type]
            if isinstance(target_emb, dict):
                target_emb = self._dict_to_embedding(target_emb, metric_type)
            
            dist = self.metrics[metric_type].compute(query_emb, target_emb)
            distances.append((ent_id, dist))
        
        # Sort by distance and take top-k
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]
        
        entity_ids = [eid for eid, _ in top_k]
        dists = [d for _, d in top_k]
        
        # Normalize distances to scores (0-1)
        if dists:
            max_dist = max(dists)
            scores = [d / (max_dist + 1e-10) for d in dists] if max_dist > 0 else [0] * len(dists)
        else:
            scores = []
        
        return QueryResult(
            entity_ids=entity_ids,
            distances=dists,
            scores=scores,
        )
    
    def range_query(
        self,
        query_entity_id: str,
        radius: float,
        metric_type: MetricType = MetricType.SEMANTIC,
    ) -> QueryResult:
        """Range query - find all entities within radius."""
        if query_entity_id not in self.entities:
            raise ValueError(f"Query entity {query_entity_id} not found")
        
        query_entity = self.entities[query_entity_id]
        if metric_type not in query_entity.embeddings:
            raise ValueError(f"Entity {query_entity_id} has no {metric_type} embedding")
        
        query_emb = query_entity.embeddings[metric_type]
        if isinstance(query_emb, dict):
            query_emb = self._dict_to_embedding(query_emb, metric_type)
        
        # Find all entities within radius
        results = []
        for ent_id, entity in self.entities.items():
            if ent_id == query_entity_id or entity.deleted:
                continue
            
            if metric_type not in entity.embeddings:
                continue
            
            target_emb = entity.embeddings[metric_type]
            if isinstance(target_emb, dict):
                target_emb = self._dict_to_embedding(target_emb, metric_type)
            
            dist = self.metrics[metric_type].compute(query_emb, target_emb)
            if dist <= radius:
                results.append((ent_id, dist))
        
        results.sort(key=lambda x: x[1])
        
        entity_ids = [eid for eid, _ in results]
        dists = [d for _, d in results]
        
        if dists:
            max_dist = max(dists)
            scores = [d / (max_dist + 1e-10) for d in dists] if max_dist > 0 else [0] * len(dists)
        else:
            scores = []
        
        return QueryResult(
            entity_ids=entity_ids,
            distances=dists,
            scores=scores,
        )
    
    def multi_metric_query(
        self,
        query_entity_id: str,
        k: int = 10,
        metric_weights: Optional[Dict[MetricType, float]] = None,
    ) -> QueryResult:
        """Multi-metric query - combine multiple distance metrics."""
        if metric_weights is None:
            metric_weights = {
                MetricType.SEMANTIC: 0.5,
                MetricType.HIERARCHY: 0.3,
                MetricType.ASSOCIATION: 0.2,
            }
        
        if query_entity_id not in self.entities:
            raise ValueError(f"Query entity {query_entity_id} not found")
        
        query_entity = self.entities[query_entity_id]
        combined_scores = defaultdict(float)
        
        # Compute scores from each metric
        for metric_type, weight in metric_weights.items():
            if metric_type not in query_entity.embeddings:
                continue
            
            query_emb = query_entity.embeddings[metric_type]
            if isinstance(query_emb, dict):
                query_emb = self._dict_to_embedding(query_emb, metric_type)
            
            # Compute distances
            distances = []
            for ent_id, entity in self.entities.items():
                if ent_id == query_entity_id or entity.deleted:
                    continue
                
                if metric_type not in entity.embeddings:
                    continue
                
                target_emb = entity.embeddings[metric_type]
                if isinstance(target_emb, dict):
                    target_emb = self._dict_to_embedding(target_emb, metric_type)
                
                dist = self.metrics[metric_type].compute(query_emb, target_emb)
                distances.append((ent_id, dist))
            
            # Normalize and accumulate
            if distances:
                distances.sort(key=lambda x: x[1])
                max_dist = distances[-1][1]
                
                for ent_id, dist in distances:
                    normalized = (dist / (max_dist + 1e-10)) if max_dist > 0 else 0
                    combined_scores[ent_id] += weight * normalized
        
        # Sort by combined score and take top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1])
        top_k = sorted_results[:k]
        
        entity_ids = [eid for eid, _ in top_k]
        scores = [s for _, s in top_k]
        distances = [s for s in scores]  # Use scores as distances for multi-metric
        
        return QueryResult(
            entity_ids=entity_ids,
            distances=distances,
            scores=scores,
        )
    
    def path_query(
        self,
        start_entity_id: str,
        end_entity_id: str,
        metric_type: MetricType = MetricType.ASSOCIATION,
    ) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        if start_entity_id not in self.entities or end_entity_id not in self.entities:
            raise ValueError("Start or end entity not found")
        
        # BFS to find path
        queue = [(start_entity_id, [start_entity_id])]
        visited = {start_entity_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == end_entity_id:
                return path
            
            current_entity = self.entities[current_id]
            if metric_type not in current_entity.embeddings:
                continue
            
            emb = current_entity.embeddings[metric_type]
            if isinstance(emb, dict):
                emb = AssociationEmbedding.from_dict(emb)
            
            # Explore neighbors
            for neighbor_id in emb.edges.keys():
                if neighbor_id not in visited and neighbor_id in self.entities:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None  # No path found
    
    def _dict_to_embedding(self, d: Dict[str, Any], metric_type: MetricType) -> Any:
        """Convert dict to appropriate embedding type."""
        if metric_type == MetricType.HIERARCHY:
            return HierarchyEmbedding.from_dict(d)
        elif metric_type == MetricType.SEMANTIC:
            return SemanticEmbedding.from_dict(d)
        elif metric_type == MetricType.ASSOCIATION:
            return AssociationEmbedding.from_dict(d)
        elif metric_type == MetricType.CAUSAL:
            return CausalEmbedding.from_dict(d)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the metric space."""
        total_entities = len(self.entities)
        active_entities = sum(1 for e in self.entities.values() if not e.deleted)
        deleted_entities = total_entities - active_entities
        
        embeddings_count = defaultdict(int)
        for entity in self.entities.values():
            for metric_type in entity.embeddings:
                embeddings_count[metric_type] += 1
        
        return {
            "total_entities": total_entities,
            "active_entities": active_entities,
            "deleted_entities": deleted_entities,
            "utilization": active_entities / self.max_entities,
            "embeddings_count": dict(embeddings_count),
        }
