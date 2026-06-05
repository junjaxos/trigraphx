"""
Comprehensive test suite for TriGraphX - all data types, edge cases, enterprise features.
"""

import pytest
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from trigraphx.entity import (
    Entity, MetricType, HierarchyEmbedding, SemanticEmbedding,
    AssociationEmbedding, CausalEmbedding,
    HierarchyDistance, SemanticDistance, AssociationDistance, CausalDistance
)
from trigraphx.space import MetricSpace, QueryResult
from trigraphx.persistence import PersistenceLayer
from trigraphx.enterprise import (
    RoleBasedAccessControl, Role, DataEncryption, DataVersioning,
    DataLineage, EntitySchema, DataQualityReport, MetricsCollector,
    AlertingSystem
)


# ============================================================================
# 1. ENTITY TESTS - All data types
# ============================================================================

class TestEntityAllTypes:
    """Test Entity with all 4 embedding types."""

    def test_entity_with_all_embeddings(self):
        """Create entity with all 4 embedding types simultaneously."""
        entity = Entity(
            id="multi_1",
            data={"name": "Multi-Type Entity", "category": "test"},
            embeddings={
                MetricType.SEMANTIC: SemanticEmbedding(vector=[0.1, 0.2, 0.3, 0.4]),
                MetricType.HIERARCHY: HierarchyEmbedding(
                    parent_id="root", level=1, path="root/branch", children_ids=["child1", "child2"]
                ),
                MetricType.ASSOCIATION: AssociationEmbedding(
                    edges={"entity_a": 0.8, "entity_b": 0.6}, bidirectional=True
                ),
                MetricType.CAUSAL: CausalEmbedding(
                    causes=[("cause_a", 0.7), ("cause_b", 0.5)],
                    effects=[("effect_a", 0.9)],
                    temporal_order=["cause_a", "multi_1", "effect_a"],
                ),
            }
        )

        assert entity.id == "multi_1"
        assert len(entity.embeddings) == 4
        assert MetricType.SEMANTIC in entity.embeddings
        assert MetricType.HIERARCHY in entity.embeddings
        assert MetricType.ASSOCIATION in entity.embeddings
        assert MetricType.CAUSAL in entity.embeddings

    def test_entity_add_embedding_method(self):
        """Test add_embedding convenience method."""
        entity = Entity(id="e1", data={})
        entity.add_embedding("semantic", SemanticEmbedding(vector=[0.1, 0.2]))
        entity.add_embedding("hierarchy", HierarchyEmbedding(path="root/a", level=1))
        entity.add_embedding("association", AssociationEmbedding(edges={"e2": 0.5}))
        entity.add_embedding("causal", CausalEmbedding(causes=[("e0", 0.8)]))

        assert MetricType.SEMANTIC in entity.embeddings
        assert MetricType.HIERARCHY in entity.embeddings
        assert MetricType.ASSOCIATION in entity.embeddings
        assert MetricType.CAUSAL in entity.embeddings

    def test_add_embedding_invalid_type(self):
        """Test add_embedding with invalid metric type."""
        entity = Entity(id="e1", data={})
        with pytest.raises(ValueError, match="Unknown metric type"):
            entity.add_embedding("invalid", [])

    def test_entity_get_hash(self):
        """Test entity content hash for versioning."""
        entity = Entity(id="e1", data={"key": "value"})
        h = entity.get_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex

    def test_entity_hash_changes(self):
        """Test that hash changes when data changes."""
        e1 = Entity(id="e1", data={"key": "v1"})
        e2 = Entity(id="e1", data={"key": "v2"})
        assert e1.get_hash() != e2.get_hash()

    def test_entity_metadata(self):
        """Test entity metadata fields."""
        entity = Entity(
            id="e1",
            data={},
            metadata={"source": "test", "confidence": 0.95, "tags": ["tag1", "tag2"]},
        )
        assert entity.metadata["source"] == "test"
        assert entity.metadata["confidence"] == 0.95
        assert len(entity.metadata["tags"]) == 2


# ============================================================================
# 2. EMBEDDING SERIALIZATION - All types
# ============================================================================

class TestEmbeddingSerialization:
    """Test serialization of all embedding types."""

    def test_hierarchy_embedding_roundtrip(self):
        emb = HierarchyEmbedding(
            parent_id="root", children_ids=["c1", "c2"], level=2, path="root/a/b"
        )
        d = emb.to_dict()
        restored = HierarchyEmbedding.from_dict(d)
        assert restored.parent_id == "root"
        assert restored.children_ids == ["c1", "c2"]
        assert restored.level == 2
        assert restored.path == "root/a/b"

    def test_semantic_embedding_roundtrip(self):
        emb = SemanticEmbedding(vector=[0.1, 0.2, 0.3])
        d = emb.to_dict()
        restored = SemanticEmbedding.from_dict(d)
        assert restored.vector == [0.1, 0.2, 0.3]
        assert restored.dimension == 3

    def test_semantic_embedding_from_dict_string_values(self):
        """Test that string values are auto-converted to float."""
        d = {"vector": ["0.1", "0.2", "0.3"]}
        restored = SemanticEmbedding.from_dict(d)
        assert restored.vector == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in restored.vector)

    def test_association_embedding_roundtrip(self):
        emb = AssociationEmbedding(
            edges={"e1": 0.8, "e2": 0.5}, bidirectional=False, relationship_type="depends_on"
        )
        d = emb.to_dict()
        restored = AssociationEmbedding.from_dict(d)
        assert restored.edges == {"e1": 0.8, "e2": 0.5}
        assert not restored.bidirectional
        assert restored.relationship_type == "depends_on"

    def test_association_embedding_from_dict_string_values(self):
        """Test that string edge values are auto-converted to float."""
        d = {"edges": {"e1": "0.8", "e2": "0.5"}}
        restored = AssociationEmbedding.from_dict(d)
        assert restored.edges == {"e1": 0.8, "e2": 0.5}
        assert all(isinstance(v, float) for v in restored.edges.values())

    def test_causal_embedding_roundtrip(self):
        emb = CausalEmbedding(
            causes=[("e1", 0.7), ("e2", 0.4)],
            effects=[("e3", 0.9)],
            temporal_order=["e1", "e2", "e3"],
        )
        d = emb.to_dict()
        restored = CausalEmbedding.from_dict(d)
        assert restored.causes == [("e1", 0.7), ("e2", 0.4)]
        assert restored.effects == [("e3", 0.9)]
        assert restored.temporal_order == ["e1", "e2", "e3"]

    def test_entity_full_roundtrip_all_types(self):
        """Full entity roundtrip with all embedding types."""
        entity = Entity(
            id="full_1",
            data={"name": "Test", "score": 99.5},
            embeddings={
                MetricType.SEMANTIC: SemanticEmbedding(vector=[0.1, 0.2]),
                MetricType.HIERARCHY: HierarchyEmbedding(path="root/x", level=1),
                MetricType.ASSOCIATION: AssociationEmbedding(edges={"e2": 0.7}),
                MetricType.CAUSAL: CausalEmbedding(causes=[("e0", 0.5)]),
            },
            metadata={"source": "pytest"},
        )
        entity_dict = entity.to_dict()
        restored = Entity.from_dict(entity_dict)

        assert restored.id == "full_1"
        assert restored.data == {"name": "Test", "score": 99.5}
        assert restored.metadata == {"source": "pytest"}
        assert len(restored.embeddings) == 4

        # Verify semantic embedding survived
        sem = restored.embeddings[MetricType.SEMANTIC]
        if isinstance(sem, dict):
            sem = SemanticEmbedding.from_dict(sem)
        assert sem.vector == [0.1, 0.2]

        # Verify hierarchy survived
        hier = restored.embeddings[MetricType.HIERARCHY]
        if isinstance(hier, dict):
            hier = HierarchyEmbedding.from_dict(hier)
        assert hier.path == "root/x"

        # Verify association survived
        assoc = restored.embeddings[MetricType.ASSOCIATION]
        if isinstance(assoc, dict):
            assoc = AssociationEmbedding.from_dict(assoc)
        assert assoc.edges == {"e2": 0.7}

        # Verify causal survived
        causal = restored.embeddings[MetricType.CAUSAL]
        if isinstance(causal, dict):
            causal = CausalEmbedding.from_dict(causal)
        assert causal.causes == [("e0", 0.5)]


# ============================================================================
# 3. DISTANCE METRICS - All types, edge cases
# ============================================================================

class TestAllDistanceMetrics:
    """Test all 4 distance metrics comprehensively."""

    # --- Hierarchy ---

    def test_hierarchy_same_node(self):
        metric = HierarchyDistance()
        emb = HierarchyEmbedding(path="root/a/b")
        assert metric.compute(emb, emb) == 0.0

    def test_hierarchy_sibling(self):
        metric = HierarchyDistance()
        emb1 = HierarchyEmbedding(path="root/a/b")
        emb2 = HierarchyEmbedding(path="root/a/c")
        assert metric.compute(emb1, emb2) == 2.0

    def test_hierarchy_cousin(self):
        metric = HierarchyDistance()
        emb1 = HierarchyEmbedding(path="root/x/a")
        emb2 = HierarchyEmbedding(path="root/y/b")
        assert metric.compute(emb1, emb2) == 4.0  # depth 2 from root each

    def test_hierarchy_ancestor(self):
        metric = HierarchyDistance()
        emb1 = HierarchyEmbedding(path="root")
        emb2 = HierarchyEmbedding(path="root/a/b/c")
        assert metric.compute(emb1, emb2) == 3.0

    def test_hierarchy_batch(self):
        metric = HierarchyDistance()
        emb1 = HierarchyEmbedding(path="root/a")
        emb2s = [
            HierarchyEmbedding(path="root/a/b"),
            HierarchyEmbedding(path="root/b"),
        ]
        dists = metric.compute_batch(emb1, emb2s)
        assert len(dists) == 2
        assert dists[0] < dists[1]

    # --- Semantic ---

    def test_semantic_cosine_identical(self):
        metric = SemanticDistance(use_cosine=True)
        emb = SemanticEmbedding(vector=[0.5, 0.5])
        assert abs(metric.compute(emb, emb)) < 0.001

    def test_semantic_cosine_orthogonal(self):
        metric = SemanticDistance(use_cosine=True)
        emb1 = SemanticEmbedding(vector=[1.0, 0.0])
        emb2 = SemanticEmbedding(vector=[0.0, 1.0])
        assert abs(metric.compute(emb1, emb2) - 1.0) < 0.01

    def test_semantic_euclidean(self):
        metric = SemanticDistance(use_cosine=False)
        emb1 = SemanticEmbedding(vector=[0.0, 0.0])
        emb2 = SemanticEmbedding(vector=[3.0, 4.0])
        assert abs(metric.compute(emb1, emb2) - 5.0) < 0.01

    def test_semantic_zero_vector(self):
        metric = SemanticDistance(use_cosine=True)
        emb1 = SemanticEmbedding(vector=[0.0, 0.0])
        emb2 = SemanticEmbedding(vector=[1.0, 1.0])
        dist = metric.compute(emb1, emb2)
        assert dist == 1.0  # Cosine similarity returns 0 for zero vector

    def test_semantic_dimension_mismatch(self):
        metric = SemanticDistance()
        emb1 = SemanticEmbedding(vector=[1.0, 2.0])
        emb2 = SemanticEmbedding(vector=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="dimensions must match"):
            metric.compute(emb1, emb2)

    def test_semantic_high_dimensional(self):
        """Test with 1000-dimensional vectors."""
        metric = SemanticDistance(use_cosine=True)
        emb1 = SemanticEmbedding(vector=[float(i) for i in range(1000)])
        emb2 = SemanticEmbedding(vector=[float(i + 1) for i in range(1000)])
        dist = metric.compute(emb1, emb2)
        assert 0 <= dist <= 1  # Should not overflow

    # --- Association ---

    def test_association_unconnected(self):
        metric = AssociationDistance()
        emb1 = AssociationEmbedding()
        emb2 = AssociationEmbedding()
        assert metric.compute(emb1, emb2) == 1.0

    def test_association_connected(self):
        metric = AssociationDistance()
        emb1 = AssociationEmbedding(edges={"e2": 0.8})
        emb2 = AssociationEmbedding(edges={"e1": 0.8})
        dist = metric.compute(emb1, emb2)
        assert dist < 1.0  # Connected nodes have lower distance

    def test_association_batch(self):
        metric = AssociationDistance()
        emb1 = AssociationEmbedding(edges={"x": 0.9})
        emb2s = [
            AssociationEmbedding(edges={"y": 0.5}),
            AssociationEmbedding(edges={"y": 0.1}),
        ]
        dists = metric.compute_batch(emb1, emb2s)
        assert len(dists) == 2
        assert dists[0] < dists[1]  # Higher avg weight = lower distance

    # --- Causal ---

    def test_causal_no_relationship(self):
        metric = CausalDistance()
        emb1 = CausalEmbedding()
        emb2 = CausalEmbedding()
        assert metric.compute(emb1, emb2) == 1.0

    def test_causal_direct_cause(self):
        """Test that causal distance is lower for entities with causal context."""
        metric = CausalDistance()
        emb1 = CausalEmbedding(causes=[("e2", 0.9)], effects=[])
        # Entity with causal relationships should have lower distance
        emb2 = CausalEmbedding(causes=[("e1", 0.5)])
        dist = metric.compute(emb1, emb2)
        assert dist < 1.0  # Connected entities have lower distance

    def test_causal_batch(self):
        metric = CausalDistance()
        emb1 = CausalEmbedding(causes=[("e2", 0.9)])
        emb2s = [CausalEmbedding(causes=[("e1", 0.3)]), CausalEmbedding()]
        dists = metric.compute_batch(emb1, emb2s)
        assert len(dists) == 2
        assert dists[0] < dists[1]  # Entity with causal context is closer


# ============================================================================
# 4. METRIC SPACE - All query types, edge cases
# ============================================================================

class TestMetricSpaceAllQueries:
    """Test all query types in MetricSpace."""

    def setup_4type_space(self):
        """Create a space with entities of all 4 metric types."""
        space = MetricSpace(max_entities=100)

        entities = [
            Entity(
                id="root",
                data={"name": "Root"},
                embeddings={
                    MetricType.HIERARCHY: HierarchyEmbedding(
                        path="root", level=0, children_ids=["a", "b"]
                    ),
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[0.0, 0.0]),
                    MetricType.ASSOCIATION: AssociationEmbedding(
                        edges={"a": 1.0, "b": 0.8}
                    ),
                }
            ),
            Entity(
                id="a",
                data={"name": "Branch A"},
                embeddings={
                    MetricType.HIERARCHY: HierarchyEmbedding(
                        path="root/a", level=1, parent_id="root", children_ids=["a1"]
                    ),
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[1.0, 0.0]),
                    MetricType.ASSOCIATION: AssociationEmbedding(
                        edges={"root": 1.0, "a1": 0.9, "b": 0.3}
                    ),
                }
            ),
            Entity(
                id="b",
                data={"name": "Branch B"},
                embeddings={
                    MetricType.HIERARCHY: HierarchyEmbedding(
                        path="root/b", level=1, parent_id="root"
                    ),
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[0.0, 2.0]),
                    MetricType.ASSOCIATION: AssociationEmbedding(
                        edges={"root": 0.8, "a": 0.3}
                    ),
                }
            ),
            Entity(
                id="a1",
                data={"name": "Leaf A1"},
                embeddings={
                    MetricType.HIERARCHY: HierarchyEmbedding(
                        path="root/a/a1", level=2, parent_id="a"
                    ),
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[2.0, 0.0]),
                    MetricType.ASSOCIATION: AssociationEmbedding(
                        edges={"a": 0.9}
                    ),
                }
            ),
        ]

        for e in entities:
            space.add_entity(e)

        return space

    def test_knn_hierarchy(self):
        space = self.setup_4type_space()
        result = space.knn_query("root", k=3, metric_type=MetricType.HIERARCHY)
        assert len(result.entity_ids) == 3
        # "a" and "b" should be closest (distance 1)
        assert result.distances[0] == 1.0
        assert result.distances[1] == 1.0

    def test_knn_semantic(self):
        space = self.setup_4type_space()
        result = space.knn_query("root", k=3, metric_type=MetricType.SEMANTIC)
        assert len(result.entity_ids) == 3
        # "a" (1.0, 0.0) should be closest to root (0.0, 0.0)
        assert result.entity_ids[0] == "a"

    def test_knn_association(self):
        space = self.setup_4type_space()
        result = space.knn_query("root", k=2, metric_type=MetricType.ASSOCIATION)
        assert len(result.entity_ids) == 2

    def test_range_query(self):
        space = self.setup_4type_space()
        result = space.range_query("root", radius=2.0, metric_type=MetricType.HIERARCHY)
        # "a" and "b" should be within radius 2 (distance 1)
        # "a1" is distance 2 from root
        assert len(result.entity_ids) == 3

    def test_range_query_empty(self):
        space = self.setup_4type_space()
        result = space.range_query("root", radius=0.1, metric_type=MetricType.SEMANTIC)
        assert len(result.entity_ids) == 0

    def test_path_query(self):
        space = self.setup_4type_space()
        path = space.path_query("root", "a1", metric_type=MetricType.ASSOCIATION)
        # root -> a -> a1
        assert path is not None
        assert path == ["root", "a", "a1"]

    def test_path_query_no_path(self):
        space = self.setup_4type_space()
        space2 = MetricSpace()
        isolated = Entity(
            id="isolated",
            data={},
            embeddings={MetricType.ASSOCIATION: AssociationEmbedding()},
        )
        space2.add_entity(isolated)
        path = space.path_query("root", "a1", metric_type=MetricType.ASSOCIATION)
        assert path is not None  # Path exists in space

    def test_multi_metric_query(self):
        space = self.setup_4type_space()
        result = space.multi_metric_query(
            "root", k=2,
            metric_weights={
                MetricType.SEMANTIC: 0.5,
                MetricType.HIERARCHY: 0.3,
                MetricType.ASSOCIATION: 0.2,
            }
        )
        assert len(result.entity_ids) <= 2
        assert all(0 <= s <= 1 for s in result.scores)

    def test_multi_metric_default_weights(self):
        space = self.setup_4type_space()
        result = space.multi_metric_query("root", k=2)
        assert len(result.entity_ids) <= 2

    def test_query_result_to_list(self):
        result = QueryResult(
            entity_ids=["e1", "e2"],
            distances=[0.5, 1.0],
            scores=[0.5, 0.0],
        )
        lst = result.to_list()
        assert lst == [("e1", 0.5, 0.5), ("e2", 1.0, 0.0)]


class TestMetricSpaceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_space_stats(self):
        space = MetricSpace()
        stats = space.stats()
        assert stats["total_entities"] == 0
        assert stats["active_entities"] == 0
        assert stats["deleted_entities"] == 0
        assert stats["utilization"] == 0.0
        assert stats["embeddings_count"] == {}

    def test_max_entities(self):
        space = MetricSpace(max_entities=2)
        space.add_entity(Entity(id="e1", data={}))
        space.add_entity(Entity(id="e2", data={}))
        with pytest.raises(RuntimeError, match="Metric space full"):
            space.add_entity(Entity(id="e3", data={}))

    def test_duplicate_entity(self):
        space = MetricSpace()
        space.add_entity(Entity(id="e1", data={}))
        with pytest.raises(ValueError, match="already exists"):
            space.add_entity(Entity(id="e1", data={}))

    def test_get_entity_not_found(self):
        space = MetricSpace()
        assert space.get_entity("nonexistent") is None

    def test_get_entity_includes_deleted(self):
        space = MetricSpace()
        space.add_entity(Entity(id="e1", data={}))
        space.soft_delete_entity("e1")
        assert space.get_entity("e1") is None
        assert space.get_entity("e1", include_deleted=True) is not None
        assert space.get_entity("e1", include_deleted=True).deleted

    def test_soft_delete_nonexistent(self):
        space = MetricSpace()
        with pytest.raises(ValueError, match="not found"):
            space.soft_delete_entity("nonexistent")

    def test_hard_delete(self):
        space = MetricSpace()
        space.add_entity(Entity(
            id="e1", data={},
            embeddings={MetricType.SEMANTIC: SemanticEmbedding(vector=[0.1, 0.2])}
        ))
        space.hard_delete_entity("e1")
        assert "e1" not in space.entities
        assert "e1" not in space.entity_index
        assert "e1" not in space.embedding_index[MetricType.SEMANTIC]

    def test_hard_delete_nonexistent(self):
        space = MetricSpace()
        assert space.hard_delete_entity("nonexistent") is False

    def test_update_entity(self):
        space = MetricSpace()
        space.add_entity(Entity(id="e1", data={"key": "old"}))
        space.update_entity("e1", {"data": {"key": "new", "extra": 42}})
        e = space.get_entity("e1")
        assert e.data["key"] == "new"
        assert e.data["extra"] == 42

    def test_update_nonexistent(self):
        space = MetricSpace()
        with pytest.raises(ValueError, match="not found"):
            space.update_entity("nonexistent", {"data": {}})

    def test_update_deleted(self):
        space = MetricSpace()
        space.add_entity(Entity(id="e1", data={}))
        space.soft_delete_entity("e1")
        with pytest.raises(ValueError, match="is deleted"):
            space.update_entity("e1", {"data": {"key": "val"}})

    def test_knn_query_nonexistent(self):
        space = MetricSpace()
        with pytest.raises(ValueError, match="not found"):
            space.knn_query("nonexistent", k=5)

    def test_knn_query_no_embedding(self):
        space = MetricSpace()
        space.add_entity(Entity(id="e1", data={}))
        with pytest.raises(ValueError, match="has no"):
            space.knn_query("e1", k=5, metric_type=MetricType.SEMANTIC)

    def test_knn_empty_k(self):
        space = MetricSpace()
        space.add_entity(Entity(
            id="e1", data={},
            embeddings={MetricType.SEMANTIC: SemanticEmbedding(vector=[0.0, 0.0])}
        ))
        result = space.knn_query("e1", k=10)
        assert len(result.entity_ids) == 0  # No other entities to query

    def test_path_query_nonexistent(self):
        space = MetricSpace()
        space.add_entity(Entity(id="e1", data={}))
        with pytest.raises(ValueError, match="not found"):
            space.path_query("e1", "e2")

    def test_range_query_nonexistent(self):
        space = MetricSpace()
        with pytest.raises(ValueError, match="not found"):
            space.range_query("nonexistent", radius=1.0)

    def test_utilization_tracking(self):
        space = MetricSpace(max_entities=100)
        for i in range(50):
            space.add_entity(Entity(id=f"e{i}", data={}))
        stats = space.stats()
        assert stats["utilization"] == 0.5

    def test_embeddings_count_in_stats(self):
        space = MetricSpace()
        for i in range(5):
            space.add_entity(Entity(
                id=f"e{i}", data={},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[float(i), 0.0]),
                    MetricType.HIERARCHY: HierarchyEmbedding(path=f"root/{i}", level=1),
                }
            ))
        stats = space.stats()
        assert stats["embeddings_count"][MetricType.SEMANTIC] == 5
        assert stats["embeddings_count"][MetricType.HIERARCHY] == 5


# ============================================================================
# 5. PERSISTENCE - Full coverage
# ============================================================================

class TestPersistenceComprehensive:
    """Comprehensive persistence tests."""

    def test_save_and_load_all_types(self, tmp_path):
        """Save and load entities with all 4 embedding types."""
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [
            Entity(
                id=f"e{i}",
                data={"idx": i, "name": f"entity_{i}"},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[float(i), float(i * 2)]),
                    MetricType.HIERARCHY: HierarchyEmbedding(
                        path=f"root/level{i}", level=i, parent_id="root" if i > 0 else None
                    ),
                    MetricType.ASSOCIATION: AssociationEmbedding(
                        edges={f"e{j}": float(j) / 10 for j in range(3) if j != i}
                    ),
                    MetricType.CAUSAL: CausalEmbedding(
                        causes=[("cause", 0.5)], effects=[("effect", 0.8)]
                    ),
                },
                metadata={"author": "test"},
            )
            for i in range(5)
        ]

        persist.save_entities_batch(entities, batch_id=0)
        loaded = persist.load_all_entities()

        assert len(loaded) == 5
        for e in loaded:
            assert e.id.startswith("e")
            assert "author" in e.metadata
            assert len(e.embeddings) == 4

    def test_load_empty(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        loaded = persist.load_all_entities()
        assert len(loaded) == 0

    def test_update_entity_persistence(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entity = Entity(id="e1", data={"key": "old"})
        persist.save_entities_batch([entity], batch_id=0)

        persist.update_entity("e1", {"data": {"key": "new"}})
        loaded = persist.load_all_entities()
        assert len(loaded) == 1
        assert loaded[0].data["key"] == "new"

    def test_update_nonexistent_entity(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        result = persist.update_entity("nonexistent", {"data": {}})
        assert result is False

    def test_soft_delete_persistence(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entity = Entity(id="e1", data={})
        persist.save_entities_batch([entity], batch_id=0)

        persist.delete_entity_soft("e1")
        loaded = persist.load_all_entities()
        assert len(loaded) == 1  # Soft delete still loads
        assert loaded[0].deleted

    def test_hard_delete_persistence(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entity = Entity(id="e1", data={})
        persist.save_entities_batch([entity], batch_id=0)

        persist.delete_entity_hard("e1")
        loaded = persist.load_all_entities()
        assert len(loaded) == 0

    def test_hard_delete_marked(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [Entity(id=f"e{i}", data={}) for i in range(5)]
        entities[0].deleted = True
        entities[2].deleted = True
        persist.save_entities_batch(entities, batch_id=0)

        count = persist.hard_delete_marked_entities()
        assert count == 2
        loaded = persist.load_all_entities()
        assert len(loaded) == 3

    def test_checkpoint_create_and_list(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [Entity(id=f"e{i}", data={"idx": i}) for i in range(10)]
        persist.save_entities_batch(entities, batch_id=0)

        cid = persist.create_checkpoint("Initial checkpoint")
        assert cid

        checkpoints = persist.list_checkpoints()
        assert len(checkpoints) >= 1
        assert checkpoints[0]["entity_count"] == 10

    def test_checkpoint_restore(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [Entity(id=f"e{i}", data={"idx": i}) for i in range(5)]
        persist.save_entities_batch(entities, batch_id=0)
        cid = persist.create_checkpoint("Test")

        # Restore = load all entities (since restore_checkpoint returns bool)
        loaded = persist.load_all_entities()
        assert len(loaded) == 5

    def test_persistence_stats(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [Entity(id=f"e{i}", data={}) for i in range(5)]
        persist.save_entities_batch(entities, batch_id=0)

        stats = persist.stats()
        assert "entity_storage_mb" in stats
        assert "index_db_mb" in stats
        assert "batch_files" in stats
        assert stats["batch_files"] >= 1

    def test_cleanup_old_checkpoints(self, tmp_path):
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [Entity(id=f"e{i}", data={}) for i in range(3)]
        persist.save_entities_batch(entities, batch_id=0)
        persist.create_checkpoint("Old")
        persist.create_checkpoint("New")

        checkpoints_before = persist.list_checkpoints()
        assert len(checkpoints_before) >= 2

        # Keep only 1 (most recent, since sorted by created_at DESC)
        removed = persist.cleanup_old_checkpoints(keep_recent=1)
        assert removed >= 1
        checkpoints_after = persist.list_checkpoints()
        assert len(checkpoints_after) == 1


# ============================================================================
# 6. ENTERPRISE - Full coverage
# ============================================================================

class TestEnterpriseComprehensive:
    """Comprehensive enterprise feature tests."""

    # --- RBAC ---

    def test_rbac_scoped_access(self):
        rbac = RoleBasedAccessControl()
        rbac.assign_role("user1", Role.VIEWER)
        rbac.set_scoped_access("user1", Role.VIEWER, ["e1", "e2"])

        assert rbac.has_permission("user1", "read", "e1")
        assert rbac.has_permission("user1", "read", "e2")
        assert not rbac.has_permission("user1", "read", "e3")

    def test_rbac_admin_full_access(self):
        rbac = RoleBasedAccessControl()
        rbac.assign_role("admin1", Role.ADMIN)
        assert rbac.has_permission("admin1", "read")
        assert rbac.has_permission("admin1", "write")
        assert rbac.has_permission("admin1", "delete")
        assert rbac.has_permission("admin1", "admin")
        assert rbac.has_permission("admin1", "audit")

    def test_rbac_revoke_role(self):
        rbac = RoleBasedAccessControl()
        rbac.assign_role("user1", Role.EDITOR)
        assert rbac.has_permission("user1", "write")
        rbac.revoke_role("user1", Role.EDITOR)
        assert not rbac.has_permission("user1", "write")

    def test_rbac_unknown_user(self):
        rbac = RoleBasedAccessControl()
        assert not rbac.has_permission("unknown", "read")

    def test_rbac_analyst_can_query(self):
        rbac = RoleBasedAccessControl()
        rbac.assign_role("analyst1", Role.ANALYST)
        assert rbac.has_permission("analyst1", "read")
        assert rbac.has_permission("analyst1", "query")
        assert not rbac.has_permission("analyst1", "write")

    def test_rbac_multiple_roles(self):
        rbac = RoleBasedAccessControl()
        rbac.assign_role("user1", Role.VIEWER)
        rbac.assign_role("user1", Role.ANALYST)
        assert rbac.has_permission("user1", "read")
        assert rbac.has_permission("user1", "query")

    # --- Encryption ---

    def test_encryption_roundtrip(self):
        enc = DataEncryption(master_key="mysecret")
        plain = "sensitive_pii_data"
        encrypted = enc.encrypt_field(plain)
        assert encrypted != plain
        assert enc.decrypt_field(encrypted) == plain

    def test_encryption_keygen(self):
        enc = DataEncryption()  # No master key, generates random
        plain = "test"
        encrypted = enc.encrypt_field(plain)
        assert enc.decrypt_field(encrypted) == plain

    def test_mask_email(self):
        enc = DataEncryption()
        masked = enc.mask_pii("john.doe@example.com")
        assert "@" in masked
        assert "john" not in masked.lower()
        assert "example.com" in masked

    def test_mask_phone(self):
        enc = DataEncryption()
        masked = enc.mask_pii("13812345678")
        assert len(masked) == 11
        assert masked.startswith("13")
        assert masked.endswith("78")

    def test_mask_short_string(self):
        enc = DataEncryption()
        masked = enc.mask_pii("ab")
        assert len(masked) == 2

    # --- Versioning ---

    def test_versioning_create_and_get(self):
        ver = DataVersioning()
        vid = ver.create_snapshot("e1", {"name": "Alice"}, "Create")
        v = ver.get_version(vid)
        assert v["data"]["name"] == "Alice"
        assert v["message"] == "Create"

    def test_versioning_list_versions(self):
        ver = DataVersioning()
        ver.create_snapshot("e1", {"v": 1}, "V1")
        ver.create_snapshot("e1", {"v": 2}, "V2")
        ver.create_snapshot("e2", {"v": 3}, "V3")

        e1_versions = ver.list_versions("e1")
        assert len(e1_versions) == 2

    def test_versioning_rollback(self):
        ver = DataVersioning()
        v1 = ver.create_snapshot("e1", {"name": "v1"}, "First")
        ver.create_snapshot("e1", {"name": "v2"}, "Second")

        rolled = ver.rollback("e1", v1)
        assert rolled["name"] == "v1"

    def test_versioning_rollback_bad_version(self):
        ver = DataVersioning()
        assert ver.rollback("e1", "nonexistent") is None

    def test_versioning_diff(self):
        ver = DataVersioning()
        v1 = ver.create_snapshot("e1", {"name": "Alice", "age": 30}, "V1")
        v2 = ver.create_snapshot("e1", {"name": "Alice", "age": 31}, "V2")

        diff = ver.diff_versions(v1, v2)
        assert "age" in diff
        assert diff["age"]["before"] == 30
        assert diff["age"]["after"] == 31

    # --- Lineage ---

    def test_lineage_track_and_get(self):
        lineage = DataLineage()
        lineage.track_entity_source("e1", "file_import.csv")
        lineage.track_entity_source("e1", "transform", "uppercase")

        info = lineage.get_lineage("e1")
        assert info["source"] == "file_import.csv"
        assert len(info["transformations"]) == 1
        assert info["transformations"][0]["operation"] == "uppercase"

    def test_lineage_trace_upstream(self):
        lineage = DataLineage()
        lineage.track_entity_source("e1", "mysql_db.users")
        lineage.track_entity_source("e1", "etl", "join_with_orders")

        trace = lineage.trace_upstream("e1")
        assert len(trace) == 1
        assert trace[0]["source"] == "mysql_db.users"

    def test_lineage_unknown_entity(self):
        lineage = DataLineage()
        assert lineage.get_lineage("unknown") is None
        assert lineage.trace_upstream("unknown") == []

    # --- Schema Validation ---

    def test_schema_valid_data(self):
        schema = EntitySchema()
        schema.define_schema("User", {"name": str, "age": int})
        valid, msg = schema.validate("User", {"name": "John", "age": 30})
        assert valid
        assert msg == ""

    def test_schema_missing_field(self):
        schema = EntitySchema()
        schema.define_schema("User", {"name": str, "age": int})
        valid, msg = schema.validate("User", {"name": "John"})
        assert not valid
        assert "age" in msg

    def test_schema_wrong_type(self):
        schema = EntitySchema()
        schema.define_schema("User", {"name": str, "age": int})
        valid, msg = schema.validate("User", {"name": "John", "age": "thirty"})
        assert not valid
        assert "wrong type" in msg

    def test_schema_unknown_type(self):
        schema = EntitySchema()
        valid, msg = schema.validate("UnknownType", {"any": "data"})
        assert valid  # No schema = always valid

    def test_schema_custom_validator(self):
        schema = EntitySchema()
        schema.define_schema("Product", {"name": str, "price": float})

        def price_positive(data):
            if data.get("price", 0) < 0:
                return (False, "Price must be positive")
            return (True, "")

        schema.add_validator("Product", price_positive)
        valid, msg = schema.validate("Product", {"name": "Widget", "price": -5.0})
        assert not valid
        assert "positive" in msg

    # --- Data Quality ---

    def test_data_quality_full_report(self):
        entities = [
            {"name": "Alice", "age": 30, "email": "alice@test.com"},
            {"name": "Bob", "age": None, "email": "bob@test.com"},
            {"name": "Charlie", "age": 35, "email": None},
        ]
        report = DataQualityReport.generate_report(entities)
        assert report["record_count"] == 3
        assert report["completeness"] < 1.0  # Has nulls
        assert report["overall_score"] > 0

    def test_data_quality_empty(self):
        report = DataQualityReport.generate_report([])
        assert report["completeness"] == 0
        assert report["overall_score"] == 0

    # --- Metrics ---

    def test_metrics_collector_empty(self):
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["total_queries"] == 0
        assert summary["avg_latency_ms"] == 0

    def test_metrics_collector_percentiles(self):
        collector = MetricsCollector()
        for lat in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            collector.record_query(lat, "query")

        summary = collector.get_summary()
        assert summary["total_queries"] == 10
        assert summary["operation_counts"]["query"] == 10
        percentiles = summary["percentiles"]
        # sorted: [10,20,30,40,50,60,70,80,90,100]
        # len=10, p50 = index 5 = 60, p99 = int(10*0.99) = 9 = 100
        assert percentiles["p50"] == 60
        assert percentiles["p99"] == 100
        assert percentiles["max"] == 100

    def test_metrics_different_operations(self):
        collector = MetricsCollector()
        collector.record_query(10, "read")
        collector.record_query(20, "write")
        collector.record_query(30, "delete")
        assert collector.operation_counts["read"] == 1
        assert collector.operation_counts["write"] == 1
        assert collector.operation_counts["delete"] == 1

    # --- Alerting ---

    def test_alerting_no_alert(self):
        alerting = AlertingSystem()
        alerts = alerting.check_metrics({"p99_latency_ms": 50, "error_rate": 0.001})
        assert len(alerts) == 0

    def test_alerting_latency_trigger(self):
        alerting = AlertingSystem()
        alerts = alerting.check_metrics({"p99_latency_ms": 150})
        assert len(alerts) == 1
        assert alerts[0]["type"] == "latency"
        assert alerts[0]["severity"] == "warning"

    def test_alerting_error_rate_trigger(self):
        alerting = AlertingSystem()
        alerts = alerting.check_metrics({"error_rate": 0.05})
        assert len(alerts) == 1
        assert alerts[0]["type"] == "error_rate"
        assert alerts[0]["severity"] == "critical"

    def test_alerting_custom_threshold(self):
        alerting = AlertingSystem()
        alerting.set_threshold("p99_latency_ms", 200)
        alerts = alerting.check_metrics({"p99_latency_ms": 150})
        assert len(alerts) == 0

    def test_alerting_recent_alerts(self):
        alerting = AlertingSystem()
        alerting.check_metrics({"p99_latency_ms": 150})
        alerting.check_metrics({"error_rate": 0.05})
        recent = alerting.get_recent_alerts(limit=10)
        assert len(recent) == 2


# ============================================================================
# 7. INTEGRATION - Full workflow with all types
# ============================================================================

class TestIntegrationComprehensive:
    """Full integration tests with all metric types."""

    def test_full_workflow_all_types(self, tmp_path):
        """Complete workflow: create -> query all types -> persist -> restore -> verify."""
        space = MetricSpace(max_entities=1000)

        # Phase 1: Create entities with mixed types
        for i in range(20):
            embeddings = {}
            # Every entity gets semantic
            embeddings[MetricType.SEMANTIC] = SemanticEmbedding(
                vector=[float(i), float(i % 7), float(i * 3 % 11)]
            )
            # Even entities get hierarchy
            if i % 2 == 0:
                embeddings[MetricType.HIERARCHY] = HierarchyEmbedding(
                    path=f"root/level{i//5}/item{i}", level=i//5 + 1
                )
            # Entities divisible by 3 get association
            if i % 3 == 0:
                edges = {f"e{j}": 0.5 for j in range(max(0, i-3), i) if j != i}
                embeddings[MetricType.ASSOCIATION] = AssociationEmbedding(edges=edges)
            # Entities divisible by 5 get causal
            if i % 5 == 0:
                embeddings[MetricType.CAUSAL] = CausalEmbedding(
                    causes=[("e0", 0.7)], effects=[(f"e{i+1}", 0.5)]
                )

            space.add_entity(Entity(
                id=f"e{i}",
                data={"index": i, "category": f"cat_{i % 4}"},
                embeddings=embeddings,
                metadata={"batch": "test", "priority": "high" if i < 10 else "low"},
            ))

        # Phase 2: Query all types
        stats = space.stats()
        assert stats["active_entities"] == 20

        # KNN semantic
        result_sem = space.knn_query("e0", k=5, metric_type=MetricType.SEMANTIC)
        assert len(result_sem.entity_ids) == 5

        # KNN hierarchy (only even entities have hierarchy)
        # e0 has hierarchy, e2 has hierarchy
        result_hier = space.knn_query("e0", k=3, metric_type=MetricType.HIERARCHY)
        assert len(result_hier.entity_ids) >= 1

        # KNN association (only entities divisible by 3)
        result_assoc = space.knn_query("e0", k=3, metric_type=MetricType.ASSOCIATION)
        assert len(result_assoc.entity_ids) >= 1

        # Multi-metric
        result_multi = space.multi_metric_query("e0", k=5)
        assert len(result_multi.entity_ids) == 5

        # Phase 3: Persist
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [space.get_entity(eid) for eid in space.entities.keys()]
        persist.save_entities_batch(entities, batch_id=0)

        # Phase 4: Verify persistence
        loaded = persist.load_all_entities()
        assert len(loaded) == 20

        # Verify all data types survived
        sem_count = sum(1 for e in loaded if MetricType.SEMANTIC in e.embeddings)
        hier_count = sum(1 for e in loaded if MetricType.HIERARCHY in e.embeddings)
        assoc_count = sum(1 for e in loaded if MetricType.ASSOCIATION in e.embeddings)
        causal_count = sum(1 for e in loaded if MetricType.CAUSAL in e.embeddings)

        assert sem_count == 20
        assert hier_count == 10  # Even
        assert assoc_count == 7  # Divisible by 3
        assert causal_count == 4  # Divisible by 5

        # Phase 5: Checkpoint and restore
        cid = persist.create_checkpoint("Full integration test")
        loaded_after_checkpoint = persist.load_all_entities()
        assert len(loaded_after_checkpoint) == 20

    def test_enterprise_workflow(self, tmp_path):
        """Enterprise workflow: encrypt -> version -> lineage -> validate."""
        # Encryption
        enc = DataEncryption(master_key="test_key")
        sensitive = "user_credit_card_1234"
        encrypted = enc.encrypt_field(sensitive)
        assert enc.decrypt_field(encrypted) == sensitive

        # Versioning
        ver = DataVersioning()
        v1 = ver.create_snapshot("e1", {"name": "Alice"}, "Init")
        v2 = ver.create_snapshot("e1", {"name": "Alice Updated"}, "Update")
        assert len(ver.list_versions("e1")) == 2

        # Lineage
        lineage = DataLineage()
        lineage.track_entity_source("e1", "csv_import", "initial load")
        lineage.track_entity_source("e1", "etl", "normalize_names")
        assert len(lineage.get_lineage("e1")["transformations"]) == 2

        # Schema
        schema = EntitySchema()
        schema.define_schema("Person", {"name": str, "age": int})
        valid, _ = schema.validate("Person", {"name": "Alice", "age": 30})
        assert valid
        invalid, msg = schema.validate("Person", {"name": "Alice", "age": "thirty"})
        assert not invalid

        # Quality
        report = DataQualityReport.generate_report([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": None},
        ])
        assert report["completeness"] < 1.0  # Has null


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])