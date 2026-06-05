"""
Comprehensive test suite for TriGraphX.
"""

import pytest
import sys
import os
from datetime import datetime
from pathlib import Path

# Add trigraphx_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trigraphx_core.entity import (
    Entity, MetricType, HierarchyEmbedding, SemanticEmbedding,
    AssociationEmbedding, CausalEmbedding,
    HierarchyDistance, SemanticDistance, AssociationDistance, CausalDistance
)
from trigraphx_core.space import MetricSpace, QueryResult
from trigraphx_core.persistence import PersistenceLayer
from trigraphx_core.enterprise import (
    RoleBasedAccessControl, Role, DataEncryption, DataVersioning,
    DataLineage, EntitySchema, DataQualityReport, MetricsCollector,
    AlertingSystem
)


class TestEntity:
    """Test Entity and Embeddings."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            id="test_1",
            data={"name": "Test Entity", "value": 42},
        )
        assert entity.id == "test_1"
        assert entity.data["name"] == "Test Entity"
        assert not entity.deleted
    
    def test_entity_serialization(self):
        """Test entity to/from dict conversion."""
        entity = Entity(
            id="test_1",
            data={"name": "Test"},
        )
        
        entity_dict = entity.to_dict()
        restored = Entity.from_dict(entity_dict)
        
        assert restored.id == entity.id
        assert restored.data == entity.data
    
    def test_hierarchy_embedding(self):
        """Test hierarchy embeddings."""
        emb = HierarchyEmbedding(
            parent_id="root",
            children_ids=["child1", "child2"],
            level=1,
            path="root/branch",
        )
        
        d = emb.to_dict()
        restored = HierarchyEmbedding.from_dict(d)
        assert restored.level == 1
        assert restored.path == "root/branch"
    
    def test_semantic_embedding(self):
        """Test semantic embeddings."""
        emb = SemanticEmbedding(
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        assert emb.dimension == 4
        
        d = emb.to_dict()
        restored = SemanticEmbedding.from_dict(d)
        assert len(restored.vector) == 4


class TestMetrics:
    """Test distance metrics."""
    
    def test_hierarchy_distance(self):
        """Test hierarchy distance computation."""
        metric = HierarchyDistance()
        
        emb1 = HierarchyEmbedding(path="root/a/b")
        emb2 = HierarchyEmbedding(path="root/a/c")
        
        dist = metric.compute(emb1, emb2)
        assert dist == 2.0  # Both at distance 1 from 'a'
    
    def test_semantic_distance(self):
        """Test semantic distance computation."""
        metric = SemanticDistance(use_cosine=True)
        
        emb1 = SemanticEmbedding(vector=[1.0, 0.0])
        emb2 = SemanticEmbedding(vector=[0.0, 1.0])
        
        dist = metric.compute(emb1, emb2)
        assert abs(dist - 1.0) < 0.01  # Orthogonal vectors
    
    def test_semantic_distance_batch(self):
        """Test batch distance computation."""
        metric = SemanticDistance()
        
        emb1 = SemanticEmbedding(vector=[1.0, 0.0, 0.0])
        embeddings2 = [
            SemanticEmbedding(vector=[1.0, 0.0, 0.0]),
            SemanticEmbedding(vector=[0.0, 1.0, 0.0]),
        ]
        
        distances = metric.compute_batch(emb1, embeddings2)
        assert len(distances) == 2
        assert distances[0] < distances[1]  # First is closer
    
    def test_association_distance(self):
        """Test association distance."""
        metric = AssociationDistance()
        
        emb1 = AssociationEmbedding(edges={"entity2": 0.8})
        emb2 = AssociationEmbedding()
        
        dist = metric.compute(emb1, emb2)
        assert dist > 1.0


class TestMetricSpace:
    """Test MetricSpace core functionality."""
    
    def test_space_creation(self):
        """Test metric space initialization."""
        space = MetricSpace(max_entities=1000)
        assert len(space.entities) == 0
        stats = space.stats()
        assert stats["total_entities"] == 0
    
    def test_add_entity(self):
        """Test adding entities."""
        space = MetricSpace()
        
        entity = Entity(
            id="e1",
            data={"value": 1},
            embeddings={
                MetricType.SEMANTIC: {
                    "vector": [0.1, 0.2],
                    "dimension": 2,
                }
            }
        )
        
        space.add_entity(entity)
        assert "e1" in space.entities
        stats = space.stats()
        assert stats["active_entities"] == 1
    
    def test_soft_delete(self):
        """Test soft delete."""
        space = MetricSpace()
        
        entity = Entity(id="e1", data={})
        space.add_entity(entity)
        
        space.soft_delete_entity("e1")
        
        assert space.entities["e1"].deleted
        assert space.get_entity("e1") is None  # Should not return deleted entities
    
    def test_knn_query(self):
        """Test KNN query."""
        space = MetricSpace()
        
        # Add entities with semantic embeddings
        entities = []
        for i in range(5):
            entity = Entity(
                id=f"e{i}",
                data={"index": i},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(
                        vector=[float(i), float(i)]
                    )
                }
            )
            space.add_entity(entity)
            entities.append(entity)
        
        # Query
        result = space.knn_query("e0", k=2, metric_type=MetricType.SEMANTIC)
        
        assert len(result.entity_ids) == 2
        assert all(0 <= s <= 1 for s in result.scores)
    
    def test_range_query(self):
        """Test range query."""
        space = MetricSpace()
        
        # Add entities
        for i in range(5):
            entity = Entity(
                id=f"e{i}",
                data={},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(
                        vector=[float(i * 10), 0.0]
                    )
                }
            )
            space.add_entity(entity)
        
        # Range query from e0
        result = space.range_query("e0", radius=15.0, metric_type=MetricType.SEMANTIC)
        
        assert len(result.entity_ids) >= 1  # At least e0's neighbors
    
    def test_multi_metric_query(self):
        """Test multi-metric query."""
        space = MetricSpace()
        
        # Add entities with multiple embeddings
        for i in range(3):
            entity = Entity(
                id=f"e{i}",
                data={},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(vector=[float(i), 0.0]),
                    MetricType.HIERARCHY: HierarchyEmbedding(
                        path=f"root/item{i}",
                        level=i
                    ),
                }
            )
            space.add_entity(entity)
        
        # Multi-metric query
        result = space.multi_metric_query(
            "e0",
            k=2,
            metric_weights={
                MetricType.SEMANTIC: 0.6,
                MetricType.HIERARCHY: 0.4,
            }
        )
        
        assert len(result.entity_ids) <= 2


class TestPersistence:
    """Test persistence layer."""
    
    def test_persistence_initialization(self, tmp_path):
        """Test persistence layer creation."""
        persist = PersistenceLayer(str(tmp_path / "db"))
        
        assert persist.entities_dir.exists()
        assert persist.index_dir.exists()
        assert persist.checkpoints_dir.exists()
    
    def test_save_and_load_entities(self, tmp_path):
        """Test entity persistence."""
        persist = PersistenceLayer(str(tmp_path / "db"))
        
        # Create and save entities
        entities = []
        for i in range(5):
            entity = Entity(
                id=f"e{i}",
                data={"index": i},
            )
            entities.append(entity)
        
        persist.save_entities_batch(entities, batch_id=0)
        
        # Load back
        loaded = persist.load_all_entities()
        assert len(loaded) == 5
        assert loaded[0].id in ["e0", "e1", "e2", "e3", "e4"]
    
    def test_checkpoints(self, tmp_path):
        """Test checkpointing."""
        persist = PersistenceLayer(str(tmp_path / "db"))
        
        # Create entities
        entities = [Entity(id=f"e{i}", data={}) for i in range(3)]
        persist.save_entities_batch(entities, batch_id=0)
        
        # Create checkpoint
        checkpoint_id = persist.create_checkpoint("Test checkpoint")
        assert checkpoint_id is not None
        
        # List checkpoints
        checkpoints = persist.list_checkpoints()
        assert len(checkpoints) >= 1


class TestEnterprise:
    """Test enterprise features."""
    
    def test_rbac(self):
        """Test role-based access control."""
        rbac = RoleBasedAccessControl()
        
        rbac.assign_role("user1", Role.EDITOR)
        assert rbac.has_permission("user1", "read")
        assert rbac.has_permission("user1", "write")
        assert not rbac.has_permission("user1", "admin")
    
    def test_encryption(self):
        """Test data encryption."""
        encryption = DataEncryption(master_key="secret_key")
        
        original = "sensitive_data"
        encrypted = encryption.encrypt_field(original)
        decrypted = encryption.decrypt_field(encrypted)
        
        assert encrypted != original
        assert decrypted == original
    
    def test_versioning(self):
        """Test data versioning."""
        versioning = DataVersioning()
        
        # Create snapshots
        v1 = versioning.create_snapshot("e1", {"name": "v1"}, "Initial")
        v2 = versioning.create_snapshot("e1", {"name": "v2"}, "Updated")
        
        # List versions
        versions = versioning.list_versions("e1")
        assert len(versions) == 2
        
        # Rollback
        rollback_data = versioning.rollback("e1", v1)
        assert rollback_data["name"] == "v1"
    
    def test_data_lineage(self):
        """Test data lineage tracking."""
        lineage = DataLineage()
        
        lineage.track_entity_source("e1", "csv_import")
        lineage.track_entity_source("e1", "transformation", "normalization")
        
        source = lineage.get_lineage("e1")
        assert source["source"] == "csv_import"
        assert len(source["transformations"]) >= 1
    
    def test_schema_validation(self):
        """Test schema validation."""
        schema = EntitySchema()
        
        schema.define_schema("User", {
            "name": str,
            "age": int,
            "email": str,
        })
        
        valid_data = {"name": "John", "age": 30, "email": "john@example.com"}
        is_valid, msg = schema.validate("User", valid_data)
        assert is_valid
    
    def test_data_quality(self):
        """Test data quality reporting."""
        entities = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
            {"name": "Bob", "age": None},
        ]
        
        report = DataQualityReport.generate_report(entities)
        
        assert "completeness" in report
        assert "overall_score" in report
        assert report["record_count"] == 3
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        collector.record_query(50.0, "query")
        collector.record_query(100.0, "query")
        collector.record_query(75.0, "query")
        
        summary = collector.get_summary()
        assert summary["total_queries"] == 3
        
        percentiles = summary["percentiles"]
        assert "p50" in percentiles
        assert "p99" in percentiles
    
    def test_alerting(self):
        """Test alert system."""
        alerting = AlertingSystem()
        
        metrics = {
            "p99_latency_ms": 150,  # Exceeds threshold of 100ms
            "error_rate": 0.001,
        }
        
        alerts = alerting.check_metrics(metrics)
        assert len(alerts) >= 1


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow: add -> query -> persist -> restore."""
        # Create metric space
        space = MetricSpace(max_entities=1000)
        
        # Add entities
        for i in range(10):
            entity = Entity(
                id=f"item_{i}",
                data={"value": i},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(
                        vector=[float(i), float(i**2) * 0.1]
                    )
                }
            )
            space.add_entity(entity)
        
        # Perform query
        result = space.knn_query("item_0", k=3)
        assert len(result.entity_ids) == 3
        
        # Persist
        persist = PersistenceLayer(str(tmp_path / "db"))
        entities = [space.get_entity(eid) for eid in space.entities.keys()]
        persist.save_entities_batch(entities, batch_id=0)
        
        # Checkpoint
        checkpoint_id = persist.create_checkpoint("After initial load")
        assert checkpoint_id
        
        # Verify persistence
        loaded = persist.load_all_entities()
        assert len(loaded) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
