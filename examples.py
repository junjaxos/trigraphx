"""
Example usage of TriGraphX - Complete workflow demonstration.
"""

import sys
from pathlib import Path

# Add parent directory to path for local testing
sys.path.insert(0, str(Path(__file__).parent))

from trigraphx import (
    Entity, MetricType, MetricSpace, PersistenceLayer,
    SemanticEmbedding, HierarchyEmbedding, AssociationEmbedding
)
from trigraphx.enterprise import (
    RoleBasedAccessControl, Role, DataVersioning, DataEncryption,
    EntitySchema, DataQualityReport, MetricsCollector
)
import random
import tempfile


def example_1_basic_queries():
    """Example 1: Basic metric space queries."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Queries")
    print("=" * 60)
    
    # Create metric space
    space = MetricSpace(max_entities=1000)
    
    # Add documents with semantic embeddings
    for i in range(10):
        entity = Entity(
            id=f"doc_{i}",
            data={"title": f"Document {i}", "content": f"Content for doc {i}"},
            embeddings={
                MetricType.SEMANTIC: SemanticEmbedding(
                    vector=[random.random() for _ in range(50)]
                )
            }
        )
        space.add_entity(entity)
    
    # KNN query
    print("\n1. KNN Query (find 3 most similar documents):")
    result = space.knn_query("doc_0", k=3, metric_type=MetricType.SEMANTIC)
    for eid, dist, score in result.to_list():
        print(f"   {eid}: distance={dist:.4f}, score={score:.4f}")
    
    # Range query
    print("\n2. Range Query (find documents within distance 0.5):")
    result = space.range_query("doc_0", radius=0.5, metric_type=MetricType.SEMANTIC)
    print(f"   Found {len(result.entity_ids)} documents")
    for eid in result.entity_ids[:5]:
        print(f"   - {eid}")
    
    print("\n✅ Example 1 complete")


def example_2_hierarchy():
    """Example 2: Hierarchical data (organization structure)."""
    print("\n" + "=" * 60)
    print("Example 2: Hierarchical Data")
    print("=" * 60)
    
    space = MetricSpace()
    
    # Create hierarchy
    hierarchy = [
        ("company", None, "root"),
        ("dept_eng", "company", "company/engineering"),
        ("dept_sales", "company", "company/sales"),
        ("team_backend", "dept_eng", "company/engineering/backend"),
        ("team_frontend", "dept_eng", "company/engineering/frontend"),
        ("team_infra", "dept_eng", "company/engineering/infra"),
    ]
    
    for eid, parent, path in hierarchy:
        entity = Entity(
            id=eid,
            data={"name": eid},
            embeddings={
                MetricType.HIERARCHY: HierarchyEmbedding(
                    parent_id=parent,
                    level=path.count("/"),
                    path=path
                )
            }
        )
        space.add_entity(entity)
    
    # Hierarchy query
    print("\nFinding nearest teams in hierarchy from 'team_backend':")
    result = space.knn_query("team_backend", k=3, metric_type=MetricType.HIERARCHY)
    for eid, dist, score in result.to_list():
        print(f"   {eid}: hierarchy_distance={dist:.1f}")
    
    print("\n✅ Example 2 complete")


def example_3_persistence():
    """Example 3: Persistence and versioning."""
    print("\n" + "=" * 60)
    print("Example 3: Persistence & Checkpointing")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        persist = PersistenceLayer(tmp_dir)
        
        # Create and save entities
        entities = []
        for i in range(100):
            entity = Entity(
                id=f"item_{i}",
                data={"value": i, "category": "dataset_1"},
            )
            entities.append(entity)
        
        persist.save_entities_batch(entities, batch_id=0)
        print(f"1. Saved {len(entities)} entities to JSONL")
        
        # Create checkpoint
        checkpoint_id = persist.create_checkpoint("Initial dataset")
        print(f"2. Created checkpoint: {checkpoint_id}")
        
        # List checkpoints
        checkpoints = persist.list_checkpoints()
        print(f"3. Available checkpoints:")
        for cp in checkpoints:
            print(f"   - {cp['checkpoint_id']}: {cp['entity_count']} entities")
        
        # Load entities
        loaded = persist.load_all_entities()
        print(f"4. Loaded {len(loaded)} entities from persistence")
        
        # Show stats
        stats = persist.stats()
        print(f"5. Storage stats:")
        print(f"   - Entity storage: {stats['entity_storage_mb']:.2f} MB")
        print(f"   - Batch files: {stats['batch_files']}")
    
    print("\n✅ Example 3 complete")


def example_4_enterprise():
    """Example 4: Enterprise features (RBAC, Encryption, Versioning)."""
    print("\n" + "=" * 60)
    print("Example 4: Enterprise Features")
    print("=" * 60)
    
    # 1. RBAC
    print("\n1. Role-Based Access Control:")
    rbac = RoleBasedAccessControl()
    rbac.assign_role("alice", Role.ADMIN)
    rbac.assign_role("bob", Role.VIEWER)
    
    print(f"   Alice (ADMIN): can read? {rbac.has_permission('alice', 'read')}")
    print(f"   Alice (ADMIN): can write? {rbac.has_permission('alice', 'write')}")
    print(f"   Bob (VIEWER): can read? {rbac.has_permission('bob', 'read')}")
    print(f"   Bob (VIEWER): can write? {rbac.has_permission('bob', 'write')}")
    
    # 2. Encryption
    print("\n2. Data Encryption:")
    encryption = DataEncryption(master_key="secret_key_123")
    
    original = "user@example.com"
    encrypted = encryption.encrypt_field(original)
    decrypted = encryption.decrypt_field(encrypted)
    masked = encryption.mask_pii(original)
    
    print(f"   Original: {original}")
    print(f"   Encrypted: {encrypted[:20]}...")
    print(f"   Decrypted: {decrypted}")
    print(f"   Masked: {masked}")
    
    # 3. Data Versioning
    print("\n3. Data Versioning:")
    versioning = DataVersioning()
    
    v1 = versioning.create_snapshot("user_001", {"name": "John", "age": 30}, "Initial")
    v2 = versioning.create_snapshot("user_001", {"name": "John", "age": 31}, "Birthday update")
    
    versions = versioning.list_versions("user_001")
    print(f"   Created {len(versions)} versions:")
    for v in versions:
        print(f"   - {v['version_id']}: {v['message']}")
    
    # 4. Schema Validation
    print("\n4. Schema Validation:")
    schema = EntitySchema()
    schema.define_schema("User", {
        "name": str,
        "age": int,
        "email": str,
    })
    
    valid_user = {"name": "Alice", "age": 28, "email": "alice@example.com"}
    is_valid, msg = schema.validate("User", valid_user)
    print(f"   Valid user: {is_valid}")
    
    invalid_user = {"name": "Bob", "age": "thirty"}
    is_valid, msg = schema.validate("User", invalid_user)
    print(f"   Invalid user: {not is_valid} ({msg})")
    
    # 5. Data Quality
    print("\n5. Data Quality Report:")
    test_data = [
        {"name": "Alice", "email": "alice@example.com", "age": 28},
        {"name": "Bob", "email": None, "age": 35},
        {"name": "Charlie", "email": "charlie@example.com", "age": None},
    ]
    
    report = DataQualityReport.generate_report(test_data)
    print(f"   Records: {report['record_count']}")
    print(f"   Completeness: {report['completeness']:.1%}")
    print(f"   Overall Score: {report['overall_score']:.1%}")
    
    # 6. Metrics Collection
    print("\n6. Metrics Collection:")
    metrics = MetricsCollector()
    
    # Simulate query latencies
    for _ in range(10):
        latency = random.uniform(50, 150)  # 50-150ms
        metrics.record_query(latency, "query")
    
    summary = metrics.get_summary()
    percentiles = summary["percentiles"]
    print(f"   Total queries: {summary['total_queries']}")
    print(f"   Avg latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"   P50: {percentiles['p50']:.1f}ms")
    print(f"   P99: {percentiles['p99']:.1f}ms")
    
    print("\n✅ Example 4 complete")


def example_5_multi_metric():
    """Example 5: Multi-metric queries combining multiple relationships."""
    print("\n" + "=" * 60)
    print("Example 5: Multi-Metric Queries")
    print("=" * 60)
    
    space = MetricSpace()
    
    # Create entities with multiple embeddings
    for i in range(10):
        entity = Entity(
            id=f"entity_{i}",
            data={"id": i},
            embeddings={
                # Semantic (vector)
                MetricType.SEMANTIC: SemanticEmbedding(
                    vector=[random.random() for _ in range(20)]
                ),
                # Hierarchy
                MetricType.HIERARCHY: HierarchyEmbedding(
                    parent_id=f"entity_{i//2}" if i > 0 else None,
                    path=f"root/level{i%3}",
                    level=i % 3,
                ),
                # Associations (graph)
                MetricType.ASSOCIATION: AssociationEmbedding(
                    edges={f"entity_{(i+1)%10}": 0.8, f"entity_{(i-1)%10}": 0.6}
                ),
            }
        )
        space.add_entity(entity)
    
    # Multi-metric query
    print("\nCombining multiple metrics:")
    result = space.multi_metric_query(
        "entity_0",
        k=5,
        metric_weights={
            MetricType.SEMANTIC: 0.5,      # Vector similarity
            MetricType.HIERARCHY: 0.3,     # Tree structure
            MetricType.ASSOCIATION: 0.2,   # Graph connections
        }
    )
    
    print("Top 5 results (combined score):")
    for eid, dist, score in result.to_list():
        print(f"   {eid}: combined_score={score:.4f}")
    
    print("\n✅ Example 5 complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TriGraphX - Complete Workflow Examples")
    print("=" * 60)
    
    example_1_basic_queries()
    example_2_hierarchy()
    example_3_persistence()
    example_4_enterprise()
    example_5_multi_metric()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
