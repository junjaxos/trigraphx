"""
Performance benchmarks for TriGraphX.
"""

import time
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from trigraphx_core.entity import Entity, MetricType, SemanticEmbedding
from trigraphx_core.space import MetricSpace
from trigraphx_core.persistence import PersistenceLayer
from trigraphx_core.enterprise import MetricsCollector
import random


class Benchmark:
    """Benchmark suite for TriGraphX performance."""
    
    def __init__(self, output_file: str = "benchmark_results.txt"):
        self.output_file = output_file
        self.results: Dict[str, Any] = {}
    
    def log(self, message: str):
        """Log message to file and console."""
        print(message)
        with open(self.output_file, 'a') as f:
            f.write(message + '\n')
    
    def benchmark_entity_creation(self, num_entities: int = 100_000) -> Dict[str, float]:
        """Benchmark entity creation."""
        self.log(f"\n=== Entity Creation ({num_entities:,} entities) ===")
        
        start = time.time()
        
        entities = []
        for i in range(num_entities):
            entity = Entity(
                id=f"entity_{i}",
                data={"index": i, "value": random.random()},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(
                        vector=[random.random() for _ in range(100)]
                    )
                }
            )
            entities.append(entity)
        
        elapsed = time.time() - start
        rate = num_entities / elapsed
        
        self.log(f"Time: {elapsed:.2f}s")
        self.log(f"Rate: {rate:.0f} entities/sec")
        
        return {"time": elapsed, "rate": rate}
    
    def benchmark_space_insertion(self, num_entities: int = 100_000) -> Dict[str, float]:
        """Benchmark adding entities to metric space."""
        self.log(f"\n=== Space Insertion ({num_entities:,} entities) ===")
        
        space = MetricSpace(max_entities=num_entities + 1000)
        
        # Create entities first
        entities = []
        for i in range(num_entities):
            entity = Entity(
                id=f"entity_{i}",
                data={"index": i},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(
                        vector=[random.random() for _ in range(100)]
                    )
                }
            )
            entities.append(entity)
        
        start = time.time()
        
        for entity in entities:
            space.add_entity(entity)
        
        elapsed = time.time() - start
        rate = num_entities / elapsed
        
        self.log(f"Time: {elapsed:.2f}s")
        self.log(f"Rate: {rate:.0f} entities/sec")
        
        return {"time": elapsed, "rate": rate, "space": space}
    
    def benchmark_knn_query(self, space: MetricSpace, num_queries: int = 1000) -> Dict[str, Any]:
        """Benchmark KNN queries."""
        self.log(f"\n=== KNN Query ({num_queries:,} queries) ===")
        
        # Get random query entities
        entity_ids = list(space.entities.keys())[:num_queries]
        
        metrics = MetricsCollector()
        
        start = time.time()
        
        for eid in entity_ids:
            query_start = time.time()
            result = space.knn_query(eid, k=10, metric_type=MetricType.SEMANTIC)
            query_time = (time.time() - query_start) * 1000  # Convert to ms
            metrics.record_query(query_time, "knn_query")
        
        elapsed = time.time() - start
        
        summary = metrics.get_summary()
        percentiles = summary["percentiles"]
        
        self.log(f"Total time: {elapsed:.2f}s")
        self.log(f"Avg latency: {summary['avg_latency_ms']:.2f}ms")
        self.log(f"P50 latency: {percentiles.get('p50', 0):.2f}ms")
        self.log(f"P95 latency: {percentiles.get('p95', 0):.2f}ms")
        self.log(f"P99 latency: {percentiles.get('p99', 0):.2f}ms")
        self.log(f"Max latency: {percentiles.get('max', 0):.2f}ms")
        
        return {
            "time": elapsed,
            "avg_latency_ms": summary["avg_latency_ms"],
            "percentiles": percentiles,
        }
    
    def benchmark_persistence(self, entities: List[Entity], batch_size: int = 10_000) -> Dict[str, float]:
        """Benchmark persistence operations."""
        self.log(f"\n=== Persistence ({len(entities):,} entities, batch size {batch_size:,}) ===")
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            persist = PersistenceLayer(tmp_dir, batch_size=batch_size)
            
            # Save
            start = time.time()
            batch_id = 0
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                persist.save_entities_batch(batch, batch_id=batch_id)
                batch_id += 1
            save_time = time.time() - start
            
            # Load
            start = time.time()
            loaded = persist.load_all_entities()
            load_time = time.time() - start
            
            stats = persist.stats()
            
            self.log(f"Save time: {save_time:.2f}s ({len(entities)/save_time:.0f} entities/sec)")
            self.log(f"Load time: {load_time:.2f}s ({len(loaded)/load_time:.0f} entities/sec)")
            self.log(f"Storage: {stats['entity_storage_mb']:.2f} MB")
            
            return {
                "save_time": save_time,
                "load_time": load_time,
                "storage_mb": stats["entity_storage_mb"],
            }
    
    def benchmark_range_query(self, space: MetricSpace, num_queries: int = 100) -> Dict[str, Any]:
        """Benchmark range queries."""
        self.log(f"\n=== Range Query ({num_queries:,} queries) ===")
        
        entity_ids = list(space.entities.keys())[:num_queries]
        
        metrics = MetricsCollector()
        
        start = time.time()
        
        for eid in entity_ids:
            query_start = time.time()
            result = space.range_query(eid, radius=0.5, metric_type=MetricType.SEMANTIC)
            query_time = (time.time() - query_start) * 1000
            metrics.record_query(query_time, "range_query")
        
        elapsed = time.time() - start
        
        summary = metrics.get_summary()
        percentiles = summary["percentiles"]
        
        self.log(f"Total time: {elapsed:.2f}s")
        self.log(f"Avg latency: {summary['avg_latency_ms']:.2f}ms")
        self.log(f"P99 latency: {percentiles.get('p99', 0):.2f}ms")
        
        return {
            "time": elapsed,
            "avg_latency_ms": summary["avg_latency_ms"],
            "percentiles": percentiles,
        }
    
    def benchmark_multi_metric_query(self, space: MetricSpace, num_queries: int = 100) -> Dict[str, Any]:
        """Benchmark multi-metric queries."""
        self.log(f"\n=== Multi-Metric Query ({num_queries:,} queries) ===")
        
        entity_ids = list(space.entities.keys())[:num_queries]
        
        metrics = MetricsCollector()
        
        start = time.time()
        
        for eid in entity_ids:
            query_start = time.time()
            result = space.multi_metric_query(eid, k=10)
            query_time = (time.time() - query_start) * 1000
            metrics.record_query(query_time, "multi_query")
        
        elapsed = time.time() - start
        
        summary = metrics.get_summary()
        percentiles = summary["percentiles"]
        
        self.log(f"Total time: {elapsed:.2f}s")
        self.log(f"Avg latency: {summary['avg_latency_ms']:.2f}ms")
        self.log(f"P99 latency: {percentiles.get('p99', 0):.2f}ms")
        
        return {
            "time": elapsed,
            "avg_latency_ms": summary["avg_latency_ms"],
            "percentiles": percentiles,
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        # Clear output file
        open(self.output_file, 'w').close()
        
        self.log("=" * 60)
        self.log("TriGraphX Performance Benchmark")
        self.log("=" * 60)
        
        # Small scale tests
        self.log("\n### PHASE 1: Small Scale (10K entities) ###")
        self.benchmark_entity_creation(10_000)
        result_insert = self.benchmark_space_insertion(10_000)
        self.benchmark_knn_query(result_insert["space"], num_queries=100)
        
        # Medium scale tests
        self.log("\n### PHASE 2: Medium Scale (100K entities) ###")
        result_insert = self.benchmark_space_insertion(100_000)
        self.benchmark_knn_query(result_insert["space"], num_queries=500)
        self.benchmark_range_query(result_insert["space"], num_queries=100)
        
        # Persistence tests
        self.log("\n### PHASE 3: Persistence ###")
        entities = []
        for i in range(100_000):
            entity = Entity(
                id=f"persist_{i}",
                data={"index": i},
                embeddings={
                    MetricType.SEMANTIC: SemanticEmbedding(
                        vector=[random.random() for _ in range(100)]
                    )
                }
            )
            entities.append(entity)
        
        self.benchmark_persistence(entities, batch_size=10_000)
        
        self.log("\n" + "=" * 60)
        self.log("Benchmark Complete")
        self.log("=" * 60)


def main():
    """Run benchmarks."""
    benchmark = Benchmark("benchmark_results.txt")
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
