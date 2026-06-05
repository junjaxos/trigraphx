"""
LSH (Locality Sensitive Hashing) for sublinear nearest neighbor search.
"""

use pyo3::prelude::*;
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

#[pyclass]
pub struct LSHIndex {
    num_tables: usize,
    hash_size: usize,
    bucket_size: usize,
    tables: Arc<RwLock<Vec<HashMap<u64, Vec<usize>>>>>,  // Vec of hash tables
}

#[pymethods]
impl LSHIndex {
    #[new]
    fn new(num_tables: usize, hash_size: usize, bucket_size: usize) -> Self {
        let mut tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            tables.push(HashMap::new());
        }
        
        LSHIndex {
            num_tables,
            hash_size,
            bucket_size,
            tables: Arc::new(RwLock::new(tables)),
        }
    }
    
    fn insert_batch(&mut self, entity_ids: Vec<usize>, vectors: Vec<Vec<f32>>) -> usize {
        if entity_ids.len() != vectors.len() {
            return 0;
        }
        
        let tables_lock = self.tables.write();
        let mut tables_mut = (*tables_lock).clone();
        
        for (entity_id, vector) in entity_ids.iter().zip(vectors.iter()) {
            for table_idx in 0..self.num_tables {
                let hash_val = self.compute_hash(vector, table_idx);
                tables_mut[table_idx]
                    .entry(hash_val)
                    .or_insert_with(Vec::new)
                    .push(*entity_id);
            }
        }
        
        *tables_lock.write() = tables_mut;
        entity_ids.len()
    }
    
    fn query_knn(&self, query_vector: Vec<f32>, k: usize, num_probe: usize) -> Vec<(usize, f32)> {
        let tables_lock = self.tables.read();
        let tables = (*tables_lock).clone();
        
        let mut candidates = std::collections::BTreeSet::new();
        
        // Probe multiple tables
        for table_idx in 0..std::cmp::min(num_probe, self.num_tables) {
            let hash_val = self.compute_hash(&query_vector, table_idx);
            
            if let Some(bucket) = tables[table_idx].get(&hash_val) {
                for &entity_id in bucket.iter().take(self.bucket_size) {
                    candidates.insert(entity_id);
                }
            }
        }
        
        // Re-rank candidates by true distance
        let mut results: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|eid| (eid, 0.0f32))  // Placeholder distances
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(k).collect()
    }
    
    fn compute_hash(&self, vector: &[f32], table_idx: usize) -> u64 {
        // Simple hash: sum of vector elements (in production use proper LSH)
        let seed = (table_idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let sum: f32 = vector.iter().sum();
        ((sum.to_bits() as u64) ^ seed) % (1u64 << self.hash_size)
    }
}
