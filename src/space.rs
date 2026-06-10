use std::collections::{HashMap, HashSet};

use crate::entity::*;
use crate::error::{Result, TriGraphXError};

/// MetricSpace: unified storage combining tree, graph, and vector
pub struct MetricSpace {
    max_entities: usize,
    entities: HashMap<String, Entity>,
    metrics: HashMap<MetricType, Box<dyn DistanceMetric + Send + Sync>>,
    hierarchy_index: HashMap<String, HashSet<String>>,
    embedding_index: HashMap<MetricType, HashMap<String, EmbeddingData>>,
}

impl MetricSpace {
    pub fn new(max_entities: usize) -> Self {
        let mut metrics: HashMap<MetricType, Box<dyn DistanceMetric + Send + Sync>> = HashMap::new();
        metrics.insert(MetricType::Hierarchy, Box::new(HierarchyDistance));
        metrics.insert(MetricType::Semantic, Box::new(SemanticDistance));
        metrics.insert(MetricType::Association, Box::new(AssociationDistance));
        metrics.insert(MetricType::Causal, Box::new(CausalDistance));

        let mut embedding_index = HashMap::new();
        embedding_index.insert(MetricType::Hierarchy, HashMap::new());
        embedding_index.insert(MetricType::Semantic, HashMap::new());
        embedding_index.insert(MetricType::Association, HashMap::new());
        embedding_index.insert(MetricType::Causal, HashMap::new());

        Self {
            max_entities,
            entities: HashMap::new(),
            metrics,
            hierarchy_index: HashMap::new(),
            embedding_index,
        }
    }

    /// Ingest data and create entity
    pub fn ingest(
        &mut self,
        data: HashMap<String, serde_json::Value>,
        embeddings: Option<HashMap<MetricType, EmbeddingData>>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        dedup_keys: Option<&[String]>,
    ) -> (Entity, bool) {
        let entity = Entity::new(data, embeddings.unwrap_or_default(), metadata.unwrap_or_default(), dedup_keys);
        self.upsert_entity(entity, true)
    }

    /// Insert or update an entity
    pub fn upsert_entity(&mut self, entity: Entity, merge_data: bool) -> (Entity, bool) {
        if self.entities.len() >= self.max_entities && !self.entities.contains_key(&entity.id) {
            return (entity, false);
        }

        if let Some(existing) = self.entities.get_mut(&entity.id) {
            if merge_data {
                existing.merge(&entity);
            } else {
                existing.data = entity.data.clone();
                existing.embeddings = entity.embeddings.clone();
                existing.metadata = entity.metadata.clone();
            }
            let result = existing.clone();
            Self::update_indices(&result.id, &result, &mut self.hierarchy_index, &mut self.embedding_index);
            (result, false)
        } else {
            let id = entity.id.clone();
            Self::update_indices(&id, &entity, &mut self.hierarchy_index, &mut self.embedding_index);
            self.entities.insert(id.clone(), entity.clone());
            (entity, true)
        }
    }

    /// Get entity by ID
    pub fn get_entity(&self, entity_id: &str) -> Option<&Entity> {
        self.entities.get(entity_id)
    }

    /// Get all entities
    pub fn get_all_entities(&self) -> &HashMap<String, Entity> {
        &self.entities
    }

    /// Update entity
    pub fn update_entity(&mut self, entity_id: &str, updates: EntityUpdate) -> Result<()> {
        if !self.entities.contains_key(entity_id) {
            return Err(TriGraphXError::EntityNotFound(entity_id.to_string()));
        }

        {
            let entity = self.entities.get_mut(entity_id).unwrap();
            if entity.deleted {
                return Err(TriGraphXError::EntityDeleted(entity_id.to_string()));
            }

            if let Some(data) = updates.data {
                for (k, v) in data {
                    entity.data.insert(k, v);
                }
            }

            if let Some(embeddings) = updates.embeddings {
                for (k, v) in embeddings {
                    entity.embeddings.insert(k, v);
                }
            }

            if let Some(metadata) = updates.metadata {
                for (k, v) in metadata {
                    entity.metadata.insert(k, v);
                }
            }

            entity.updated_at = chrono::Utc::now();
        }
        let entity = self.entities.get(entity_id).unwrap();
        Self::update_indices(entity_id, entity, &mut self.hierarchy_index, &mut self.embedding_index);
        Ok(())
    }

    /// Soft delete
    pub fn soft_delete_entity(&mut self, entity_id: &str) -> Result<()> {
        if let Some(entity) = self.entities.get_mut(entity_id) {
            entity.deleted = true;
            entity.updated_at = chrono::Utc::now();
        }
        Ok(())
    }

    /// Hard delete
    pub fn hard_delete_entity(&mut self, entity_id: &str) -> Result<()> {
        if let Some(_entity) = self.entities.remove(entity_id) {
            self.remove_from_indices(entity_id);
        }
        Ok(())
    }

    /// KNN query
    pub fn knn_query(
        &self,
        query_entity_id: &str,
        k: usize,
        metric_type: MetricType,
    ) -> Result<QueryResult> {
        let query_entity = self.entities.get(query_entity_id)
            .ok_or_else(|| TriGraphXError::EntityNotFound(query_entity_id.to_string()))?;

        let query_emb = query_entity.embeddings.get(&metric_type)
            .ok_or_else(|| TriGraphXError::InvalidEmbedding(format!("No {} embedding for {}", metric_type_str(metric_type), query_entity_id)))?;

        let metric = self.metrics.get(&metric_type)
            .ok_or_else(|| TriGraphXError::QueryError(format!("Unknown metric: {}", metric_type_str(metric_type))))?;

        let mut distances: Vec<(String, f64)> = self.entities.iter()
            .filter(|(id, entity)| {
                *id != query_entity_id && !entity.deleted && entity.embeddings.contains_key(&metric_type)
            })
            .map(|(id, entity)| {
                let emb = entity.embeddings.get(&metric_type).unwrap();
                let dist = metric.compute(query_emb, emb);
                (id.clone(), dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = k.min(distances.len());
        let results: Vec<(String, f64)> = distances.into_iter().take(k).collect();

        let max_dist = results.iter().map(|(_, d)| *d).fold(0.0f64, f64::max);

        let entity_ids: Vec<String> = results.iter().map(|(id, _)| id.clone()).collect();
        let scores: Vec<f64> = results.iter().map(|(_, d)| {
            if max_dist > 0.0 { *d / max_dist } else { 0.0 }
        }).collect();

        Ok(QueryResult { entity_ids, scores })
    }

    /// Range query
    pub fn range_query(
        &self,
        query_entity_id: &str,
        threshold: f64,
        metric_type: MetricType,
    ) -> Result<QueryResult> {
        let query_entity = self.entities.get(query_entity_id)
            .ok_or_else(|| TriGraphXError::EntityNotFound(query_entity_id.to_string()))?;

        let query_emb = query_entity.embeddings.get(&metric_type)
            .ok_or_else(|| TriGraphXError::InvalidEmbedding(format!("No {} embedding for {}", metric_type_str(metric_type), query_entity_id)))?;

        let metric = self.metrics.get(&metric_type)
            .ok_or_else(|| TriGraphXError::QueryError(format!("Unknown metric: {}", metric_type_str(metric_type))))?;

        let mut results: Vec<(String, f64)> = self.entities.iter()
            .filter(|(id, entity)| {
                *id != query_entity_id && !entity.deleted && entity.embeddings.contains_key(&metric_type)
            })
            .filter_map(|(id, entity)| {
                let emb = entity.embeddings.get(&metric_type).unwrap();
                let dist = metric.compute(query_emb, emb);
                if dist <= threshold {
                    Some((id.clone(), dist))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let entity_ids: Vec<String> = results.iter().map(|(id, _)| id.clone()).collect();
        let scores: Vec<f64> = results.iter().map(|(_, d)| *d).collect();

        Ok(QueryResult { entity_ids, scores })
    }

    /// Multi-metric query
    pub fn multi_metric_query(
        &self,
        query_entity_id: &str,
        k: usize,
        metric_weights: HashMap<MetricType, f64>,
    ) -> Result<QueryResult> {
        let query_entity = self.entities.get(query_entity_id)
            .ok_or_else(|| TriGraphXError::EntityNotFound(query_entity_id.to_string()))?;

        let mut combined_distances: HashMap<String, f64> = HashMap::new();

        for (metric_type, weight) in &metric_weights {
            let query_emb = match query_entity.embeddings.get(metric_type) {
                Some(emb) => emb,
                None => continue,
            };

            let metric = match self.metrics.get(metric_type) {
                Some(m) => m,
                None => continue,
            };

            for (id, entity) in &self.entities {
                if id == query_entity_id || entity.deleted {
                    continue;
                }
                if let Some(emb) = entity.embeddings.get(metric_type) {
                    let dist = metric.compute(query_emb, emb);
                    *combined_distances.entry(id.clone()).or_insert(0.0) += dist * weight;
                }
            }
        }

        let mut results: Vec<(String, f64)> = combined_distances.into_iter().collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = k.min(results.len());
        let results: Vec<(String, f64)> = results.into_iter().take(k).collect();

        let max_dist = results.iter().map(|(_, d)| *d).fold(0.0f64, f64::max);

        let entity_ids: Vec<String> = results.iter().map(|(id, _)| id.clone()).collect();
        let scores: Vec<f64> = results.iter().map(|(_, d)| {
            if max_dist > 0.0 { *d / max_dist } else { 0.0 }
        }).collect();

        Ok(QueryResult { entity_ids, scores })
    }

    /// Get statistics
    pub fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_entities".to_string(), self.entities.len());
        stats.insert("active_entities".to_string(), self.entities.values().filter(|e| !e.deleted).count());
        stats.insert("deleted_entities".to_string(), self.entities.values().filter(|e| e.deleted).count());
        stats.insert("max_entities".to_string(), self.max_entities);
        stats
    }

    /// Update indices
    fn update_indices(
        entity_id: &str,
        entity: &Entity,
        hierarchy_index: &mut HashMap<String, HashSet<String>>,
        embedding_index: &mut HashMap<MetricType, HashMap<String, EmbeddingData>>,
    ) {
        for index in embedding_index.values_mut() {
            index.remove(entity_id);
        }

        for (mt, emb) in &entity.embeddings {
            if let Some(index) = embedding_index.get_mut(mt) {
                index.insert(entity_id.to_string(), emb.clone());
            }

            if let MetricType::Hierarchy = mt {
                if let EmbeddingData::Hierarchy(h) = emb {
                    if let Some(parent_id) = &h.parent_id {
                        hierarchy_index
                            .entry(parent_id.clone())
                            .or_insert_with(HashSet::new)
                            .insert(entity_id.to_string());
                    }
                }
            }
        }
    }

    /// Remove from indices
    fn remove_from_indices(&mut self, entity_id: &str) {
        for index in self.embedding_index.values_mut() {
            index.remove(entity_id);
        }
        self.hierarchy_index.retain(|_, children| {
            children.remove(entity_id);
            !children.is_empty()
        });
    }
}

fn metric_type_str(mt: MetricType) -> &'static str {
    match mt {
        MetricType::Hierarchy => "Hierarchy",
        MetricType::Semantic => "Semantic",
        MetricType::Association => "Association",
        MetricType::Causal => "Causal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(name: &str) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("name".to_string(), serde_json::json!(name));
        data
    }

    #[test]
    fn test_ingest_and_query() {
        let mut space = MetricSpace::new(100);

        let (entity, created) = space.ingest(make_data("test"), None, None, None);
        assert!(created);
        assert!(entity.id.starts_with("test_"));

        let (entity2, created2) = space.ingest(make_data("test"), None, None, None);
        assert!(!created2);
        assert_eq!(entity.id, entity2.id);
    }

    #[test]
    fn test_knn_query() {
        let mut space = MetricSpace::new(100);

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.9, 0.1, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        let mut emb1 = HashMap::new();
        emb1.insert(MetricType::Semantic, EmbeddingData::Semantic(SemanticEmbedding::new(v1)));
        let (a, _) = space.ingest(make_data("A"), Some(emb1), None, None);

        let mut emb2 = HashMap::new();
        emb2.insert(MetricType::Semantic, EmbeddingData::Semantic(SemanticEmbedding::new(v2)));
        let (b, _) = space.ingest(make_data("B"), Some(emb2), None, None);

        let mut emb3 = HashMap::new();
        emb3.insert(MetricType::Semantic, EmbeddingData::Semantic(SemanticEmbedding::new(v3)));
        space.ingest(make_data("C"), Some(emb3), None, None);

        let result = space.knn_query(&a.id, 2, MetricType::Semantic).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.entity_ids[0], b.id);
    }

    #[test]
    fn test_hierarchy_distance() {
        let dist = HierarchyDistance;
        let root = HierarchyEmbedding::new(None, vec!["a".to_string()], 0, "root".to_string());
        let a = HierarchyEmbedding::new(Some("root".to_string()), vec![], 1, "root/a".to_string());
        let b = HierarchyEmbedding::new(Some("root".to_string()), vec![], 1, "root/b".to_string());

        let d1 = dist.compute(&EmbeddingData::Hierarchy(root.clone()), &EmbeddingData::Hierarchy(a.clone()));
        assert_eq!(d1, 1.0);

        let d2 = dist.compute(&EmbeddingData::Hierarchy(a), &EmbeddingData::Hierarchy(b));
        assert_eq!(d2, 2.0);
    }

    #[test]
    fn test_semantic_distance() {
        let dist = SemanticDistance;
        let v1 = SemanticEmbedding::new(vec![1.0, 0.0, 0.0]);
        let v2 = SemanticEmbedding::new(vec![1.0, 0.0, 0.0]);
        let v3 = SemanticEmbedding::new(vec![0.0, 1.0, 0.0]);

        let d1 = dist.compute(&EmbeddingData::Semantic(v1.clone()), &EmbeddingData::Semantic(v2));
        assert!((d1 - 0.0).abs() < 1e-10);

        let d2 = dist.compute(&EmbeddingData::Semantic(v1), &EmbeddingData::Semantic(v3));
        assert!((d2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_soft_delete() {
        let mut space = MetricSpace::new(100);
        let (entity, _) = space.ingest(make_data("test"), None, None, None);

        space.soft_delete_entity(&entity.id).unwrap();
        let e = space.get_entity(&entity.id).unwrap();
        assert!(e.deleted);
    }

    #[test]
    fn test_stats() {
        let mut space = MetricSpace::new(100);
        space.ingest(make_data("a"), None, None, None);
        space.ingest(make_data("b"), None, None, None);

        let stats = space.stats();
        assert_eq!(*stats.get("total_entities").unwrap(), 2);
        assert_eq!(*stats.get("active_entities").unwrap(), 2);
    }

    #[test]
    fn test_upsert_merge() {
        let mut space = MetricSpace::new(100);
        let mut data1 = make_data("test");
        data1.insert("age".to_string(), serde_json::json!(25));
        let (e1, _) = space.ingest(data1, None, None, None);

        let mut data2 = make_data("test");
        data2.insert("company".to_string(), serde_json::json!("ACME"));
        let (e2, _) = space.ingest(data2, None, None, None);

        assert_eq!(e1.id, e2.id);
        assert!(e2.data.contains_key("age"));
        assert!(e2.data.contains_key("company"));
    }
}
