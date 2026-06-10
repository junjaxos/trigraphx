use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use chrono::Utc;
use std::collections::HashMap;

/// Metric types for different distance computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    Hierarchy,
    Semantic,
    Association,
    Causal,
}

/// Hierarchy embedding for tree-structured data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyEmbedding {
    pub parent_id: Option<String>,
    pub children_ids: Vec<String>,
    pub level: i32,
    pub path: String,
}

impl HierarchyEmbedding {
    pub fn new(parent_id: Option<String>, children_ids: Vec<String>, level: i32, path: String) -> Self {
        Self { parent_id, children_ids, level, path }
    }
}

/// Semantic embedding for vector similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEmbedding {
    pub vector: Vec<f64>,
}

impl SemanticEmbedding {
    pub fn new(vector: Vec<f64>) -> Self {
        Self { vector }
    }
}

/// Association embedding for graph relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationEmbedding {
    pub edges: HashMap<String, f64>,
    pub bidirectional: bool,
    pub relationship_type: String,
}

impl AssociationEmbedding {
    pub fn new(edges: HashMap<String, f64>, bidirectional: bool, relationship_type: String) -> Self {
        Self { edges, bidirectional, relationship_type }
    }
}

/// Causal embedding for cause-effect relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEmbedding {
    pub causes: Vec<String>,
    pub effects: Vec<String>,
    pub strength: f64,
}

impl CausalEmbedding {
    pub fn new(causes: Vec<String>, effects: Vec<String>, strength: f64) -> Self {
        Self { causes, effects, strength }
    }
}

/// Embedding data wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum EmbeddingData {
    Hierarchy(HierarchyEmbedding),
    Semantic(SemanticEmbedding),
    Association(AssociationEmbedding),
    Causal(CausalEmbedding),
}

/// Entity in the metric space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub data: HashMap<String, serde_json::Value>,
    pub embeddings: HashMap<MetricType, EmbeddingData>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: chrono::DateTime<Utc>,
    pub updated_at: chrono::DateTime<Utc>,
    pub deleted: bool,
}

impl Entity {
    pub fn new(
        data: HashMap<String, serde_json::Value>,
        embeddings: HashMap<MetricType, EmbeddingData>,
        metadata: HashMap<String, serde_json::Value>,
        dedup_keys: Option<&[String]>,
    ) -> Self {
        let id = Self::compute_id(&data, dedup_keys);
        let now = Utc::now();
        Self {
            id,
            data,
            embeddings,
            metadata,
            created_at: now,
            updated_at: now,
            deleted: false,
        }
    }

    pub fn compute_id(data: &HashMap<String, serde_json::Value>, dedup_keys: Option<&[String]>) -> String {
        let default_keys = vec!["name".to_string()];
        let keys = dedup_keys.unwrap_or(&default_keys);

        let mut hasher = Sha256::new();
        for key in keys {
            if let Some(value) = data.get(key) {
                hasher.update(value.to_string().as_bytes());
            }
        }
        let hash = hex::encode(&hasher.finalize()[..8]);

        let name = data.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("entity");

        format!("{}_{}", name, hash)
    }

    pub fn merge(&mut self, other: &Entity) {
        for (k, v) in &other.data {
            self.data.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.embeddings {
            self.embeddings.entry(*k).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.metadata {
            self.metadata.entry(k.clone()).or_insert_with(|| v.clone());
        }
        self.updated_at = Utc::now();
    }
}

/// Distance metric trait
pub trait DistanceMetric {
    fn metric_type(&self) -> MetricType;
    fn compute(&self, emb1: &EmbeddingData, emb2: &EmbeddingData) -> f64;
}

/// Hierarchy distance: tree path distance
pub struct HierarchyDistance;

impl DistanceMetric for HierarchyDistance {
    fn metric_type(&self) -> MetricType {
        MetricType::Hierarchy
    }

    fn compute(&self, emb1: &EmbeddingData, emb2: &EmbeddingData) -> f64 {
        let (h1, h2) = match (emb1, emb2) {
            (EmbeddingData::Hierarchy(a), EmbeddingData::Hierarchy(b)) => (a, b),
            _ => return f64::INFINITY,
        };

        let path1: Vec<&str> = h1.path.split('/').collect();
        let path2: Vec<&str> = h2.path.split('/').collect();

        let mut lca_depth = 0;
        for i in 0..path1.len().min(path2.len()) {
            if path1[i] == path2[i] {
                lca_depth = i + 1;
            } else {
                break;
            }
        }

        let depth1 = path1.len().saturating_sub(lca_depth);
        let depth2 = path2.len().saturating_sub(lca_depth);

        (depth1 + depth2) as f64
    }
}

/// Semantic distance: cosine distance
pub struct SemanticDistance;

impl DistanceMetric for SemanticDistance {
    fn metric_type(&self) -> MetricType {
        MetricType::Semantic
    }

    fn compute(&self, emb1: &EmbeddingData, emb2: &EmbeddingData) -> f64 {
        let (s1, s2) = match (emb1, emb2) {
            (EmbeddingData::Semantic(a), EmbeddingData::Semantic(b)) => (a, b),
            _ => return f64::INFINITY,
        };

        let dot: f64 = s1.vector.iter().zip(s2.vector.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = s1.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = s2.vector.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 1.0;
        }

        1.0 - dot / (norm1 * norm2)
    }
}

/// Association distance: graph edge weight
pub struct AssociationDistance;

impl DistanceMetric for AssociationDistance {
    fn metric_type(&self) -> MetricType {
        MetricType::Association
    }

    fn compute(&self, emb1: &EmbeddingData, emb2: &EmbeddingData) -> f64 {
        let (a1, a2) = match (emb1, emb2) {
            (EmbeddingData::Association(a), EmbeddingData::Association(b)) => (a, b),
            _ => return f64::INFINITY,
        };

        // Check if entities have direct edges
        if let Some(w) = a1.edges.get(&a2.relationship_type) {
            return 1.0 - w;
        }
        if a1.bidirectional {
            if let Some(w) = a2.edges.get(&a1.relationship_type) {
                return 1.0 - w;
            }
        }

        1.0 // No direct connection
    }
}

/// Causal distance: cause-effect strength
pub struct CausalDistance;

impl DistanceMetric for CausalDistance {
    fn metric_type(&self) -> MetricType {
        MetricType::Causal
    }

    fn compute(&self, emb1: &EmbeddingData, emb2: &EmbeddingData) -> f64 {
        let (c1, c2) = match (emb1, emb2) {
            (EmbeddingData::Causal(a), EmbeddingData::Causal(b)) => (a, b),
            _ => return f64::INFINITY,
        };

        // Check if emb1 causes emb2 or vice versa
        let has_causal = c1.effects.iter().any(|e| c2.causes.contains(e))
            || c2.effects.iter().any(|e| c1.causes.contains(e));

        if has_causal {
            1.0 - (c1.strength + c2.strength) / 2.0
        } else {
            1.0
        }
    }
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub entity_ids: Vec<String>,
    pub scores: Vec<f64>,
}

impl QueryResult {
    pub fn len(&self) -> usize {
        self.entity_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entity_ids.is_empty()
    }
}

/// Entity update payload
pub struct EntityUpdate {
    pub data: Option<HashMap<String, serde_json::Value>>,
    pub embeddings: Option<HashMap<MetricType, EmbeddingData>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}
