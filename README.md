# TriGraphX: Multi-dimensional Relational Metric Space

**Pure Rust** unified database model replacing **tree index + graph DB + vector DB** with a single metric space abstraction.

## 🚀 Quick Start

### Add as Dependency

```toml
[dependencies]
trigraphx = { path = "/path/to/trigraphx" }
```

### Basic Usage

```rust
use trigraphx::{MetricSpace, MetricType, SemanticEmbedding, Entity};
use std::collections::HashMap;

fn main() {
    // Create metric space
    let mut space = MetricSpace::new(10000);

    // Ingest entity (auto-ID, auto-dedup by "name" field)
    let mut data = HashMap::new();
    data.insert("name".to_string(), serde_json::json!("张三"));
    data.insert("type".to_string(), serde_json::json!("person"));
    let (entity, created) = space.ingest(data, None, None, None);
    println!("ID: {}, created: {}", entity.id, created);

    // Same entity mentioned again — data merges automatically
    let mut data2 = HashMap::new();
    data2.insert("name".to_string(), serde_json::json!("张三"));
    data2.insert("company".to_string(), serde_json::json!("红杉资本"));
    let (entity2, created2) = space.ingest(data2, None, None, None);
    println!("ID: {}, created: {}", entity2.id, created2); // Same ID, created: false

    // Add semantic embedding
    let mut emb = HashMap::new();
    emb.insert(
        MetricType::Semantic,
        trigraphx::EmbeddingData::Semantic(SemanticEmbedding::new(vec![0.1, 0.2, 0.3, 0.4])),
    );
    space.ingest(HashMap::new(), Some(emb), None, None);

    // KNN query
    let result = space.knn_query(&entity.id, 10, MetricType::Semantic).unwrap();
    println!("Found {} neighbors", result.len());
}
```

## 📋 Project Structure

```
trigraphx/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Public API exports
│   ├── entity.rs        # Entity, Embeddings, Distance Metrics
│   ├── space.rs         # MetricSpace (KNN, Range, Multi-metric, ingest)
│   ├── persistence.rs   # JSONL + SQLite + Checkpointing
│   ├── config.rs        # Configuration management
│   └── error.rs         # Error types
└── README.md
```

## 🎯 Core Features

### 1. **Unified Query Model**

All data relationships (trees, graphs, vectors, causality) expressed as distance functions in metric space.

```rust
// Hierarchy query (tree distance)
let result = space.knn_query(&entity_id, 5, MetricType::Hierarchy)?;

// Semantic query (vector similarity)
let result = space.knn_query(&entity_id, 5, MetricType::Semantic)?;

// Association query (graph relationships)
let result = space.knn_query(&entity_id, 5, MetricType::Association)?;

// Multi-metric query (combined)
let mut weights = HashMap::new();
weights.insert(MetricType::Semantic, 0.5);
weights.insert(MetricType::Hierarchy, 0.3);
weights.insert(MetricType::Association, 0.2);
let result = space.multi_metric_query(&entity_id, 5, weights)?;
```

### 2. **Complete CRUD Operations**

```rust
// Create — auto-ID, auto-dedup
let (entity, created) = space.ingest(data, None, None, None);

// Upsert — insert if new, update if exists
let (entity, created) = space.upsert_entity(entity, merge_data: true);

// Read
let entity = space.get_entity(&entity_id);

// Delete (soft - recoverable)
space.soft_delete_entity(&entity_id)?;

// Delete (hard - permanent)
space.hard_delete_entity(&entity_id)?;
```

### 3. **Persistence with Multiple Formats**

```rust
use trigraphx::PersistenceLayer;

let persist = PersistenceLayer::new(PathBuf::from("./trigraphx_data/default"), 10000)?;

// Save entities to JSONL
persist.save_entities_batch(&entities, batch_id: 0)?;

// Load all entities
let entities = persist.load_all_entities()?;

// Create checkpoint (snapshot for recovery)
let checkpoint_id = persist.create_checkpoint("Before migration")?;

// List checkpoints
let checkpoints = persist.list_checkpoints()?;

// Restore from checkpoint
persist.restore_checkpoint(&checkpoint_id)?;
```

### 4. **Multi-Database Support**

```rust
use trigraphx::Config;

let mut config = Config::new();

// List existing databases
let dbs = config.list_databases();

// Create and switch to a new database
config.create_database("my_project")?;

// Or via environment variables:
// TRIGRAPHX_DATA_DIR=/path/to/data
// TRIGRAPHX_DB_NAME=my_project
```

## 📊 Embedding Types

| Type | Use Case | Distance |
|------|----------|----------|
| `HierarchyEmbedding` | Tree-structured data, taxonomy | Tree path distance |
| `SemanticEmbedding` | Text, images, vectors | Cosine distance |
| `AssociationEmbedding` | Graph relationships, social network | Edge weight distance |
| `CausalEmbedding` | Cause-effect, event chains | Causal strength |

## 🧪 Building & Testing

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run with all features
cargo test --release
```

## � Configuration

### Environment Variables

```bash
# Data storage directory (default: trigraphx_data/)
export TRIGRAPHX_DATA_DIR=/path/to/my/data

# Database name (default: default)
export TRIGRAPHX_DB_NAME=my_project

# Max entities in MetricSpace (default: 10000)
export TRIGRAPHX_MAX_ENTITIES=100000

# Batch size for JSONL persistence (default: 10000)
export TRIGRAPHX_BATCH_SIZE=5000

# Log level (default: INFO)
export TRIGRAPHX_LOG_LEVEL=DEBUG
```

### Programmatic Configuration

```rust
use trigraphx::Config;

let mut config = Config::new();
config.data_dir = PathBuf::from("/custom/path");
config.max_entities = 50000;
config.ensure_dirs()?;
```

## 📄 License

MIT License
