use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::PathBuf;

use rusqlite::Connection;
use chrono::Utc;

use crate::entity::*;
use crate::error::{Result, TriGraphXError};

/// Persistence layer for entities
pub struct PersistenceLayer {
    entities_dir: PathBuf,
    checkpoints_dir: PathBuf,
    index_db: PathBuf,
    #[allow(dead_code)]
    db_root: PathBuf,
    #[allow(dead_code)]
    batch_size: usize,
}

impl PersistenceLayer {
    pub fn new(db_root: PathBuf, batch_size: usize) -> Result<Self> {
        let entities_dir = db_root.join("entities");
        let checkpoints_dir = db_root.join("checkpoints");
        let index_db = db_root.join("index").join("metadata.db");

        fs::create_dir_all(&entities_dir)?;
        fs::create_dir_all(&checkpoints_dir)?;
        fs::create_dir_all(index_db.parent().unwrap())?;

        Ok(Self {
            db_root,
            entities_dir,
            checkpoints_dir,
            index_db,
            batch_size,
        })
    }

    /// Save entities to JSONL
    pub fn save_entities_batch(&self, entities: &[Entity], batch_id: usize) -> Result<()> {
        let file_path = self.entities_dir.join(format!("batch_{}.jsonl", batch_id));
        let file = fs::File::create(&file_path)?;
        let mut writer = BufWriter::new(file);

        for entity in entities {
            let json = serde_json::to_string(entity)
                .map_err(|e| TriGraphXError::SerializationError(e.to_string()))?;
            writeln!(writer, "{}", json)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load all entities from JSONL files
    pub fn load_all_entities(&self) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        if !self.entities_dir.exists() {
            return Ok(entities);
        }

        for entry in fs::read_dir(&self.entities_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                let file = fs::File::open(&path)?;
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let entity: Entity = serde_json::from_str(&line)
                        .map_err(|e| TriGraphXError::SerializationError(e.to_string()))?;
                    entities.push(entity);
                }
            }
        }

        Ok(entities)
    }

    /// Create checkpoint
    pub fn create_checkpoint(&self, name: &str) -> Result<String> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let checkpoint_id = format!("{}_{}", timestamp, name);
        let checkpoint_dir = self.checkpoints_dir.join(&checkpoint_id);

        fs::create_dir_all(&checkpoint_dir)?;

        // Copy current entities
        if self.entities_dir.exists() {
            for entry in fs::read_dir(&self.entities_dir)? {
                let entry = entry?;
                let src = entry.path();
                let dst = checkpoint_dir.join(entry.file_name());
                fs::copy(&src, &dst)?;
            }
        }

        // Save checkpoint metadata
        let meta = serde_json::json!({
            "id": checkpoint_id,
            "timestamp": timestamp.to_string(),
            "name": name,
            "entity_count": self.load_all_entities()?.len(),
        });

        let meta_path = checkpoint_dir.join("metadata.json");
        let mut file = fs::File::create(&meta_path)?;
        serde_json::to_writer_pretty(&mut file, &meta)
            .map_err(|e| TriGraphXError::SerializationError(e.to_string()))?;

        // Record in database
        let entity_count = self.load_all_entities()?.len();
        let conn = Connection::open(&self.index_db)
            .map_err(|e| TriGraphXError::DatabaseError(e.to_string()))?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, name TEXT, timestamp TEXT, entity_count INTEGER)",
            [],
        ).map_err(|e| TriGraphXError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (id, name, timestamp, entity_count) VALUES (?1, ?2, ?3, ?4)",
            [&checkpoint_id, name, &timestamp.to_string(), &entity_count.to_string()],
        ).map_err(|e| TriGraphXError::DatabaseError(e.to_string()))?;

        Ok(checkpoint_id)
    }

    /// List checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<String>> {
        if !self.checkpoints_dir.exists() {
            return Ok(vec![]);
        }

        let mut checkpoints = Vec::new();
        for entry in fs::read_dir(&self.checkpoints_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    checkpoints.push(name.to_string());
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Restore from checkpoint
    pub fn restore_checkpoint(&self, checkpoint_id: &str) -> Result<()> {
        let checkpoint_dir = self.checkpoints_dir.join(checkpoint_id);
        if !checkpoint_dir.exists() {
            return Err(TriGraphXError::QueryError(format!("Checkpoint not found: {}", checkpoint_id)));
        }

        // Clear current entities
        if self.entities_dir.exists() {
            fs::remove_dir_all(&self.entities_dir)?;
        }
        fs::create_dir_all(&self.entities_dir)?;

        // Copy checkpoint files back
        for entry in fs::read_dir(&checkpoint_dir)? {
            let entry = entry?;
            let src = entry.path();
            let file_name = entry.file_name();
            if file_name != "metadata.json" {
                let dst = self.entities_dir.join(&file_name);
                fs::copy(&src, &dst)?;
            }
        }

        Ok(())
    }

    /// Get stats
    pub fn stats(&self) -> Result<HashMap<String, usize>> {
        let entities = self.load_all_entities()?;
        let mut stats = HashMap::new();
        stats.insert("total_entities".to_string(), entities.len());
        stats.insert("active".to_string(), entities.iter().filter(|e| !e.deleted).count());
        stats.insert("deleted".to_string(), entities.iter().filter(|e| e.deleted).count());

        let checkpoints = self.list_checkpoints()?;
        stats.insert("checkpoints".to_string(), checkpoints.len());

        Ok(stats)
    }
}
