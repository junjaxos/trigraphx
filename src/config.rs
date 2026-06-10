use std::path::PathBuf;
use std::env;

use crate::error::Result;

/// Configuration for TriGraphX
#[derive(Clone)]
pub struct Config {
    pub data_dir: PathBuf,
    pub db_name: String,
    pub max_entities: usize,
    pub batch_size: usize,
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        Self {
            data_dir: PathBuf::from(env::var("TRIGRAPHX_DATA_DIR").unwrap_or_else(|_| "trigraphx_data".to_string())),
            db_name: env::var("TRIGRAPHX_DB_NAME").unwrap_or_else(|_| "default".to_string()),
            max_entities: env::var("TRIGRAPHX_MAX_ENTITIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10000),
            batch_size: env::var("TRIGRAPHX_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10000),
            log_level: env::var("TRIGRAPHX_LOG_LEVEL").unwrap_or_else(|_| "INFO".to_string()),
        }
    }

    pub fn db_root(&self) -> PathBuf {
        self.data_dir.join(&self.db_name)
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        let root = self.db_root();
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(root.join("entities"))?;
        std::fs::create_dir_all(root.join("checkpoints"))?;
        std::fs::create_dir_all(root.join("index"))?;
        Ok(())
    }

    pub fn list_databases(&self) -> Vec<String> {
        if !self.data_dir.exists() {
            return vec![];
        }
        std::fs::read_dir(&self.data_dir)
            .ok()
            .into_iter()
            .flat_map(|entries| {
                entries.filter_map(|e| e.ok()).filter(|e| {
                    e.path().is_dir() && e.path().join("index").exists()
                }).filter_map(|e| e.file_name().to_str().map(String::from))
            })
            .collect()
    }

    pub fn create_database(&mut self, name: &str) -> Result<PathBuf> {
        self.db_name = name.to_string();
        self.ensure_dirs()?;
        Ok(self.db_root())
    }
}
