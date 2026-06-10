use thiserror::Error;

#[derive(Error, Debug)]
pub enum TriGraphXError {
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    #[error("Entity is deleted: {0}")]
    EntityDeleted(String),

    #[error("Metric space is full")]
    SpaceFull,

    #[error("Invalid embedding: {0}")]
    InvalidEmbedding(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Query error: {0}")]
    QueryError(String),
}

pub type Result<T> = std::result::Result<T, TriGraphXError>;
