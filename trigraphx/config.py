"""
TriGraphX configuration module.

All configurable settings are centralized here. Override via environment
variables or by modifying the Config singleton before creating any components.

Environment variable overrides:
    TRIGRAPHX_DATA_DIR     - Data storage root directory
    TRIGRAPHX_MAX_ENTITIES  - Maximum entities in MetricSpace
    TRIGRAPHX_BATCH_SIZE    - JSONL batch size for persistence
    TRIGRAPHX_LOG_LEVEL     - Logging level (DEBUG, INFO, WARNING, ERROR)
"""

import os
from pathlib import Path


class Config:
    """Singleton configuration for TriGraphX."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # ── Data Storage ──────────────────────────────────────────
        # Root directory for all persistent data (JSONL, SQLite, checkpoints)
        self.data_dir = Path(
            os.environ.get("TRIGRAPHX_DATA_DIR", "trigraphx_data")
        )

        # ── MetricSpace ───────────────────────────────────────────
        # Maximum number of entities in the metric space
        self.max_entities = int(
            os.environ.get("TRIGRAPHX_MAX_ENTITIES", "10000")
        )

        # ── Persistence ───────────────────────────────────────────
        # Number of entities per JSONL batch file
        self.batch_size = int(
            os.environ.get("TRIGRAPHX_BATCH_SIZE", "10000")
        )

        # ── Logging ───────────────────────────────────────────────
        self.log_level = os.environ.get("TRIGRAPHX_LOG_LEVEL", "INFO")

        # ── Deduplication ─────────────────────────────────────────
        # Default keys to use for entity ID generation and deduplication
        self.dedup_keys = ["name"]

    @property
    def entities_dir(self) -> Path:
        return self.data_dir / "entities"

    @property
    def metrics_dir(self) -> Path:
        return self.data_dir / "metrics"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def operations_dir(self) -> Path:
        return self.data_dir / "operations"

    def ensure_dirs(self):
        """Create all data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for d in [self.entities_dir, self.metrics_dir, self.index_dir,
                   self.checkpoints_dir, self.operations_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton
config = Config()