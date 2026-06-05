"""
Persistence layer for TriGraphX - handles JSONL storage, SQLite indices, and checkpointing.
"""

import os
import json
import sqlite3
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .entity import Entity, MetricType

logger = logging.getLogger(__name__)


class PersistenceLayer:
    """
    Multi-format persistence:
    - JSONL: Batch entity storage (10K entities per file)
    - SQLite: Indices and metadata
    - tar.gz: Snapshots for versioning
    """
    
    def __init__(self, db_root: str, batch_size: int = 10000):
        self.db_root = Path(db_root)
        self.batch_size = batch_size
        
        # Create directory structure
        self.entities_dir = self.db_root / "entities"
        self.metrics_dir = self.db_root / "metrics"
        self.index_dir = self.db_root / "index"
        self.checkpoints_dir = self.db_root / "checkpoints"
        self.operations_dir = self.db_root / "operations"
        
        for d in [self.entities_dir, self.metrics_dir, self.index_dir, 
                  self.checkpoints_dir, self.operations_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # SQLite indices
        self.index_db = self.index_dir / "index.db"
        self._init_sqlite()
        
        # Operation log for audit trail
        self.operations_log = self.operations_dir / "operations.jsonl"
    
    def _init_sqlite(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(str(self.index_db))
        cursor = conn.cursor()
        
        # Entity index
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                batch_id INTEGER,
                created_at TEXT,
                updated_at TEXT,
                deleted BOOLEAN,
                hash TEXT
            )
        """)
        
        # Metric index
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id TEXT PRIMARY KEY,
                entity_id TEXT,
                metric_type TEXT,
                indexed_at TEXT
            )
        """)
        
        # Checkpoint index
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                created_at TEXT,
                entity_count INTEGER,
                description TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_entities_batch(self, entities: List[Entity], batch_id: int) -> bool:
        """Save batch of entities to JSONL."""
        batch_file = self.entities_dir / f"batch_{batch_id:06d}.jsonl"
        
        try:
            with open(batch_file, 'w') as f:
                for entity in entities:
                    f.write(json.dumps(entity.to_dict()) + '\n')
            
            # Update index
            conn = sqlite3.connect(str(self.index_db))
            cursor = conn.cursor()
            
            for entity in entities:
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, batch_id, created_at, updated_at, deleted, hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    batch_id,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.deleted,
                    entity.get_hash()
                ))
            
            conn.commit()
            conn.close()
            
            self._log_operation("save_batch", {"batch_id": batch_id, "count": len(entities)})
            return True
        
        except Exception as e:
            logger.error(f"Failed to save batch {batch_id}: {e}")
            return False
    
    def load_all_entities(self) -> List[Entity]:
        """Load all entities from JSONL files."""
        entities = []
        batch_files = sorted(self.entities_dir.glob("batch_*.jsonl"))
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entity_dict = json.loads(line)
                            entity = Entity.from_dict(entity_dict)
                            entities.append(entity)
            except Exception as e:
                logger.error(f"Failed to load {batch_file}: {e}")
        
        return entities
    
    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity in persistence layer."""
        try:
            # Find entity in batches
            for batch_file in sorted(self.entities_dir.glob("batch_*.jsonl")):
                entities = []
                found = False
                
                with open(batch_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entity_dict = json.loads(line)
                            if entity_dict['id'] == entity_id:
                                # Apply updates
                                if 'data' in updates:
                                    entity_dict['data'].update(updates['data'])
                                if 'metadata' in updates:
                                    entity_dict['metadata'].update(updates['metadata'])
                                if 'deleted' in updates:
                                    entity_dict['deleted'] = updates['deleted']
                                entity_dict['updated_at'] = datetime.utcnow().isoformat()
                                found = True
                            entities.append(entity_dict)
                
                if found:
                    # Rewrite batch
                    with open(batch_file, 'w') as f:
                        for entity_dict in entities:
                            f.write(json.dumps(entity_dict) + '\n')
                    
                    self._log_operation("update_entity", {"entity_id": entity_id})
                    return True
            
            logger.warning(f"Entity {entity_id} not found")
            return False
        
        except Exception as e:
            logger.error(f"Failed to update entity {entity_id}: {e}")
            return False
    
    def delete_entity_soft(self, entity_id: str) -> bool:
        """Mark entity as deleted (soft delete)."""
        return self.update_entity(entity_id, {"deleted": True})
    
    def delete_entity_hard(self, entity_id: str) -> bool:
        """Permanently remove entity from storage."""
        try:
            for batch_file in sorted(self.entities_dir.glob("batch_*.jsonl")):
                entities = []
                found = False
                
                with open(batch_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entity_dict = json.loads(line)
                            if entity_dict['id'] != entity_id:
                                entities.append(entity_dict)
                            else:
                                found = True
                
                if found:
                    with open(batch_file, 'w') as f:
                        for entity_dict in entities:
                            f.write(json.dumps(entity_dict) + '\n')
                    
                    # Update index
                    conn = sqlite3.connect(str(self.index_db))
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
                    conn.commit()
                    conn.close()
                    
                    self._log_operation("hard_delete_entity", {"entity_id": entity_id})
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to hard delete entity {entity_id}: {e}")
            return False
    
    def hard_delete_marked_entities(self) -> int:
        """Reclaim space by removing all soft-deleted entities."""
        count = 0
        for batch_file in sorted(self.entities_dir.glob("batch_*.jsonl")):
            entities = []
            
            with open(batch_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entity_dict = json.loads(line)
                        if not entity_dict.get('deleted', False):
                            entities.append(entity_dict)
                        else:
                            count += 1
            
            with open(batch_file, 'w') as f:
                for entity_dict in entities:
                    f.write(json.dumps(entity_dict) + '\n')
        
        self._log_operation("hard_delete_marked", {"count": count})
        return count
    
    def create_checkpoint(self, description: str = "") -> str:
        """Create snapshot for versioning/recovery."""
        timestamp = datetime.utcnow().isoformat()
        checkpoint_id = f"checkpoint_{timestamp.replace(':', '-').replace('.', '_')}"
        
        # Load all entities
        entities = self.load_all_entities()
        
        # Create tar.gz
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.tar.gz"
        
        with tarfile.open(checkpoint_file, 'w:gz') as tar:
            # Add entities
            entities_list = [e.to_dict() for e in entities]
            entities_json = json.dumps(entities_list, indent=2).encode('utf-8')
            
            import io
            tarinfo = tarfile.TarInfo(name="entities.json")
            tarinfo.size = len(entities_json)
            tar.addfile(tarinfo, io.BytesIO(entities_json))
            
            # Add metadata
            metadata = {
                "checkpoint_id": checkpoint_id,
                "created_at": timestamp,
                "entity_count": len(entities),
                "description": description,
            }
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
            
            tarinfo = tarfile.TarInfo(name="metadata.json")
            tarinfo.size = len(metadata_json)
            tar.addfile(tarinfo, io.BytesIO(metadata_json))
        
        # Record in index
        conn = sqlite3.connect(str(self.index_db))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO checkpoints (checkpoint_id, created_at, entity_count, description)
            VALUES (?, ?, ?, ?)
        """, (checkpoint_id, timestamp, len(entities), description))
        conn.commit()
        conn.close()
        
        self._log_operation("create_checkpoint", {
            "checkpoint_id": checkpoint_id,
            "entity_count": len(entities)
        })
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from checkpoint."""
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.tar.gz"
        
        if not checkpoint_file.exists():
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        try:
            # Extract and reload entities
            with tarfile.open(checkpoint_file, 'r:gz') as tar:
                entities_member = tar.getmember("entities.json")
                entities_file = tar.extractfile(entities_member)
                entities_data = json.load(entities_file)
            
            # Clear current data
            for batch_file in self.entities_dir.glob("batch_*.jsonl"):
                batch_file.unlink()
            
            # Rewrite all batches
            for i in range(0, len(entities_data), self.batch_size):
                batch = entities_data[i:i+self.batch_size]
                batch_id = i // self.batch_size
                
                batch_file = self.entities_dir / f"batch_{batch_id:06d}.jsonl"
                with open(batch_file, 'w') as f:
                    for entity_dict in batch:
                        f.write(json.dumps(entity_dict) + '\n')
            
            self._log_operation("restore_checkpoint", {"checkpoint_id": checkpoint_id})
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return False
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        conn = sqlite3.connect(str(self.index_db))
        cursor = conn.cursor()
        cursor.execute("SELECT checkpoint_id, created_at, entity_count, description FROM checkpoints ORDER BY created_at DESC")
        
        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append({
                "checkpoint_id": row[0],
                "created_at": row[1],
                "entity_count": row[2],
                "description": row[3],
            })
        
        conn.close()
        return checkpoints
    
    def _log_operation(self, op_type: str, details: Dict[str, Any]):
        """Log operation for audit trail."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": op_type,
            "details": details,
        }
        
        with open(self.operations_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_operation_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent operation logs."""
        operations = []
        
        if not self.operations_log.exists():
            return operations
        
        with open(self.operations_log, 'r') as f:
            for line in f:
                if line.strip():
                    operations.append(json.loads(line))
        
        return operations[-limit:]
    
    def cleanup_old_checkpoints(self, keep_recent: int = 5) -> int:
        """Delete old checkpoints, keeping only recent ones."""
        checkpoints = self.list_checkpoints()
        deleted = 0
        
        conn = sqlite3.connect(str(self.index_db))
        cursor = conn.cursor()
        
        for checkpoint in checkpoints[keep_recent:]:
            try:
                checkpoint_file = self.checkpoints_dir / f"{checkpoint['checkpoint_id']}.tar.gz"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                cursor.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint["checkpoint_id"],))
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        batch_files = list(self.entities_dir.glob("batch_*.jsonl"))
        total_bytes = sum(f.stat().st_size for f in batch_files)
        
        checkpoints = self.list_checkpoints()
        checkpoint_bytes = sum((self.checkpoints_dir / f"{c['checkpoint_id']}.tar.gz").stat().st_size 
                              for c in checkpoints if (self.checkpoints_dir / f"{c['checkpoint_id']}.tar.gz").exists())
        
        return {
            "batch_files": len(batch_files),
            "entity_storage_mb": total_bytes / 1024 / 1024,
            "checkpoints": len(checkpoints),
            "checkpoint_storage_mb": checkpoint_bytes / 1024 / 1024,
            "index_db_mb": self.index_db.stat().st_size / 1024 / 1024 if self.index_db.exists() else 0,
        }
