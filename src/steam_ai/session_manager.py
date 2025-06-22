import json
import gzip
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from dataclasses import dataclass, asdict
import time
from pathlib import Path
import hashlib
import shutil
from enum import Enum

from .config import Config
from .steam_client import Review
from .index_manager import IndexMetadata

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"


@dataclass
class SessionMetadata:
    """Metadata for a session."""
    app_id: str
    app_name: Optional[str]
    creation_time: float
    last_modified: float
    version: str
    total_reviews: int
    processed_reviews: int
    embedding_model: str
    index_type: str
    config_hash: str
    compression_type: str
    file_sizes: Dict[str, int]
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary."""
        return cls(**data)


class SessionManager:
    """Manages session storage with compression and optimization."""
    
    VERSION = "2.0.0"
    
    def __init__(self, config: Config):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.compression_type = CompressionType.GZIP  # Default compression
        
    def save_session(self, 
                    app_id: str,
                    reviews: List[Review],
                    processed_texts: List[str],
                    embeddings: List[List[float]],
                    index: faiss.Index,
                    index_metadata: IndexMetadata,
                    app_name: Optional[str] = None,
                    compression: CompressionType = CompressionType.GZIP) -> str:
        """
        Save a complete session with compression and metadata.
        
        Args:
            app_id: Steam app ID
            reviews: List of Review objects
            processed_texts: Processed review texts
            embeddings: Embedding vectors
            index: FAISS index
            index_metadata: Index metadata
            app_name: Optional app name
            compression: Compression type to use
            
        Returns:
            str: Path to session directory
        """
        session_dir = self.save_dir / app_id
        session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving session for app {app_id} to {session_dir}")
        
        file_sizes = {}
        
        try:
            # Save reviews
            reviews_path = session_dir / "reviews.json"
            if compression == CompressionType.GZIP:
                reviews_path = reviews_path.with_suffix(".json.gz")
                
            self._save_reviews(reviews, reviews_path, compression)
            file_sizes["reviews"] = reviews_path.stat().st_size
            
            # Save processed texts
            texts_path = session_dir / "texts.txt"
            if compression == CompressionType.GZIP:
                texts_path = texts_path.with_suffix(".txt.gz")
                
            self._save_texts(processed_texts, texts_path, compression)
            file_sizes["texts"] = texts_path.stat().st_size
            
            # Save embeddings
            embeddings_path = session_dir / "embeddings.npy"
            if compression == CompressionType.GZIP:
                embeddings_path = embeddings_path.with_suffix(".npy.gz")
                
            self._save_embeddings(embeddings, embeddings_path, compression)
            file_sizes["embeddings"] = embeddings_path.stat().st_size
            
            # Save FAISS index
            index_path = session_dir / "index.faiss"
            faiss.write_index(index, str(index_path))
            file_sizes["index"] = index_path.stat().st_size
            
            # Save index metadata
            index_metadata_path = session_dir / "index_metadata.json"
            with open(index_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(index_metadata.to_dict(), f, indent=2)
            file_sizes["index_metadata"] = index_metadata_path.stat().st_size
            
            # Create session metadata
            session_metadata = SessionMetadata(
                app_id=app_id,
                app_name=app_name,
                creation_time=time.time(),
                last_modified=time.time(),
                version=self.VERSION,
                total_reviews=len(reviews),
                processed_reviews=len(processed_texts),
                embedding_model=self.config.embed_model,
                index_type=index_metadata.index_type,
                config_hash=self._get_config_hash(),
                compression_type=compression.value,
                file_sizes=file_sizes,
                checksum=self._calculate_session_checksum(session_dir)
            )
            
            # Save session metadata
            metadata_path = session_dir / "session_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(session_metadata.to_dict(), f, indent=2)
            
            logger.info(f"Session saved successfully. Total size: {sum(file_sizes.values()) / 1024 / 1024:.2f} MB")
            return str(session_dir)
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            # Clean up partial save
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise
    
    def load_session(self, app_id: str) -> Optional[Tuple[List[Review], List[str], List[List[float]], faiss.Index, IndexMetadata, SessionMetadata]]:
        """
        Load a complete session.
        
        Args:
            app_id: Steam app ID
            
        Returns:
            Optional[Tuple]: Session data or None if not found/invalid
        """
        session_dir = self.save_dir / app_id
        if not session_dir.exists():
            logger.warning(f"Session not found for app {app_id}")
            return None
        
        try:
            logger.info(f"Loading session for app {app_id} from {session_dir}")
            
            # Load session metadata
            metadata_path = session_dir / "session_metadata.json"
            if not metadata_path.exists():
                logger.warning("Session metadata not found")
                return None
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                session_metadata = SessionMetadata.from_dict(json.load(f))
            
            # Validate session integrity
            if not self._validate_session_integrity(session_dir, session_metadata):
                logger.warning("Session integrity validation failed")
                return None
            
            # Determine compression type
            compression = CompressionType(session_metadata.compression_type)
            
            # Load reviews
            reviews = self._load_reviews(session_dir, compression)
            
            # Load processed texts
            processed_texts = self._load_texts(session_dir, compression)
            
            # Load embeddings
            embeddings = self._load_embeddings(session_dir, compression)
            
            # Load FAISS index
            index_path = session_dir / "index.faiss"
            if not index_path.exists():
                logger.error("FAISS index file not found")
                return None
            index = faiss.read_index(str(index_path))
            
            # Load index metadata
            index_metadata_path = session_dir / "index_metadata.json"
            if not index_metadata_path.exists():
                logger.warning("Index metadata not found, using defaults")
                from .index_manager import IndexMetadata
                index_metadata = IndexMetadata(
                    index_type="unknown",
                    dimension=len(embeddings[0]) if embeddings else 0,
                    num_vectors=len(embeddings),
                    creation_time=session_metadata.creation_time,
                    model_name=session_metadata.embedding_model,
                    app_id=app_id,
                    config_hash="",
                    index_parameters={}
                )
            else:
                with open(index_metadata_path, 'r', encoding='utf-8') as f:
                    from .index_manager import IndexMetadata
                    index_metadata = IndexMetadata.from_dict(json.load(f))
            
            logger.info(f"Session loaded successfully. {len(reviews)} reviews, {len(embeddings)} embeddings")
            return reviews, processed_texts, embeddings, index, index_metadata, session_metadata
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    def _save_reviews(self, reviews: List[Review], path: Path, compression: CompressionType) -> None:
        """Save reviews with optional compression."""
        # Convert reviews to serializable format
        reviews_data = [asdict(review) for review in reviews]
        
        if compression == CompressionType.GZIP:
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(reviews_data, f, indent=2)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(reviews_data, f, indent=2)
    
    def _load_reviews(self, session_dir: Path, compression: CompressionType) -> List[Review]:
        """Load reviews with decompression."""
        if compression == CompressionType.GZIP:
            reviews_path = session_dir / "reviews.json.gz"
        else:
            reviews_path = session_dir / "reviews.json"
        
        if not reviews_path.exists():
            # Try alternative path
            alt_path = session_dir / "reviews.json" if compression == CompressionType.GZIP else session_dir / "reviews.json.gz"
            if alt_path.exists():
                reviews_path = alt_path
                compression = CompressionType.NONE if compression == CompressionType.GZIP else CompressionType.GZIP
            else:
                logger.error("Reviews file not found")
                return []
        
        try:
            if compression == CompressionType.GZIP:
                with gzip.open(reviews_path, 'rt', encoding='utf-8') as f:
                    reviews_data = json.load(f)
            else:
                with open(reviews_path, 'r', encoding='utf-8') as f:
                    reviews_data = json.load(f)
            
            # Convert back to Review objects
            from .steam_client import Review
            return [Review(**data) for data in reviews_data]
            
        except Exception as e:
            logger.error(f"Failed to load reviews: {e}")
            return []
    
    def _save_texts(self, texts: List[str], path: Path, compression: CompressionType) -> None:
        """Save processed texts with optional compression."""
        content = '\n'.join(texts)
        
        if compression == CompressionType.GZIP:
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _load_texts(self, session_dir: Path, compression: CompressionType) -> List[str]:
        """Load processed texts with decompression."""
        if compression == CompressionType.GZIP:
            texts_path = session_dir / "texts.txt.gz"
        else:
            texts_path = session_dir / "texts.txt"
        
        if not texts_path.exists():
            # Try alternative path
            alt_path = session_dir / "texts.txt" if compression == CompressionType.GZIP else session_dir / "texts.txt.gz"
            if alt_path.exists():
                texts_path = alt_path
                compression = CompressionType.NONE if compression == CompressionType.GZIP else CompressionType.GZIP
            else:
                logger.error("Texts file not found")
                return []
        
        try:
            if compression == CompressionType.GZIP:
                with gzip.open(texts_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(texts_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            return content.strip().split('\n') if content.strip() else []
            
        except Exception as e:
            logger.error(f"Failed to load texts: {e}")
            return []
    
    def _save_embeddings(self, embeddings: List[List[float]], path: Path, compression: CompressionType) -> None:
        """Save embeddings with optional compression."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if compression == CompressionType.GZIP:
            with gzip.open(path, 'wb') as f:
                np.save(f, embeddings_array)
        else:
            np.save(path, embeddings_array)
    
    def _load_embeddings(self, session_dir: Path, compression: CompressionType) -> List[List[float]]:
        """Load embeddings with decompression."""
        if compression == CompressionType.GZIP:
            embeddings_path = session_dir / "embeddings.npy.gz"
        else:
            embeddings_path = session_dir / "embeddings.npy"
        
        if not embeddings_path.exists():
            # Try alternative path
            alt_path = session_dir / "embeddings.npy" if compression == CompressionType.GZIP else session_dir / "embeddings.npy.gz"
            if alt_path.exists():
                embeddings_path = alt_path
                compression = CompressionType.NONE if compression == CompressionType.GZIP else CompressionType.GZIP
            else:
                logger.error("Embeddings file not found")
                return []
        
        try:
            if compression == CompressionType.GZIP:
                with gzip.open(embeddings_path, 'rb') as f:
                    embeddings_array = np.load(f)
            else:
                embeddings_array = np.load(embeddings_path)
            
            return embeddings_array.tolist()
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return []
    
    def _find_session_directory(self, app_id: str) -> Optional[Path]:
        """Find session directory."""
        # Check main sessions directory
        session_dir = self.save_dir / app_id
        if session_dir.exists():
            return session_dir
        
        # Check for legacy versioned directories
        for version_dir in self.save_dir.glob("v*"):
            if version_dir.is_dir():
                legacy_session_dir = version_dir / app_id
                if legacy_session_dir.exists():
                    logger.info(f"Found session in legacy version directory: {version_dir.name}")
                    return legacy_session_dir
        
        # Check legacy format (files directly in save_dir)
        legacy_session_files = list(self.save_dir.glob(f"{app_id}.*"))
        if legacy_session_files:
            logger.info("Found legacy session format")
            return self.save_dir
        
        return None
    
    def _validate_session_integrity(self, session_dir: Path, metadata: SessionMetadata) -> bool:
        """Validate session integrity using checksums and file sizes."""
        try:
            # Check if all required files exist
            required_files = ["session_metadata.json", "index.faiss"]
            
            # Check for data files based on compression type
            compression = CompressionType(metadata.compression_type)
            if compression == CompressionType.GZIP:
                required_files.extend(["reviews.json.gz", "texts.txt.gz", "embeddings.npy.gz"])
            else:
                required_files.extend(["reviews.json", "texts.txt", "embeddings.npy"])
            
            for filename in required_files:
                if not (session_dir / filename).exists():
                    logger.warning(f"Missing required file: {filename}")
                    return False
            
            # Validate file sizes if available
            if metadata.file_sizes:
                for filename, expected_size in metadata.file_sizes.items():
                    # Map filename to actual file path based on compression type and file type
                    if filename == "reviews":
                        actual_file_path = session_dir / ("reviews.json.gz" if compression == CompressionType.GZIP else "reviews.json")
                    elif filename == "texts":
                        actual_file_path = session_dir / ("texts.txt.gz" if compression == CompressionType.GZIP else "texts.txt")
                    elif filename == "embeddings":
                        actual_file_path = session_dir / ("embeddings.npy.gz" if compression == CompressionType.GZIP else "embeddings.npy")
                    elif filename == "index":
                        actual_file_path = session_dir / "index.faiss"
                    elif filename == "index_metadata":
                        actual_file_path = session_dir / "index_metadata.json"
                    else:
                        # Fallback to original behavior for unknown files
                        actual_files = list(session_dir.glob(f"{filename}*"))
                        actual_file_path = actual_files[0] if actual_files else None
                    
                    if actual_file_path and actual_file_path.exists():
                        actual_size = actual_file_path.stat().st_size
                        if abs(actual_size - expected_size) > expected_size * 0.1:  # 10% tolerance
                            logger.warning(f"File size mismatch for {filename}: expected {expected_size}, got {actual_size}")
                            return False
            
            # Validate checksum if available
            if metadata.checksum:
                current_checksum = self._calculate_session_checksum(session_dir)
                if current_checksum != metadata.checksum:
                    logger.warning("Session checksum mismatch")
                    # Don't fail on checksum mismatch, just warn
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating session integrity: {e}")
            return False
    
    def _calculate_session_checksum(self, session_dir: Path) -> str:
        """Calculate checksum for session directory."""
        hash_md5 = hashlib.md5()
        
        # Sort files for consistent checksum
        for file_path in sorted(session_dir.glob("*")):
            if file_path.is_file() and file_path.name != "session_metadata.json":
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_config_hash(self) -> str:
        """Generate hash of relevant configuration parameters."""
        config_str = f"{self.config.embed_model}_{self.config.embedding_batch_size}_{self.config.similarity_top_k}_{self.config.review_max_length}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions with metadata."""
        sessions = []
        
        # Check main sessions directory
        for session_dir in self.save_dir.iterdir():
            if session_dir.is_dir() and not session_dir.name.startswith("v"):
                metadata_path = session_dir / "session_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        metadata["session_dir"] = str(session_dir)
                        sessions.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {session_dir}: {e}")
        
        # Check legacy version directories
        for version_dir in self.save_dir.glob("v*"):
            if version_dir.is_dir():
                for session_dir in version_dir.iterdir():
                    if session_dir.is_dir():
                        metadata_path = session_dir / "session_metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                metadata["version_dir"] = version_dir.name
                                metadata["session_dir"] = str(session_dir)
                                sessions.append(metadata)
                            except Exception as e:
                                logger.warning(f"Failed to load metadata for {session_dir}: {e}")
        
        # Check for legacy sessions (files directly in save_dir)
        legacy_files = list(self.save_dir.glob("*.index.faiss"))
        for index_file in legacy_files:
            app_id = index_file.stem.replace(".index", "")
            sessions.append({
                "app_id": app_id,
                "version": "legacy",
                "version_dir": "legacy",
                "session_dir": str(self.save_dir),
                "creation_time": index_file.stat().st_mtime,
                "last_modified": index_file.stat().st_mtime
            })
        
        return sorted(sessions, key=lambda x: x.get("last_modified", 0), reverse=True)
    
    def delete_session(self, app_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            app_id: Steam app ID
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        session_dir = self._find_session_directory(app_id)
        if not session_dir:
            logger.warning(f"Session not found for app {app_id}")
            return False
        
        try:
            if session_dir == self.save_dir:
                # Legacy format - delete individual files
                legacy_files = list(self.save_dir.glob(f"{app_id}.*"))
                for file_path in legacy_files:
                    file_path.unlink()
                logger.info(f"Deleted legacy session files for app {app_id}")
            else:
                # Directory format - delete directory
                shutil.rmtree(session_dir)
                logger.info(f"Deleted session directory: {session_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session for app {app_id}: {e}")
            return False
    
    def migrate_legacy_session(self, app_id: str) -> bool:
        """
        Migrate a legacy session to the new format.
        
        Args:
            app_id: Steam app ID
            
        Returns:
            bool: True if migrated successfully, False otherwise
        """
        try:
            # Check for legacy files
            legacy_files = {
                'index': self.save_dir / f"{app_id}.index.faiss",
                'embeddings': self.save_dir / f"{app_id}.embeddings.npy",
                'texts': self.save_dir / f"{app_id}.texts.txt"
            }
            
            if not all(path.exists() for path in legacy_files.values()):
                logger.warning(f"Incomplete legacy session for app {app_id}")
                return False
            
            # Load legacy data
            index = faiss.read_index(str(legacy_files['index']))
            embeddings = np.load(legacy_files['embeddings']).tolist()
            with open(legacy_files['texts'], 'r', encoding='utf-8') as f:
                processed_texts = f.read().strip().split('\n')
            
            # Create minimal Review objects (we don't have full review data in legacy format)
            from .steam_client import Review
            reviews = []
            for i, text in enumerate(processed_texts):
                review = Review(
                    text=text,
                    author=f"legacy_user_{i}",
                    helpful=0,
                    funny=0,
                    timestamp_created=0,
                    timestamp_updated=0,
                    voted_up=True,
                    votes_up=0,
                    votes_funny=0,
                    weighted_vote_score=0.0,
                    comment_count=0,
                    steam_purchase=True,
                    received_for_free=False,
                    written_during_early_access=False,
                    playtime_forever=0,
                    playtime_at_review=0
                )
                reviews.append(review)
            
            # Create index metadata
            index_metadata = IndexMetadata(
                index_type="legacy",
                dimension=len(embeddings[0]) if embeddings else 0,
                num_vectors=len(embeddings),
                creation_time=legacy_files['index'].stat().st_mtime,
                model_name=self.config.embed_model,
                app_id=app_id,
                config_hash="legacy",
                index_parameters={}
            )
            
            # Save in new format
            self.save_session(
                app_id=app_id,
                reviews=reviews,
                processed_texts=processed_texts,
                embeddings=embeddings,
                index=index,
                index_metadata=index_metadata,
                app_name=None,
                compression=CompressionType.GZIP
            )
            
            # Delete legacy files
            for file_path in legacy_files.values():
                file_path.unlink()
            
            logger.info(f"Successfully migrated legacy session for app {app_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy session for app {app_id}: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions."""
        sessions = self.list_sessions()
        
        total_size = 0
        version_counts = {}
        
        for session in sessions:
            if "file_sizes" in session:
                total_size += sum(session["file_sizes"].values())
            
            version = session.get("version", "unknown")
            version_counts[version] = version_counts.get(version, 0) + 1
        
        return {
            "total_sessions": len(sessions),
            "total_size_mb": total_size / (1024 * 1024),
            "version_counts": version_counts,
            "current_version": self.VERSION,
            "save_directory": str(self.save_dir)
        }
    
    def migrate_legacy_sessions(self) -> int:
        """
        Migrate sessions from legacy version directories to main sessions folder.
        
        Returns:
            int: Number of sessions migrated
        """
        migrated_count = 0
        
        # Migrate from version directories
        for version_dir in self.save_dir.glob("v*"):
            if version_dir.is_dir():
                for session_dir in version_dir.iterdir():
                    if session_dir.is_dir():
                        target_dir = self.save_dir / session_dir.name
                        if not target_dir.exists():
                            try:
                                shutil.move(str(session_dir), str(target_dir))
                                logger.info(f"Migrated session {session_dir.name} from {version_dir.name}")
                                migrated_count += 1
                            except Exception as e:
                                logger.error(f"Failed to migrate session {session_dir.name}: {e}")
                
                # Remove empty version directory
                try:
                    if not any(version_dir.iterdir()):
                        version_dir.rmdir()
                        logger.info(f"Removed empty version directory: {version_dir.name}")
                except Exception as e:
                    logger.warning(f"Could not remove version directory {version_dir.name}: {e}")
        
        return migrated_count