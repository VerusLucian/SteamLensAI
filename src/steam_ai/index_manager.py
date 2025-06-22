import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import faiss
import json
from dataclasses import dataclass, asdict
import time

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata for FAISS index."""
    index_type: str
    dimension: int
    num_vectors: int
    creation_time: float
    model_name: str
    app_id: str
    config_hash: str
    index_parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary."""
        return cls(**data)


class FAISSIndexManager:
    """Optimized FAISS index manager with better performance and memory management."""
    
    def __init__(self, config: Config):
        self.config = config
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[IndexMetadata] = None
        self._is_trained = False
        
    def create_optimized_index(self, 
                              embeddings: List[List[float]], 
                              index_type: str = "auto") -> faiss.Index:
        """
        Create an optimized FAISS index based on the dataset size and characteristics.
        
        Args:
            embeddings: List of embedding vectors
            index_type: Type of index to create ("auto", "flat", "ivf", "hnsw")
            
        Returns:
            faiss.Index: Created and trained index
        """
        if not embeddings:
            raise ValueError("Cannot create index with empty embeddings")
            
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        num_vectors = embeddings_array.shape[0]
        
        logger.info(f"Creating index for {num_vectors} vectors of dimension {dimension}")
        
        # Determine optimal index type
        if index_type == "auto":
            index_type = self._determine_optimal_index_type(num_vectors, dimension)
            
        logger.info(f"Using index type: {index_type}")
        
        # Create index based on type
        if index_type == "flat":
            index = self._create_flat_index(dimension)
        elif index_type == "ivf":
            index = self._create_ivf_index(dimension, num_vectors)
        elif index_type == "hnsw":
            index = self._create_hnsw_index(dimension, num_vectors)
        elif index_type == "pq":
            index = self._create_pq_index(dimension, num_vectors)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Train index if needed
        if not index.is_trained:
            logger.info("Training index...")
            start_time = time.time()
            index.train(embeddings_array)
            training_time = time.time() - start_time
            logger.info(f"Index training completed in {training_time:.2f}s")
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        start_time = time.time()
        index.add(embeddings_array)
        add_time = time.time() - start_time
        logger.info(f"Added {num_vectors} vectors in {add_time:.2f}s")
        
        # Store metadata
        self.metadata = IndexMetadata(
            index_type=index_type,
            dimension=dimension,
            num_vectors=num_vectors,
            creation_time=time.time(),
            model_name=self.config.embed_model,
            app_id="",  # Will be set when saving
            config_hash=self._get_config_hash(),
            index_parameters=self._get_index_parameters(index_type, num_vectors, dimension)
        )
        
        self.index = index
        self._is_trained = True
        
        return index
    
    def _determine_optimal_index_type(self, num_vectors: int, dimension: int) -> str:
        """
        Determine optimal index type based on dataset characteristics.
        
        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension
            
        Returns:
            str: Recommended index type
        """
        # Memory estimation (rough)
        flat_memory_mb = (num_vectors * dimension * 4) / (1024 * 1024)  # 4 bytes per float32
        
        if num_vectors < 1000:
            # Small dataset - use flat index
            return "flat"
        elif num_vectors < 10000:
            # Medium dataset - use IVF if memory allows
            if flat_memory_mb < 100:  # Less than 100MB
                return "flat"
            else:
                return "ivf"
        elif num_vectors < 100000:
            # Large dataset - use IVF
            return "ivf"
        else:
            # Very large dataset - use HNSW for better performance
            return "hnsw"
    
    def _create_flat_index(self, dimension: int) -> faiss.Index:
        """Create a flat (brute force) index."""
        return faiss.IndexFlatL2(dimension)
    
    def _create_ivf_index(self, dimension: int, num_vectors: int) -> faiss.Index:
        """Create an IVF (Inverted File) index."""
        # Determine number of clusters
        nlist = min(4 * int(np.sqrt(num_vectors)), num_vectors // 10)
        nlist = max(nlist, 1)  # At least 1 cluster
        
        # Create quantizer and index
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Set search parameters - will be set later when optimizing
        # Note: nprobe will be set in optimize_index method
        
        return index
    
    def _create_hnsw_index(self, dimension: int, num_vectors: int) -> faiss.Index:
        """Create an HNSW (Hierarchical Navigable Small World) index."""
        # Determine M parameter (number of connections)
        M = 16 if num_vectors < 100000 else 32
        
        index = faiss.IndexHNSWFlat(dimension, M)
        
        # Set construction parameters
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        
        return index
    
    def _create_pq_index(self, dimension: int, num_vectors: int) -> faiss.Index:
        """Create a Product Quantization index."""
        # Determine subquantizer parameters
        m = 8  # Number of subquantizers
        if dimension % m != 0:
            # Adjust m to be a divisor of dimension
            for candidate_m in [4, 8, 16, 32]:
                if dimension % candidate_m == 0:
                    m = candidate_m
                    break
            else:
                m = 4  # Fallback
        
        nbits = 8  # Number of bits per subquantizer
        
        index = faiss.IndexPQ(dimension, m, nbits)
        return index
    
    def _get_config_hash(self) -> str:
        """Generate hash of relevant configuration parameters."""
        import hashlib
        config_str = f"{self.config.embed_model}_{self.config.embedding_batch_size}_{self.config.similarity_top_k}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_index_parameters(self, index_type: str, num_vectors: int, dimension: int) -> Dict[str, Any]:
        """Get index-specific parameters."""
        params = {
            "index_type": index_type,
            "num_vectors": num_vectors,
            "dimension": dimension
        }
        
        if index_type == "ivf":
            nlist = min(4 * int(np.sqrt(num_vectors)), num_vectors // 10)
            params.update({
                "nlist": max(nlist, 1),
                "nprobe": min(nlist, 10)
            })
        elif index_type == "hnsw":
            params.update({
                "M": 16 if num_vectors < 100000 else 32,
                "efConstruction": 200,
                "efSearch": 50
            })
        elif index_type == "pq":
            m = 8
            if dimension % m != 0:
                for candidate_m in [4, 8, 16, 32]:
                    if dimension % candidate_m == 0:
                        m = candidate_m
                        break
                else:
                    m = 4
            params.update({
                "m": m,
                "nbits": 8
            })
        
        return params
    
    def search(self, 
               query_vector: List[float], 
               k: Optional[int] = None,
               search_params: Optional[Dict[str, Any]] = None) -> Tuple[List[float], List[int]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            search_params: Optional search parameters
            
        Returns:
            Tuple[List[float], List[int]]: Distances and indices of nearest neighbors
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
            
        k = k or self.config.similarity_top_k
        query_array = np.array([query_vector], dtype=np.float32)
        
        # Apply search parameters if provided
        original_nprobe = None
        if search_params and 'nprobe' in search_params:
            try:
                # Try to cast to IndexIVF for nprobe access
                if hasattr(self.index, 'nprobe'):
                    original_nprobe = self.index.nprobe
                    self.index.nprobe = search_params.get('nprobe', original_nprobe)
            except Exception:
                # If casting fails, ignore search parameters
                pass
        
        try:
            distances, indices = self.index.search(query_array, k)
            return distances[0].tolist(), indices[0].tolist()
        finally:
            # Restore original parameters
            if original_nprobe is not None:
                try:
                    if hasattr(self.index, 'nprobe'):
                        self.index.nprobe = original_nprobe
                except Exception:
                    pass
    
    def batch_search(self, 
                    query_vectors: List[List[float]], 
                    k: Optional[int] = None) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Batch search for multiple query vectors.
        
        Args:
            query_vectors: List of query vectors
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple[List[List[float]], List[List[int]]]: Distances and indices for each query
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
            
        k = k or self.config.similarity_top_k
        query_array = np.array(query_vectors, dtype=np.float32)
        
        distances, indices = self.index.search(query_array, k)
        
        return distances.tolist(), indices.tolist()
    
    def save_index(self, app_id: str, save_dir: Optional[str] = None) -> str:
        """
        Save the index and metadata to disk.
        
        Args:
            app_id: Steam app ID
            save_dir: Directory to save files (defaults to config save_dir)
            
        Returns:
            str: Path to saved index file
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index or metadata not initialized")
            
        save_dir = save_dir or self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Update metadata with app_id
        self.metadata.app_id = app_id
        
        # File paths
        index_path = os.path.join(save_dir, f"{app_id}.index.faiss")
        metadata_path = os.path.join(save_dir, f"{app_id}.metadata.json")
        
        # Save index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
        return index_path
    
    def load_index(self, app_id: str, save_dir: Optional[str] = None) -> bool:
        """
        Load index and metadata from disk.
        
        Args:
            app_id: Steam app ID
            save_dir: Directory to load from (defaults to config save_dir)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        save_dir = save_dir or self.config.save_dir
        
        index_path = os.path.join(save_dir, f"{app_id}.index.faiss")
        metadata_path = os.path.join(save_dir, f"{app_id}.metadata.json")
        
        if not os.path.exists(index_path):
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        try:
            # Load index
            self.index = faiss.read_index(index_path)
            
            # Load metadata if available
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                self.metadata = IndexMetadata.from_dict(metadata_dict)
                
                # Validate compatibility
                if not self._is_compatible_index():
                    logger.warning("Index may be incompatible with current configuration")
            else:
                logger.warning("Metadata file not found, using defaults")
                self.metadata = None
            
            self._is_trained = True
            logger.info(f"Index loaded from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _is_compatible_index(self) -> bool:
        """Check if loaded index is compatible with current configuration."""
        if self.metadata is None:
            return True  # Assume compatible if no metadata
            
        # Check model compatibility
        if self.metadata.model_name != self.config.embed_model:
            logger.warning(f"Model mismatch: index uses {self.metadata.model_name}, "
                          f"config uses {self.config.embed_model}")
            return False
            
        # Check configuration hash
        current_hash = self._get_config_hash()
        if self.metadata.config_hash != current_hash:
            logger.info("Configuration has changed since index creation")
            
        return True
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dict[str, Any]: Index information
        """
        if self.index is None:
            return {"status": "not_initialized"}
            
        info = {
            "status": "loaded",
            "is_trained": self.index.is_trained,
            "ntotal": self.index.ntotal,
            "dimension": self.index.d,
            "metric_type": "L2",  # Assuming L2 metric
        }
        
        if self.metadata:
            info.update({
                "metadata": self.metadata.to_dict(),
                "creation_time": self.metadata.creation_time,
                "index_type": self.metadata.index_type,
                "model_name": self.metadata.model_name,
            })
        
        return info
    
    def optimize_index(self) -> None:
        """Optimize the index performance."""
        if self.index is None:
            return
            
        # Optimize search parameters based on index type
        if self.metadata:
            try:
                # For IVF indices, optimize nprobe
                if hasattr(self.index, 'nprobe'):
                    num_vectors = self.metadata.num_vectors
                    optimal_nprobe = min(max(1, num_vectors // 1000), 50)
                    self.index.nprobe = optimal_nprobe
                    logger.info(f"Set nprobe to {optimal_nprobe}")
            except Exception as e:
                logger.warning(f"Could not set nprobe: {e}")
        
        try:
            # For HNSW indices, optimize efSearch
            if hasattr(self.index, 'hnsw') and hasattr(self.index.hnsw, 'efSearch'):
                self.index.hnsw.efSearch = 50
                logger.info("Set HNSW efSearch to 50")
        except Exception as e:
            logger.warning(f"Could not set HNSW parameters: {e}")
    
    def get_memory_usage(self) -> Dict[str, Union[float, str]]:
        """
        Estimate memory usage of the index.
        
        Returns:
            Dict[str, Union[float, str]]: Memory usage estimates in MB
        """
        if self.index is None or self.metadata is None:
            return {"total_mb": 0.0}
            
        # Rough estimation based on index type and size
        num_vectors = self.metadata.num_vectors
        dimension = self.metadata.dimension
        
        base_memory = num_vectors * dimension * 4  # 4 bytes per float32
        
        if self.metadata.index_type == "flat":
            total_memory = base_memory
        elif self.metadata.index_type == "ivf":
            # IVF has additional overhead for clusters
            total_memory = base_memory * 1.2
        elif self.metadata.index_type == "hnsw":
            # HNSW has significant overhead for graph structure
            total_memory = base_memory * 1.5
        elif self.metadata.index_type == "pq":
            # PQ compresses vectors significantly
            total_memory = base_memory * 0.25
        else:
            total_memory = base_memory
            
        return {
            "base_mb": base_memory / (1024 * 1024),
            "total_mb": total_memory / (1024 * 1024),
            "index_type": self.metadata.index_type
        }
    
    def rebuild_if_needed(self, 
                         embeddings: List[List[float]], 
                         force_rebuild: bool = False) -> bool:
        """
        Rebuild index if needed based on configuration changes or corruption.
        
        Args:
            embeddings: Current embeddings
            force_rebuild: Force rebuild regardless of compatibility
            
        Returns:
            bool: True if index was rebuilt, False otherwise
        """
        should_rebuild = force_rebuild
        
        if not should_rebuild and self.index is None:
            should_rebuild = True
            logger.info("Index not loaded, rebuilding")
        
        if not should_rebuild and self.metadata:
            # Check if significant configuration changes
            if self.metadata.model_name != self.config.embed_model:
                should_rebuild = True
                logger.info("Model changed, rebuilding index")
            
            if len(embeddings) != self.metadata.num_vectors:
                should_rebuild = True
                logger.info("Number of vectors changed, rebuilding index")
        
        if should_rebuild:
            logger.info("Rebuilding index...")
            self.create_optimized_index(embeddings)
            return True
            
        return False