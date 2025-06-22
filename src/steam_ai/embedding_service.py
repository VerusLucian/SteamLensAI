import logging
from typing import List, Dict, Any, Optional, Generator
import numpy as np
from dataclasses import dataclass
import time
import gc

from .config import Config
from .http_client import SyncHTTPClient

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBatch:
    """Container for a batch of embeddings with metadata."""
    texts: List[str]
    embeddings: List[List[float]]
    batch_id: int
    processing_time: float


class EmbeddingService:
    """Service for generating embeddings with batch processing and memory optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.http_client = SyncHTTPClient(
            pool_size=config.connection_pool_size,
            timeout=config.ollama_timeout_embed
        )
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts with batch processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        if not texts:
            return []
            
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[cache_key]))
                self._cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_misses += 1
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = self._generate_embeddings_batch(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._embedding_cache[cache_key] = embedding
        
        # Reconstruct full embeddings list
        result_embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        
        # Fill cached embeddings
        for i, embedding in cached_embeddings:
            result_embeddings[i] = embedding
            
        # Fill new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            result_embeddings[i] = embedding
            
        return result_embeddings
    
    def get_embeddings_streaming(self, 
                                texts: List[str],
                                batch_size: Optional[int] = None) -> Generator[EmbeddingBatch, None, None]:
        """
        Generate embeddings in streaming fashion for memory optimization.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of each batch (defaults to config value)
            
        Yields:
            EmbeddingBatch: Batches of embeddings
        """
        if not texts:
            return
            
        batch_size = batch_size or self.config.embedding_batch_size
        
        for batch_id, i in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            start_time = time.time()
            batch_embeddings = self.get_embeddings_batch(batch_texts)
            processing_time = time.time() - start_time
            
            yield EmbeddingBatch(
                texts=batch_texts,
                embeddings=batch_embeddings,
                batch_id=batch_id,
                processing_time=processing_time
            )
            
            # Force garbage collection between batches
            gc.collect()
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0] if embeddings else []
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Ollama API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: Generated embeddings
        """
        if not texts:
            return []
            
        try:
            payload = {
                "model": self.config.embed_model,
                "input": texts
            }
            
            response = self.http_client.post(
                self.config.ollama_embed_url,
                json_data=payload
            )
            
            embeddings = response.get("embeddings", [])
            
            if len(embeddings) != len(texts):
                logger.warning(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                # Pad with zeros if needed
                while len(embeddings) < len(texts):
                    embeddings.append([0.0] * (len(embeddings[0]) if embeddings else 384))
                    
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            fallback_dim = 384  # Default dimension for bge-m3
            return [[0.0] * fallback_dim for _ in texts]
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            str: Cache key
        """
        # Use hash of text with model name for cache key
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self.config.embed_model}:{text_hash}"
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        gc.collect()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._embedding_cache)
        }
    
    def precompute_embeddings(self, 
                            texts: List[str], 
                            show_progress: bool = True) -> Dict[str, List[float]]:
        """
        Precompute embeddings for a list of texts and store in cache.
        
        Args:
            texts: List of texts to precompute
            show_progress: Whether to show progress bar
            
        Returns:
            Dict[str, List[float]]: Mapping of cache keys to embeddings
        """
        if show_progress:
            from tqdm import tqdm
            tqdm(texts, desc="Precomputing embeddings")
            
        result = {}
        
        # Process in batches
        batch_size = self.config.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings_batch(batch_texts)
            
            for text, embedding in zip(batch_texts, batch_embeddings):
                cache_key = self._get_cache_key(text)
                result[cache_key] = embedding
                
        return result
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate that embeddings are properly formatted.
        
        Args:
            embeddings: List of embeddings to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not embeddings:
            return False
            
        # Check if all embeddings have the same dimension
        first_dim = len(embeddings[0]) if embeddings[0] else 0
        
        for embedding in embeddings:
            if not isinstance(embedding, list):
                return False
            if len(embedding) != first_dim:
                return False
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
                
        return True
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings for the current model.
        
        Returns:
            Optional[int]: Embedding dimension or None if unknown
        """
        # Try to get dimension from a test embedding
        try:
            test_embedding = self.get_single_embedding("test")
            return len(test_embedding) if test_embedding else None
        except Exception:
            return None
    
    def optimize_memory(self) -> None:
        """Optimize memory usage by clearing caches and forcing garbage collection."""
        # Keep only most recent cache entries if cache is too large
        max_cache_size = 10000  # Configurable limit
        
        if len(self._embedding_cache) > max_cache_size:
            # Remove oldest entries (simple FIFO, could be improved with LRU)
            items = list(self._embedding_cache.items())
            self._embedding_cache = dict(items[-max_cache_size//2:])
            
        gc.collect()
    
    def estimate_memory_usage(self, num_embeddings: int, embedding_dim: int) -> Dict[str, float]:
        """
        Estimate memory usage for given number of embeddings.
        
        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Dimension of each embedding
            
        Returns:
            Dict[str, float]: Memory usage estimates in MB
        """
        # Estimate memory usage
        bytes_per_float = 4  # float32
        embedding_memory = num_embeddings * embedding_dim * bytes_per_float
        
        # Add overhead for Python objects and caching
        overhead_factor = 1.5
        total_memory = embedding_memory * overhead_factor
        
        return {
            "embeddings_mb": embedding_memory / (1024 * 1024),
            "total_estimated_mb": total_memory / (1024 * 1024),
            "cache_mb": len(self._embedding_cache) * embedding_dim * bytes_per_float * overhead_factor / (1024 * 1024)
        }


class EmbeddingManager:
    """High-level manager for embedding operations with optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.service = EmbeddingService(config)
        
    def process_texts_optimized(self, 
                              texts: List[str],
                              enable_streaming: bool = True) -> List[List[float]]:
        """
        Process texts with memory optimization.
        
        Args:
            texts: List of texts to process
            enable_streaming: Whether to use streaming processing
            
        Returns:
            List[List[float]]: Processed embeddings
        """
        if not enable_streaming or len(texts) <= self.config.embedding_batch_size:
            # Process all at once for small datasets
            return self.service.get_embeddings_batch(texts)
        
        # Use streaming for large datasets
        all_embeddings = []
        
        for batch in self.service.get_embeddings_streaming(texts):
            all_embeddings.extend(batch.embeddings)
            
            # Log progress
            logger.info(f"Processed batch {batch.batch_id} with {len(batch.texts)} texts "
                       f"in {batch.processing_time:.2f}s")
        
        return all_embeddings
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        cache_stats = self.service.get_cache_stats()
        
        return {
            "cache_stats": cache_stats,
            "config": {
                "batch_size": self.config.embedding_batch_size,
                "model": self.config.embed_model,
                "timeout": self.config.ollama_timeout_embed
            }
        }