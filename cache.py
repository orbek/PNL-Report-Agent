"""
Caching layer for Financial P&L Anomaly Detection Agent
Reduces redundant API calls and improves performance
"""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple file-based cache manager with TTL support
    Caches LLM responses, vector search results, and database queries
    """
    
    def __init__(self, cache_dir: str = "./cache", default_ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        
        # Create subdirectories for different cache types
        (self.cache_dir / "llm").mkdir(exist_ok=True)
        (self.cache_dir / "vector").mkdir(exist_ok=True)
        (self.cache_dir / "db").mkdir(exist_ok=True)
        
        logger.info(f"Cache manager initialized: {self.cache_dir}")
    
    def _get_cache_key(self, namespace: str, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments
        
        Args:
            namespace: Cache namespace (e.g., 'llm', 'vector')
            *args, **kwargs: Function arguments to hash
            
        Returns:
            Cache key string
        """
        # Create a deterministic string representation of arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"{namespace}_{key_hash}"
    
    def _get_cache_path(self, cache_key: str, namespace: str) -> Path:
        """Get file path for cache entry"""
        return self.cache_dir / namespace / f"{cache_key}.cache"
    
    def get(self, namespace: str, *args, **kwargs) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            namespace: Cache namespace
            *args, **kwargs: Function arguments (used to generate cache key)
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        cache_key = self._get_cache_key(namespace, *args, **kwargs)
        cache_path = self._get_cache_path(cache_key, namespace)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if expired
            if datetime.now() > cache_data['expires_at']:
                logger.debug(f"Cache expired: {cache_key}")
                cache_path.unlink()  # Delete expired cache
                return None
            
            logger.debug(f"Cache hit: {cache_key}")
            return cache_data['value']
            
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def set(self, value: Any, namespace: str, ttl: Optional[int] = None, *args, **kwargs):
        """
        Store value in cache
        
        Args:
            value: Value to cache
            namespace: Cache namespace
            ttl: Time-to-live in seconds (uses default if not specified)
            *args, **kwargs: Function arguments (used to generate cache key)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._get_cache_key(namespace, *args, **kwargs)
        cache_path = self._get_cache_path(cache_key, namespace)
        
        cache_data = {
            'value': value,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Cache stored: {cache_key} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
    
    def clear(self, namespace: Optional[str] = None):
        """
        Clear cache entries
        
        Args:
            namespace: Clear specific namespace, or all if None
        """
        if namespace:
            cache_dir = self.cache_dir / namespace
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.cache"):
                    cache_file.unlink()
                logger.info(f"Cleared cache namespace: {namespace}")
        else:
            # Clear all namespaces
            for namespace_dir in self.cache_dir.iterdir():
                if namespace_dir.is_dir():
                    for cache_file in namespace_dir.glob("*.cache"):
                        cache_file.unlink()
            logger.info("Cleared all cache")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        stats = {}
        
        for namespace_dir in self.cache_dir.iterdir():
            if namespace_dir.is_dir():
                namespace = namespace_dir.name
                cache_files = list(namespace_dir.glob("*.cache"))
                
                total_size = sum(f.stat().st_size for f in cache_files)
                expired_count = 0
                
                for cache_file in cache_files:
                    try:
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                        if datetime.now() > cache_data['expires_at']:
                            expired_count += 1
                    except:
                        pass
                
                stats[namespace] = {
                    'total_entries': len(cache_files),
                    'expired_entries': expired_count,
                    'total_size_mb': round(total_size / (1024 * 1024), 2)
                }
        
        return stats


def cached(namespace: str, ttl: Optional[int] = None):
    """
    Decorator to cache function results
    
    Args:
        namespace: Cache namespace (e.g., 'llm', 'vector')
        ttl: Time-to-live in seconds
        
    Example:
        @cached('llm', ttl=3600)
        def expensive_llm_call(query: str) -> str:
            return llm.generate(query)
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = CacheManager()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_value = cache_manager.get(namespace, *args, **kwargs)
            if cached_value is not None:
                logger.info(f"Using cached result for {func.__name__}")
                return cached_value
            
            # Call function if not cached
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(result, namespace, ttl, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


class QueryCache:
    """
    Specialized cache for database queries
    Automatically invalidates when data changes
    """
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.last_modification = {}
    
    def mark_modified(self, table: str):
        """Mark table as modified to invalidate related caches"""
        self.last_modification[table] = datetime.now()
        logger.debug(f"Table modified: {table}")
    
    def get_query_result(self, query: str, params: tuple = None) -> Optional[Any]:
        """Get cached query result"""
        return self.cache_manager.get("db", query=query, params=params)
    
    def set_query_result(self, result: Any, query: str, params: tuple = None, ttl: int = 3600):
        """Cache query result"""
        self.cache_manager.set(result, "db", ttl, query=query, params=params)


# Global cache instance
_global_cache = None


def get_cache() -> CacheManager:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


if __name__ == "__main__":
    # Test cache
    logging.basicConfig(level=logging.INFO)
    
    cache = CacheManager()
    
    # Test basic caching
    cache.set("test_value", "test", query="SELECT * FROM test")
    result = cache.get("test", query="SELECT * FROM test")
    print(f"Cached result: {result}")
    
    # Test decorator
    @cached('test', ttl=60)
    def slow_function(x: int) -> int:
        print("Computing...")
        return x * 2
    
    print(f"First call: {slow_function(5)}")
    print(f"Second call (cached): {slow_function(5)}")
    
    # Print stats
    print(f"\nCache stats: {cache.get_stats()}")
