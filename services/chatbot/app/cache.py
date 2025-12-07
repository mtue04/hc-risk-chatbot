"""
Caching utilities for chatbot tools.

Provides Redis-based caching for expensive operations like model predictions
and database queries.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

import redis
import structlog
from functools import wraps

logger = structlog.get_logger()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_CACHE_DB = int(os.getenv("REDIS_CACHE_DB", "1"))  # Use DB 1 for caching

# Initialize Redis client
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client."""
    global _redis_client

    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_CACHE_DB,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Test connection
            _redis_client.ping()
            logger.info("redis_cache_connected", host=REDIS_HOST, port=REDIS_PORT, db=REDIS_CACHE_DB)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("redis_cache_unavailable", error=str(e))
            _redis_client = None

    return _redis_client


def cached(ttl: int = 3600, key_prefix: str = "cache"):
    """
    Decorator to cache function results in Redis.

    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        key_prefix: Prefix for cache keys

    Example:
        @cached(ttl=3600, key_prefix="prediction")
        def get_risk_prediction(applicant_id: int) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            client = get_redis_client()

            # If Redis unavailable, call function directly
            if client is None:
                logger.debug("cache_miss_no_redis", func=func.__name__)
                return func(*args, **kwargs)

            # Build cache key from function name and arguments
            # Convert args/kwargs to a deterministic string
            arg_str = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
            cache_key = f"{key_prefix}:{func.__name__}:{hash(arg_str)}"

            try:
                # Try to get from cache
                cached_value = client.get(cache_key)
                if cached_value is not None:
                    logger.debug("cache_hit", func=func.__name__, key=cache_key)
                    return json.loads(cached_value)

                # Cache miss - call function
                logger.debug("cache_miss", func=func.__name__, key=cache_key)
                result = func(*args, **kwargs)

                # Store in cache
                try:
                    client.setex(cache_key, ttl, json.dumps(result, default=str))
                    logger.debug("cache_set", func=func.__name__, key=cache_key, ttl=ttl)
                except (TypeError, ValueError) as e:
                    # If result isn't JSON serializable, don't cache
                    logger.warning("cache_set_failed", func=func.__name__, error=str(e))

                return result

            except redis.RedisError as e:
                logger.warning("cache_error", func=func.__name__, error=str(e))
                return func(*args, **kwargs)

        return wrapper
    return decorator


def invalidate_cache(key_pattern: str) -> int:
    """
    Invalidate cache entries matching a pattern.

    Args:
        key_pattern: Redis key pattern (e.g., "prediction:*")

    Returns:
        Number of keys deleted
    """
    client = get_redis_client()
    if client is None:
        return 0

    try:
        keys = client.keys(key_pattern)
        if keys:
            deleted = client.delete(*keys)
            logger.info("cache_invalidated", pattern=key_pattern, count=deleted)
            return deleted
        return 0
    except redis.RedisError as e:
        logger.error("cache_invalidation_failed", pattern=key_pattern, error=str(e))
        return 0
