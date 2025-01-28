from cachetools import cached, TTLCache

vector_cache = TTLCache(maxsize=100, ttl=300)

# Vectorization with caching
@cached(vector_cache)
def cached_vectorize_input(user_input):
    """Cache vectorization to speed up repeated queries."""
    return vector_model.encode([user_input])[0]