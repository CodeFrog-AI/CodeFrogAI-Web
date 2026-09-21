"""Embedding generation and indexing for semantic code search."""

# Size of RepositoryChunk.embedding. Providers must return vectors of exactly this
# length; changing it requires a migration and re-embedding every chunk.
EMBEDDING_DIMENSIONS = 1536
