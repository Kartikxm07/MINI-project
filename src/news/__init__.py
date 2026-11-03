"""News ingestion, embedding and date-alignment for the attention models."""

from src.news.providers import NewsItem, get_provider
from src.news.embeddings import get_embedder
from src.news.aggregate import (
    group_by_date,
    build_news_tensor,
    synthetic_news_embeddings,
)

__all__ = [
    "NewsItem",
    "get_provider",
    "get_embedder",
    "group_by_date",
    "build_news_tensor",
    "synthetic_news_embeddings",
]
