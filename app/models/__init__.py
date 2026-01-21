"""Database models package."""

from app.models.database import get_db, ensure_collection_schema

__all__ = ["get_db", "ensure_collection_schema"]
