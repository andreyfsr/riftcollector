"""Configuration classes for different environments."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""

    # Application paths
    APP_ROOT = Path(__file__).resolve().parent.parent
    DB_PATH = APP_ROOT / "cards.sqlite"

    # Flask settings
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")

    # Compression settings
    COMPRESS_LEVEL = 6
    COMPRESS_MIN_SIZE = 512

    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

    # App origin for OAuth
    APP_ORIGIN = os.getenv("APP_ORIGIN")


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    TESTING = False

    # Override with a strong secret key in production
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

    # Production-specific settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class TestingConfig(Config):
    """Testing configuration."""

    DEBUG = True
    TESTING = True
    DB_PATH = Config.APP_ROOT / "test_cards.sqlite"


# Configuration dictionary for easy access
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config():
    """Get configuration based on FLASK_ENV environment variable."""
    env = os.getenv("FLASK_ENV", "development")
    return config.get(env, config["default"])
