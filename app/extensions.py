"""Flask extensions initialization."""

from flask_compress import Compress

# Initialize extensions without app
compress = Compress()


def init_extensions(app):
    """Initialize Flask extensions with the app instance."""
    compress.init_app(app)
