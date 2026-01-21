"""
Flask Application Factory.

This module creates and configures the Flask application using the factory pattern,
following Flask best practices for scalable application structure.
"""

from flask import Flask

from app.config import get_config
from app.extensions import init_extensions
from app.models.database import init_db
from app.routes import register_blueprints


def create_app(config_class=None):
    """
    Create and configure the Flask application.

    Args:
        config_class: Configuration class to use. Defaults to auto-detection
                     based on FLASK_ENV environment variable.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Initialize extensions
    init_extensions(app)

    # Initialize database
    init_db(app)

    # Register blueprints
    register_blueprints(app)

    return app
