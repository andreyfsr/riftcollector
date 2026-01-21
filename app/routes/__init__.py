"""Routes package - Blueprint registration."""

from app.routes.pages import pages_bp
from app.routes.auth import auth_bp
from app.routes.collection import collection_bp
from app.routes.match import match_bp


def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(pages_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(collection_bp)
    app.register_blueprint(match_bp)
