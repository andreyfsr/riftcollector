"""Page routes blueprint."""

import os

from flask import Blueprint, render_template, request, send_from_directory

from card_matcher import CARDS_WEBP_DIR
from app.services.cards import CardService

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Home page with scanner."""
    return render_template(
        "index.html",
        card_count=CardService.get_card_count(),
        image_count=CardService.get_image_count(),
        active_page="scan",
    )


@pages_bp.route("/cards")
def cards():
    """Card gallery page."""
    return render_template(
        "cards.html",
        card_count=CardService.get_card_count(),
        image_count=CardService.get_image_count(),
        cards=CardService.get_cards_json(),
        active_page="gallery",
    )


@pages_bp.route("/library")
def library():
    """User library page."""
    return render_template(
        "library.html",
        card_count=CardService.get_card_count(),
        image_count=CardService.get_image_count(),
        cards=CardService.get_cards_json(),
        active_page="library",
    )


@pages_bp.route("/live-scan")
def live_scan():
    """Live scanning page."""
    return render_template(
        "live_scan.html",
        card_count=CardService.get_card_count(),
        image_count=CardService.get_image_count(),
        active_page="live",
    )


@pages_bp.route("/profile")
def profile():
    """User profile page."""
    app_origin = os.getenv("APP_ORIGIN")
    if not app_origin:
        app_origin = request.url_root.rstrip("/")
    else:
        app_origin = app_origin.rstrip("/")

    return render_template(
        "profile.html",
        card_count=CardService.get_card_count(),
        image_count=CardService.get_image_count(),
        active_page="profile",
        google_client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        app_origin=app_origin,
    )


@pages_bp.route("/cards_webp/<path:filename>")
def cards_webp(filename):
    """Serve card images with caching."""
    response = send_from_directory(CARDS_WEBP_DIR, filename)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@pages_bp.after_app_request
def add_static_cache_headers(response):
    """Add cache headers to static files."""
    if request.path.startswith("/static/"):
        response.headers.setdefault(
            "Cache-Control", "public, max-age=31536000, immutable"
        )
    return response
