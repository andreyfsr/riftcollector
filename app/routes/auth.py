"""Authentication routes blueprint."""

import os
import sqlite3

from flask import Blueprint, jsonify, request, session
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from app.models.database import get_db

auth_bp = Blueprint("auth", __name__)


def get_authenticated_user():
    """Get the currently authenticated user from session."""
    user = session.get("user") or {}
    sub = user.get("sub")
    if not sub:
        return None
    return user


@auth_bp.route("/me")
def auth_me():
    """Get current user info."""
    user = session.get("user")
    if not user:
        return jsonify({"user": None}), 200
    return jsonify({"user": user}), 200


@auth_bp.route("/logout", methods=["POST"])
def auth_logout():
    """Log out the current user."""
    session.pop("user", None)
    return jsonify({"success": True}), 200


@auth_bp.route("/google", methods=["POST"])
def auth_google():
    """Authenticate with Google OAuth."""
    payload = request.get_json(silent=True) or {}
    credential = payload.get("credential")
    client_id = os.getenv("GOOGLE_CLIENT_ID")

    if not credential:
        return jsonify({"error": "Missing credential"}), 400
    if not client_id:
        return jsonify({"error": "Google client ID not configured"}), 400

    try:
        idinfo = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), client_id
        )
    except ValueError:
        return jsonify({"error": "Invalid Google credential"}), 400

    user = {
        "sub": idinfo.get("sub"),
        "name": idinfo.get("name"),
        "email": idinfo.get("email"),
        "picture": idinfo.get("picture"),
    }
    session["user"] = user

    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO users (sub, name, email, picture)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sub) DO UPDATE SET
                    name = excluded.name,
                    email = excluded.email,
                    picture = excluded.picture,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user.get("sub"), user.get("name"), user.get("email"), user.get("picture")),
            )
    except sqlite3.Error:
        pass

    return jsonify({"user": user}), 200
