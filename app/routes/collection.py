"""Collection management routes blueprint."""

from flask import Blueprint, jsonify, request

from app.models.database import get_db
from app.routes.auth import get_authenticated_user

collection_bp = Blueprint("collection", __name__)


@collection_bp.route("/collection", methods=["GET"])
def get_collection():
    """Get user's card collection."""
    user = get_authenticated_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT card_key, name, image_url, count
            FROM user_collection
            WHERE user_sub = ?
            ORDER BY updated_at DESC
            """,
            (user["sub"],),
        ).fetchall()

    return jsonify({"cards": [dict(row) for row in rows]}), 200


@collection_bp.route("/collection", methods=["POST"])
def save_collection():
    """Save user's card collection."""
    user = get_authenticated_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    cards = payload.get("cards")

    if not isinstance(cards, list):
        return jsonify({"error": "Invalid cards payload"}), 400

    normalized_cards = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_key = card.get("key") or card.get("card_key")
        if not card_key:
            continue
        count = card.get("count")
        try:
            count = int(count) if count is not None else 1
        except (TypeError, ValueError):
            count = 1
        if count <= 0:
            continue
        normalized_cards.append(
            {
                "card_key": str(card_key),
                "name": card.get("name"),
                "image_url": card.get("image_url"),
                "count": count,
            }
        )

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
        conn.execute("DELETE FROM user_collection WHERE user_sub = ?", (user["sub"],))
        for card in normalized_cards:
            conn.execute(
                """
                INSERT INTO user_collection (user_sub, card_key, name, image_url, count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user["sub"],
                    card["card_key"],
                    card["name"],
                    card["image_url"],
                    card["count"],
                ),
            )

    return jsonify({"saved": len(normalized_cards)}), 200
