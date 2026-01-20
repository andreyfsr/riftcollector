import base64
import io
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, session
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from PIL import Image
import cv2
import numpy as np
from flask_compress import Compress

from card_matcher import (
    CARDS_WEBP_DIR,
    MAX_DISTANCE,
    MIN_GAP,
    WEAK_DISTANCE,
    build_hash_cache,
    extract_card,
    find_best_match,
    load_cards_collection,
    load_cards_index,
    preprocess_image,
)
from card_detector import CardDetector

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.config["COMPRESS_LEVEL"] = 6
app.config["COMPRESS_MIN_SIZE"] = 512
Compress(app)

APP_ROOT = Path(__file__).resolve().parent
DB_PATH = APP_ROOT / "cards.sqlite"
_card_detector = None


def _get_card_detector():
    """Get or create the card detector instance."""
    global _card_detector
    if _card_detector is not None:
        return _card_detector

    # Read configuration from environment
    model_path_str = os.getenv("DETECTOR_MODEL_PATH")
    model_path = Path(model_path_str) if model_path_str else None

    confidence = os.getenv("DETECTOR_CONFIDENCE")
    try:
        confidence = float(confidence) if confidence else 0.5
    except ValueError:
        confidence = 0.5

    max_detections = os.getenv("DETECTOR_MAX_DETECTIONS")
    try:
        max_detections = int(max_detections) if max_detections else 5
    except ValueError:
        max_detections = 5

    min_card_area = os.getenv("DETECTOR_MIN_CARD_AREA")
    try:
        min_card_area = int(min_card_area) if min_card_area else 5000
    except ValueError:
        min_card_area = 5000

    aspect_ratio_min = os.getenv("DETECTOR_ASPECT_MIN")
    aspect_ratio_max = os.getenv("DETECTOR_ASPECT_MAX")
    try:
        aspect_ratio_min = float(aspect_ratio_min) if aspect_ratio_min else 1.2
    except ValueError:
        aspect_ratio_min = 1.2
    try:
        aspect_ratio_max = float(aspect_ratio_max) if aspect_ratio_max else 1.8
    except ValueError:
        aspect_ratio_max = 1.8

    _card_detector = CardDetector(
        model_path=model_path,
        confidence_threshold=confidence,
        max_detections=max_detections,
        min_card_area=min_card_area,
        aspect_ratio_range=(aspect_ratio_min, aspect_ratio_max),
        use_hash_matching=True,
    )
    return _card_detector


def _ensure_collection_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            sub TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            picture TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS user_collection (
            user_sub TEXT NOT NULL,
            card_key TEXT NOT NULL,
            name TEXT,
            image_url TEXT,
            count INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_sub, card_key),
            FOREIGN KEY (user_sub) REFERENCES users(sub) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_collection_user
            ON user_collection(user_sub);
        """
    )


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    _ensure_collection_schema(conn)
    return conn


def normalize_set_label(set_label, set_id):
    if (set_label or "").strip().upper() == "SFD" or (set_id or "").strip().upper() == "SFD":
        return "Spiritforged"
    return set_label


def _sort_collector_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 999999


SET_ORDER = {
    "Origins": 0,
    "Proving Grounds": 1,
    "Spiritforged": 2,
    "OGN": 0,
    "OGS": 1,
    "SFD": 2,
}


def _build_cards_payload():
    webp_index = set()
    if CARDS_WEBP_DIR.exists():
        webp_index = {
            path.stem.lower()
            for path in CARDS_WEBP_DIR.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }

    cards_payload = []
    for card in load_cards_collection():
        riftbound_id = card.get("riftbound_id")
        if not riftbound_id:
            continue
        set_label = normalize_set_label(card.get("set_label"), card.get("set_id"))
        normalized_id = riftbound_id.lower()
        if normalized_id in webp_index:
            image_url = f"/cards_webp/{normalized_id}.webp"
        else:
            image_url = card.get("image_url")
        cards_payload.append(
            {
                "id": riftbound_id,
                "name": card.get("name"),
                "public_code": card.get("public_code"),
                "collector_number": card.get("collector_number"),
                "type": card.get("type"),
                "supertype": card.get("supertype"),
                "rarity": card.get("rarity"),
                "domains": card.get("domains") or [],
                "energy": card.get("energy"),
                "power": card.get("power"),
                "might": card.get("might"),
                "text": card.get("text_plain"),
                "set_label": set_label,
                "set_id": card.get("set_id"),
                "artist": card.get("artist"),
                "image_url": image_url,
            }
        )

    cards_payload.sort(
        key=lambda card: (
            SET_ORDER.get(
                (card.get("set_label") or "").strip(),
                SET_ORDER.get((card.get("set_id") or "").strip(), 99),
            ),
            _sort_collector_number(card.get("collector_number")),
            card.get("name") or "",
        )
    )

    return cards_payload


@app.route("/")
def index():
    card_count = len(load_cards_index())
    image_count = len(build_hash_cache())
    return render_template(
        "index.html",
        card_count=card_count,
        image_count=image_count,
        active_page="scan",
    )


@app.route("/cards")
def cards():
    card_count = len(load_cards_index())
    image_count = len(build_hash_cache())
    cards_payload = _build_cards_payload()
    return render_template(
        "cards.html",
        card_count=card_count,
        image_count=image_count,
        cards=json.dumps(cards_payload),
        active_page="gallery",
    )


@app.route("/library")
def library():
    card_count = len(load_cards_index())
    image_count = len(build_hash_cache())
    cards_payload = _build_cards_payload()

    return render_template(
        "library.html",
        card_count=card_count,
        image_count=image_count,
        cards=json.dumps(cards_payload),
        active_page="library",
    )


@app.route("/live-scan")
def live_scan():
    card_count = len(load_cards_index())
    image_count = len(build_hash_cache())
    return render_template(
        "live_scan.html",
        card_count=card_count,
        image_count=image_count,
        active_page="live",
    )


@app.route("/profile")
def profile():
    card_count = len(load_cards_index())
    image_count = len(build_hash_cache())
    app_origin = os.getenv("APP_ORIGIN")
    if not app_origin:
        app_origin = request.url_root.rstrip("/")
    else:
        app_origin = app_origin.rstrip("/")
    return render_template(
        "profile.html",
        card_count=card_count,
        image_count=image_count,
        active_page="profile",
        google_client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        app_origin=app_origin,
    )


@app.route("/auth/me")
def auth_me():
    user = session.get("user")
    if not user:
        return jsonify({"user": None}), 200
    return jsonify({"user": user}), 200


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user", None)
    return jsonify({"success": True}), 200


@app.route("/auth/google", methods=["POST"])
def auth_google():
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
        with _get_db() as conn:
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


def _get_authenticated_user():
    user = session.get("user") or {}
    sub = user.get("sub")
    if not sub:
        return None
    return user


@app.route("/collection", methods=["GET"])
def get_collection():
    user = _get_authenticated_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    with _get_db() as conn:
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


@app.route("/collection", methods=["POST"])
def save_collection():
    user = _get_authenticated_user()
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

    with _get_db() as conn:
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


@app.route("/cards_webp/<path:filename>")
def cards_webp(filename):
    response = send_from_directory(CARDS_WEBP_DIR, filename)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@app.after_request
def add_static_cache_headers(response):
    if request.path.startswith("/static/"):
        response.headers.setdefault(
            "Cache-Control", "public, max-age=31536000, immutable"
        )
    return response


@app.route("/match", methods=["POST"])
def match_card():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    if "base64," in image_data:
        image_data = image_data.split("base64,", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image payload"}), 400

    card_frame = extract_card(image)
    card_image = Image.fromarray(cv2.cvtColor(card_frame, cv2.COLOR_BGR2RGB))
    card_image = preprocess_image(card_image)
    match = find_best_match(card_image)
    fallback_match = find_best_match(image)
    if fallback_match and (not match or fallback_match["distance"] < match["distance"]):
        match = fallback_match
    if not match:
        return jsonify({"error": "No match found"}), 404
    candidates = match.get("candidates") or []
    gap = None
    if len(candidates) > 1:
        gap = float(candidates[1]["distance"]) - float(candidates[0]["distance"])
    if match["distance"] > MAX_DISTANCE:
        return (
            jsonify(
                {
                    "error": "Match too weak",
                    "candidates": candidates,
                    "gap": gap,
                    "max_distance": MAX_DISTANCE,
                    "min_gap": MIN_GAP,
                    "weak_distance": WEAK_DISTANCE,
                }
            ),
            404,
        )
    if gap is not None and match["distance"] > WEAK_DISTANCE and gap < MIN_GAP:
        return (
            jsonify(
                {
                    "error": "Match too weak",
                    "candidates": candidates,
                    "gap": gap,
                    "max_distance": MAX_DISTANCE,
                    "min_gap": MIN_GAP,
                    "weak_distance": WEAK_DISTANCE,
                }
            ),
            404,
        )

    card = load_cards_index().get(match["riftbound_id"])
    return jsonify(
        {
            "riftbound_id": match["riftbound_id"],
            "name": card.get("name") if card else None,
            "distance": match["distance"],
            "image_url": f"/cards_webp/{match['filename']}",
            "candidates": candidates,
            "max_distance": MAX_DISTANCE,
            "cache_size": len(build_hash_cache()),
            "gap": gap,
            "min_gap": MIN_GAP,
            "weak_distance": WEAK_DISTANCE,
        }
    )


@app.route("/detect", methods=["POST"])
def detect_cards():
    detector = _get_card_detector()
    if detector is None:
        return jsonify({"error": "Detector not available"}), 503

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    if "base64," in image_data:
        image_data = image_data.split("base64,", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None

    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    detections = detector.detect(frame)
    serialized = []
    for det in detections:
        polygon = det.polygon if det.polygon is not None else []
        serialized.append(
            {
                "card_id": det.card_id,
                "confidence": det.confidence,
                "bbox": list(det.bbox),
                "polygon": [[float(x), float(y)] for x, y in polygon] if len(polygon) > 0 else [],
                "match_distance": det.match_distance,
            }
        )
    return jsonify({"detections": serialized}), 200


if __name__ == "__main__":
    app.run(debug=True)
