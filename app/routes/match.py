"""Card matching routes blueprint."""

from flask import Blueprint, jsonify, request

from app.services.matcher import MatcherService

match_bp = Blueprint("match", __name__)


@match_bp.route("/api/fast-match", methods=["POST"])
def fast_match():
    """Fast hash-based card matching for live scanning."""
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    threshold = payload.get("threshold", 28)

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    result = MatcherService.fast_match(image_data, threshold)

    if "error" in result:
        status = result.pop("status", 400)
        return jsonify(result), status

    return jsonify(result), 200


@match_bp.route("/match", methods=["POST"])
def match_card():
    """Full card matching with extraction."""
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    result = MatcherService.match_card(image_data)

    if "error" in result:
        status = result.pop("status", 404)
        return jsonify(result), status

    return jsonify(result), 200


@match_bp.route("/detect", methods=["POST"])
def detect_cards():
    """Legacy detect endpoint - deprecated."""
    return jsonify({"error": "Detector removed. Use /api/fast-match instead."}), 410
