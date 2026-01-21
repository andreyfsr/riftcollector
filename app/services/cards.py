"""Card-related business logic."""

import json

from card_matcher import (
    CARDS_WEBP_DIR,
    build_hash_cache,
    load_cards_collection,
    load_cards_index,
)

# Set ordering for card sorting
SET_ORDER = {
    "Origins": 0,
    "Proving Grounds": 1,
    "Spiritforged": 2,
    "OGN": 0,
    "OGS": 1,
    "SFD": 2,
}


class CardService:
    """Service for card-related operations."""

    @staticmethod
    def normalize_set_label(set_label, set_id):
        """Normalize set labels for consistency."""
        if (set_label or "").strip().upper() == "SFD" or (set_id or "").strip().upper() == "SFD":
            return "Spiritforged"
        return set_label

    @staticmethod
    def _sort_collector_number(value):
        """Sort collector numbers, handling non-numeric values."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 999999

    @classmethod
    def get_card_count(cls):
        """Get total number of cards in the index."""
        return len(load_cards_index())

    @classmethod
    def get_image_count(cls):
        """Get total number of images in the hash cache."""
        return len(build_hash_cache())

    @classmethod
    def build_cards_payload(cls):
        """Build the cards payload for rendering."""
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
            set_label = cls.normalize_set_label(card.get("set_label"), card.get("set_id"))
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
                cls._sort_collector_number(card.get("collector_number")),
                card.get("name") or "",
            )
        )

        return cards_payload

    @classmethod
    def get_cards_json(cls):
        """Get cards payload as JSON string."""
        return json.dumps(cls.build_cards_payload())
