"""Card matching service."""

import base64
import io
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from card_matcher import (
    MAX_DISTANCE,
    MIN_GAP,
    WEAK_DISTANCE,
    build_hash_cache,
    extract_card,
    find_best_match,
    load_cards_index,
    preprocess_image,
)

# Fast matcher cache (module-level singleton)
_fast_matcher_cache = None


class MatcherService:
    """Service for card matching operations."""

    APP_ROOT = Path(__file__).resolve().parent.parent.parent

    @classmethod
    def get_fast_matcher(cls):
        """Get or initialize the fast matcher cache."""
        global _fast_matcher_cache
        if _fast_matcher_cache is not None:
            return _fast_matcher_cache

        cache_path = cls.APP_ROOT / "hash_cache.npz"
        if not cache_path.exists():
            return None

        data = np.load(cache_path, allow_pickle=True)
        _fast_matcher_cache = {
            "riftbound_ids": data["riftbound_ids"],
            "filenames": data["filenames"],
            "full_phash": data["full_phash"],
            "full_dhash": data["full_dhash"],
        }
        return _fast_matcher_cache

    @staticmethod
    def decode_image(image_data):
        """Decode base64 image data to PIL Image."""
        if "base64," in image_data:
            image_data = image_data.split("base64,", 1)[1]

        raw_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    @classmethod
    def fast_match(cls, image_data, threshold=28):
        """
        Fast hash-based card matching for live scanning.

        Returns:
            dict: Match result with 'matched', 'riftbound_id', 'name', 'distance', 'image_url'
                  or 'error' on failure.
        """
        try:
            pil_image = cls.decode_image(image_data)
        except Exception:
            return {"error": "Invalid image payload"}

        # Resize to standard card size
        pil_image = pil_image.resize((300, 420), Image.BILINEAR)

        # Get fast matcher cache
        matcher = cls.get_fast_matcher()
        if matcher is None:
            return {"error": "Hash cache not available", "status": 503}

        # Compute hashes
        phash = imagehash.phash(pil_image, hash_size=8)
        dhash = imagehash.dhash(pil_image, hash_size=8)

        phash_arr = phash.hash.astype(np.uint8)
        dhash_arr = dhash.hash.astype(np.uint8)

        # Vectorized comparison
        phash_diff = np.sum(matcher["full_phash"] != phash_arr, axis=(1, 2))
        dhash_diff = np.sum(matcher["full_dhash"] != dhash_arr, axis=(1, 2))
        total_diff = phash_diff + dhash_diff

        # Find best match
        best_idx = np.argmin(total_diff)
        best_distance = int(total_diff[best_idx])

        if best_distance > threshold:
            return {"matched": False, "distance": best_distance}

        rid = str(matcher["riftbound_ids"][best_idx])
        filename = str(matcher["filenames"][best_idx])

        # Get card info
        card = load_cards_index().get(rid, {})

        return {
            "matched": True,
            "riftbound_id": rid,
            "name": card.get("name", rid),
            "distance": best_distance,
            "image_url": f"/cards_webp/{filename}",
        }

    @classmethod
    def match_card(cls, image_data):
        """
        Full card matching with extraction and multiple matching strategies.

        Returns:
            dict: Match result with card details and candidates, or 'error' on failure.
        """
        try:
            image = cls.decode_image(image_data)
        except Exception:
            return {"error": "Invalid image payload"}

        card_frame = extract_card(image)
        card_image = Image.fromarray(cv2.cvtColor(card_frame, cv2.COLOR_BGR2RGB))
        card_image = preprocess_image(card_image)
        match = find_best_match(card_image)
        fallback_match = find_best_match(image)

        if fallback_match and (not match or fallback_match["distance"] < match["distance"]):
            match = fallback_match

        if not match:
            return {"error": "No match found", "status": 404}

        candidates = match.get("candidates") or []
        gap = None
        if len(candidates) > 1:
            gap = float(candidates[1]["distance"]) - float(candidates[0]["distance"])

        if match["distance"] > MAX_DISTANCE:
            return {
                "error": "Match too weak",
                "candidates": candidates,
                "gap": gap,
                "max_distance": MAX_DISTANCE,
                "min_gap": MIN_GAP,
                "weak_distance": WEAK_DISTANCE,
                "status": 404,
            }

        if gap is not None and match["distance"] > WEAK_DISTANCE and gap < MIN_GAP:
            return {
                "error": "Match too weak",
                "candidates": candidates,
                "gap": gap,
                "max_distance": MAX_DISTANCE,
                "min_gap": MIN_GAP,
                "weak_distance": WEAK_DISTANCE,
                "status": 404,
            }

        card = load_cards_index().get(match["riftbound_id"])
        return {
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
