import sqlite3
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent
CARDS_PATH = APP_ROOT / "cards.txt"
CARDS_DB_PATH = APP_ROOT / "cards.sqlite"
CARDS_WEBP_DIR = APP_ROOT / "cards_webp"
CACHE_PATH = APP_ROOT / "hash_cache.npz"
CACHE_VERSION = 1

HASH_SIZE = 8
MAX_DISTANCE = 300
WEAK_DISTANCE = 260
MIN_GAP = 6
TOP_K = 5
ORB_TOP_N = 60
RGB_FEATURE_SIZE = (80, 112)

_card_index = None
_card_collection = None
_hash_cache = None
_orb = cv2.ORB_create(800)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_card(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))

        if max_width < 200 or max_height < 200:
            continue

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))
        return warped

    return frame


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((300, 420), Image.BICUBIC)
    width, height = image.size
    pad = int(min(width, height) * 0.04)
    if pad > 0:
        image = image.crop((pad, pad, width - pad, height - pad))
    return image


def crop_art_region(image):
    width, height = image.size
    left = int(width * 0.08)
    right = int(width * 0.92)
    top = int(height * 0.12)
    bottom = int(height * 0.82)
    return image.crop((left, top, right, bottom))


def rotated_variants(image):
    return [
        image,
        image.rotate(90, expand=True),
        image.rotate(180, expand=True),
        image.rotate(270, expand=True),
    ]


def compute_hashes(image):
    phash = imagehash.phash(image, hash_size=HASH_SIZE)
    dhash = imagehash.dhash(image, hash_size=HASH_SIZE)
    return phash, dhash


def compute_colorhash(image):
    return imagehash.colorhash(image)


def prepare_similarity_features(image):
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 88), interpolation=cv2.INTER_AREA)
    gray = gray.astype("float32") / 255.0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return gray, hist


def prepare_rgb_feature(image):
    rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(
        rgb, RGB_FEATURE_SIZE, interpolation=cv2.INTER_AREA
    ).astype("float32")
    return resized / 255.0


def compute_orb_descriptors(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = _orb.detectAndCompute(gray, None)
    return descriptors


def _open_cards_db():
    if not CARDS_DB_PATH.exists():
        return None
    conn = sqlite3.connect(CARDS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_cards_index():
    global _card_index
    if _card_index is not None:
        return _card_index

    index = {}
    conn = _open_cards_db()
    if conn is None:
        _card_index = index
        return index

    rows = conn.execute(
        """
        SELECT
            riftbound_id,
            name,
            public_code,
            tcgplayer_id,
            collector_number,
            type,
            supertype,
            rarity,
            orientation,
            energy,
            might,
            power,
            text_plain,
            text_rich,
            image_url,
            artist,
            accessibility_text,
            set_id,
            clean_name,
            alternate_art,
            overnumbered,
            signature
        FROM cards
        """
    ).fetchall()
    for row in rows:
        riftbound_id = row["riftbound_id"]
        if riftbound_id:
            index[riftbound_id] = dict(row)

    conn.close()
    _card_index = index
    return index


def load_cards_collection():
    global _card_collection
    if _card_collection is not None:
        return _card_collection

    conn = _open_cards_db()
    if conn is None:
        _card_collection = []
        return _card_collection

    card_rows = conn.execute(
        """
        SELECT
            c.id,
            c.riftbound_id,
            c.name,
            c.public_code,
            c.tcgplayer_id,
            c.collector_number,
            c.type,
            c.supertype,
            c.rarity,
            c.orientation,
            c.energy,
            c.might,
            c.power,
            c.text_plain,
            c.text_rich,
            c.image_url,
            c.artist,
            c.accessibility_text,
            c.set_id,
            s.label AS set_label,
            c.clean_name,
            c.alternate_art,
            c.overnumbered,
            c.signature
        FROM cards c
        LEFT JOIN sets s ON s.set_id = c.set_id
        """
    ).fetchall()
    domain_rows = conn.execute(
        """
        SELECT cd.card_id, d.name
        FROM card_domains cd
        JOIN domains d ON d.id = cd.domain_id
        ORDER BY d.name
        """
    ).fetchall()
    conn.close()

    domain_map = {}
    for row in domain_rows:
        domain_map.setdefault(row["card_id"], []).append(row["name"])

    cards = []
    for row in card_rows:
        cards.append(
            {
                "riftbound_id": row["riftbound_id"],
                "name": row["name"],
                "public_code": row["public_code"],
                "tcgplayer_id": row["tcgplayer_id"],
                "collector_number": row["collector_number"],
                "type": row["type"],
                "supertype": row["supertype"],
                "rarity": row["rarity"],
                "orientation": row["orientation"],
                "energy": row["energy"],
                "might": row["might"],
                "power": row["power"],
                "text_plain": row["text_plain"],
                "text_rich": row["text_rich"],
                "image_url": row["image_url"],
                "artist": row["artist"],
                "accessibility_text": row["accessibility_text"],
                "set_id": row["set_id"],
                "set_label": row["set_label"],
                "clean_name": row["clean_name"],
                "alternate_art": row["alternate_art"],
                "overnumbered": row["overnumbered"],
                "signature": row["signature"],
                "domains": domain_map.get(row["id"], []),
            }
        )

    _card_collection = cards
    return cards


def build_hash_cache():
    global _hash_cache
    if _hash_cache is not None:
        return _hash_cache

    if not CARDS_WEBP_DIR.exists():
        _hash_cache = []
        return _hash_cache

    image_paths = _list_card_images()
    cached = _load_hash_cache(image_paths)
    if cached is not None:
        _hash_cache = cached
        return cached

    cache = _compute_hash_cache(image_paths)
    _save_hash_cache(cache, image_paths)
    _hash_cache = cache
    return cache


def _list_card_images():
    if not CARDS_WEBP_DIR.exists():
        return []
    images = [
        path
        for path in CARDS_WEBP_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    return sorted(images, key=lambda path: path.name)


def _compute_hash_cache(image_paths):
    cache = []
    for image_path in image_paths:
        try:
            image = preprocess_image(Image.open(image_path))
        except Exception:
            continue
        full_hashes = compute_hashes(image)
        full_rgb = prepare_rgb_feature(image)
        full_gray, full_hist = prepare_similarity_features(image)
        art_image = crop_art_region(image)
        art_hashes = compute_hashes(art_image)
        art_color = compute_colorhash(art_image)
        art_gray, art_hist = prepare_similarity_features(art_image)
        art_rgb = prepare_rgb_feature(art_image)
        art_desc = compute_orb_descriptors(art_image)
        cache.append(
            {
                "riftbound_id": image_path.stem,
                "filename": image_path.name,
                "full_hashes": full_hashes,
                "full_rgb": full_rgb,
                "full_gray": full_gray,
                "full_hist": full_hist,
                "art_hashes": art_hashes,
                "art_color": art_color,
                "art_gray": art_gray,
                "art_hist": art_hist,
                "art_rgb": art_rgb,
                "art_desc": art_desc,
            }
        )
    return cache


def _hash_to_array(hash_obj):
    return hash_obj.hash.astype(np.uint8)


def _array_to_hash(hash_array):
    return imagehash.ImageHash(hash_array.astype(bool))


def _cache_manifest(image_paths):
    names = [path.name for path in image_paths]
    mtimes = [path.stat().st_mtime_ns for path in image_paths]
    return names, mtimes


def _load_hash_cache(image_paths):
    if not CACHE_PATH.exists():
        return None

    try:
        data = np.load(CACHE_PATH, allow_pickle=True)
    except Exception:
        return None

    if int(data.get("version", [0])[0]) != CACHE_VERSION:
        return None

    names, mtimes = _cache_manifest(image_paths)
    if list(data.get("files", [])) != names:
        return None
    if list(data.get("mtimes", [])) != mtimes:
        return None

    try:
        full_phash = data["full_phash"]
        full_dhash = data["full_dhash"]
        art_phash = data["art_phash"]
        art_dhash = data["art_dhash"]
        art_color = data["art_color"]
        full_rgb = data["full_rgb"]
        full_gray = data["full_gray"]
        full_hist = data["full_hist"]
        art_gray = data["art_gray"]
        art_hist = data["art_hist"]
        art_rgb = data["art_rgb"]
        art_desc = data["art_desc"]
        riftbound_ids = data["riftbound_ids"]
        filenames = data["filenames"]
    except KeyError:
        return None

    cache = []
    for index in range(len(riftbound_ids)):
        cache.append(
            {
                "riftbound_id": str(riftbound_ids[index]),
                "filename": str(filenames[index]),
                "full_hashes": (
                    _array_to_hash(full_phash[index]),
                    _array_to_hash(full_dhash[index]),
                ),
                "full_rgb": full_rgb[index],
                "full_gray": full_gray[index],
                "full_hist": full_hist[index],
                "art_hashes": (
                    _array_to_hash(art_phash[index]),
                    _array_to_hash(art_dhash[index]),
                ),
                "art_color": _array_to_hash(art_color[index]),
                "art_gray": art_gray[index],
                "art_hist": art_hist[index],
                "art_rgb": art_rgb[index],
                "art_desc": art_desc[index],
            }
        )
    return cache


def _save_hash_cache(cache, image_paths):
    if not cache:
        return

    names, mtimes = _cache_manifest(image_paths)
    np.savez_compressed(
        CACHE_PATH,
        version=np.array([CACHE_VERSION], dtype=np.int32),
        files=np.array(names),
        mtimes=np.array(mtimes, dtype=np.int64),
        riftbound_ids=np.array([item["riftbound_id"] for item in cache]),
        filenames=np.array([item["filename"] for item in cache]),
        full_phash=np.stack(
            [_hash_to_array(item["full_hashes"][0]) for item in cache]
        ),
        full_dhash=np.stack(
            [_hash_to_array(item["full_hashes"][1]) for item in cache]
        ),
        art_phash=np.stack(
            [_hash_to_array(item["art_hashes"][0]) for item in cache]
        ),
        art_dhash=np.stack(
            [_hash_to_array(item["art_hashes"][1]) for item in cache]
        ),
        art_color=np.stack(
            [_hash_to_array(item["art_color"]) for item in cache]
        ),
        full_rgb=np.stack([item["full_rgb"] for item in cache]),
        full_gray=np.stack([item["full_gray"] for item in cache]),
        full_hist=np.stack([item["full_hist"] for item in cache]),
        art_gray=np.stack([item["art_gray"] for item in cache]),
        art_hist=np.stack([item["art_hist"] for item in cache]),
        art_rgb=np.stack([item["art_rgb"] for item in cache]),
        art_desc=np.array([item["art_desc"] for item in cache], dtype=object),
    )


def _map_variants(image, transform):
    return [transform(variant) for variant in rotated_variants(image)]


def find_best_match(image):
    hash_cache = build_hash_cache()
    if not hash_cache:
        return None

    target_full = preprocess_image(image)
    target_art = crop_art_region(target_full)
    target_desc = compute_orb_descriptors(target_art)
    target_full_variants = _map_variants(target_full, compute_hashes)
    target_art_variants = _map_variants(target_art, compute_hashes)
    target_art_color_variants = _map_variants(target_art, compute_colorhash)
    target_feature_variants = _map_variants(target_art, prepare_similarity_features)
    target_full_rgb_variants = _map_variants(target_full, prepare_rgb_feature)
    target_art_rgb_variants = _map_variants(target_art, prepare_rgb_feature)
    target_full_gray_variants = _map_variants(target_full, prepare_similarity_features)
    candidates = []
    for item in hash_cache:
        best_full = min(
            (p - item["full_hashes"][0]) + (d - item["full_hashes"][1])
            for p, d in target_full_variants
        )
        best_art = min(
            (p - item["art_hashes"][0]) + (d - item["art_hashes"][1])
            for p, d in target_art_variants
        )
        best_color = min(
            color - item["art_color"] for color in target_art_color_variants
        )
        best_gray = min(
            float(np.mean(np.abs(gray - item["art_gray"]))) * 100.0
            for gray, _ in target_feature_variants
        )
        best_hist = min(
            float(cv2.compareHist(hist, item["art_hist"], cv2.HISTCMP_BHATTACHARYYA))
            * 50.0
            for _, hist in target_feature_variants
        )
        best_full_rgb = min(
            float(np.mean(np.abs(rgb - item["full_rgb"]))) * 100.0
            for rgb in target_full_rgb_variants
        )
        best_art_rgb = min(
            float(np.mean(np.abs(rgb - item["art_rgb"]))) * 100.0
            for rgb in target_art_rgb_variants
        )
        best_full_hist = min(
            float(
                cv2.compareHist(
                    hist, item["full_hist"], cv2.HISTCMP_BHATTACHARYYA
                )
            )
            * 50.0
            for _, hist in target_full_gray_variants
        )
        distance = (
            best_full
            + best_art
            + (best_color * 2)
            + best_gray
            + best_hist
            + best_full_rgb
            + best_art_rgb
            + best_full_hist
        )
        candidates.append((distance, item))

    candidates.sort(key=lambda entry: entry[0])
    best_distance, best_match = candidates[0]

    if target_desc is not None and len(candidates) > 1:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        orb_scored = []
        for distance, item in candidates:
            desc = item.get("art_desc")
            if desc is None:
                continue
            matches = list(bf.knnMatch(target_desc, desc, k=2))
            if not matches:
                continue
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                first, second = pair
                if first.distance < 0.75 * second.distance:
                    good.append(first)
            if not good:
                continue
            avg_distance = float(np.mean([match.distance for match in good]))
            orb_scored.append((len(good), avg_distance, distance, item))

        if orb_scored:
            orb_scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
            best_matches, best_avg, best_distance, best_match = orb_scored[0]

    return {
        "riftbound_id": best_match["riftbound_id"],
        "filename": best_match["filename"],
        "distance": best_distance,
        "candidates": [
            {
                "riftbound_id": item["riftbound_id"],
                "filename": item["filename"],
                "distance": distance,
            }
            for distance, item in candidates[:TOP_K]
        ],
    }
