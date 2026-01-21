"""
Live Riftbound card detection using OpenCV and perceptual hashing.

Hold the card in the center of the frame to detect it.

Usage:
    python live_detect.py                # Default camera
    python live_detect.py --debug        # Show debug info
    python live_detect.py --help         # Show all options

Controls:
    q / ESC  - Quit
    s        - Save screenshot
    d        - Toggle debug view
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent
CACHE_PATH = APP_ROOT / "hash_cache.npz"
CARDS_DB_PATH = APP_ROOT / "cards.sqlite"


@dataclass
class Detection:
    """A detected and matched card."""
    riftbound_id: str
    name: str
    distance: int
    bbox: tuple[int, int, int, int]


class FastMatcher:
    """Fast card matcher using vectorized hash comparison."""

    def __init__(self, threshold: int = 25):
        self.threshold = threshold
        self.riftbound_ids = None
        self.filenames = None
        self.full_phash = None
        self.full_dhash = None
        self.card_names = {}

        self._load_cache()
        self._load_card_names()

    def _load_cache(self):
        """Load hash arrays directly from npz file."""
        if not CACHE_PATH.exists():
            print(f"[FastMatcher] Cache not found: {CACHE_PATH}")
            return

        print("[FastMatcher] Loading hash cache...")
        data = np.load(CACHE_PATH, allow_pickle=True)

        self.riftbound_ids = data["riftbound_ids"]
        self.filenames = data["filenames"]
        self.full_phash = data["full_phash"]
        self.full_dhash = data["full_dhash"]

        print(f"[FastMatcher] Loaded {len(self.riftbound_ids)} cards")

    def _load_card_names(self):
        """Load card names from database."""
        if not CARDS_DB_PATH.exists():
            return

        import sqlite3
        conn = sqlite3.connect(CARDS_DB_PATH)
        cursor = conn.execute("SELECT riftbound_id, name FROM cards")
        for row in cursor:
            self.card_names[row[0]] = row[1]
        conn.close()
        print(f"[FastMatcher] Loaded {len(self.card_names)} card names")

    def match(self, image: np.ndarray) -> dict | None:
        """Match an image against the hash cache."""
        if self.full_phash is None:
            return None

        # Convert BGR to RGB PIL image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Resize to standard card size
        pil_image = pil_image.resize((300, 420), Image.BILINEAR)

        # Compute hashes
        phash = imagehash.phash(pil_image, hash_size=8)
        dhash = imagehash.dhash(pil_image, hash_size=8)

        # Convert to numpy arrays
        phash_arr = phash.hash.astype(np.uint8)
        dhash_arr = dhash.hash.astype(np.uint8)

        # Vectorized hamming distance calculation
        phash_diff = np.sum(self.full_phash != phash_arr, axis=(1, 2))
        dhash_diff = np.sum(self.full_dhash != dhash_arr, axis=(1, 2))
        total_diff = phash_diff + dhash_diff

        # Find best match
        best_idx = np.argmin(total_diff)
        best_distance = int(total_diff[best_idx])

        if best_distance > self.threshold:
            return None

        rid = str(self.riftbound_ids[best_idx])
        return {
            "riftbound_id": rid,
            "filename": str(self.filenames[best_idx]),
            "distance": best_distance,
            "name": self.card_names.get(rid, rid),
        }


class CardScanner:
    """Live card scanner with ROI-based detection."""

    def __init__(
        self,
        threshold: int = 25,
        roi_scale: float = 0.7,
        detection_interval: float = 0.15,
    ):
        self.roi_scale = roi_scale
        self.detection_interval = detection_interval
        self.matcher = FastMatcher(threshold=threshold)

        self.last_detection_time = 0
        self.last_match: Detection | None = None
        self.debug_info = ""

    def detect(self, frame: np.ndarray) -> Detection | None:
        """Detect card in frame using ROI matching."""
        now = time.perf_counter()

        # Throttle detection
        if now - self.last_detection_time < self.detection_interval:
            return self.last_match

        self.last_detection_time = now

        h, w = frame.shape[:2]

        # Calculate ROI (center region, card aspect ratio ~1.4)
        roi_h = int(h * self.roi_scale)
        roi_w = int(roi_h / 1.4)

        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2

        # Extract ROI
        roi = frame[y1:y1 + roi_h, x1:x1 + roi_w]

        # Match
        match_start = time.perf_counter()
        result = self.matcher.match(roi)
        match_time = (time.perf_counter() - match_start) * 1000

        self.debug_info = f"Match: {match_time:.1f}ms"

        if result is None:
            self.last_match = None
            return None

        self.debug_info += f" | {result['name']} (d={result['distance']})"

        self.last_match = Detection(
            riftbound_id=result["riftbound_id"],
            name=result["name"],
            distance=result["distance"],
            bbox=(x1, y1, roi_w, roi_h),
        )
        return self.last_match

    def get_roi_rect(self, frame_shape: tuple) -> tuple[int, int, int, int]:
        """Get the ROI rectangle."""
        h, w = frame_shape[:2]
        roi_h = int(h * self.roi_scale)
        roi_w = int(roi_h / 1.4)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        return x1, y1, roi_w, roi_h


def draw_overlay(
    frame: np.ndarray,
    scanner: CardScanner,
    detection: Detection | None,
    fps: float,
    debug: bool = False,
) -> np.ndarray:
    """Draw all overlays on frame."""
    output = frame.copy()
    x, y, w, h = scanner.get_roi_rect(frame.shape)

    # ROI color based on match
    has_match = detection is not None
    color = (0, 255, 0) if has_match else (100, 100, 255)

    # Draw ROI rectangle with corner markers
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    corner_len = 25
    t = 4  # thickness
    # Top-left
    cv2.line(output, (x, y), (x + corner_len, y), color, t)
    cv2.line(output, (x, y), (x, y + corner_len), color, t)
    # Top-right
    cv2.line(output, (x + w, y), (x + w - corner_len, y), color, t)
    cv2.line(output, (x + w, y), (x + w, y + corner_len), color, t)
    # Bottom-left
    cv2.line(output, (x, y + h), (x + corner_len, y + h), color, t)
    cv2.line(output, (x, y + h), (x, y + h - corner_len), color, t)
    # Bottom-right
    cv2.line(output, (x + w, y + h), (x + w - corner_len, y + h), color, t)
    cv2.line(output, (x + w, y + h), (x + w, y + h - corner_len), color, t)

    # Status bar
    status = f"FPS: {fps:.0f}"
    if has_match:
        status += f" | {detection.name} (d={detection.distance})"
    cv2.putText(
        output, status, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
    )

    # Card label above ROI
    if has_match:
        label = detection.name
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # Background
        cv2.rectangle(
            output,
            (x, y - lh - 15),
            (x + lw + 10, y - 5),
            (0, 150, 0),
            -1,
        )
        cv2.putText(
            output, label, (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
    else:
        # Instruction
        cv2.putText(
            output, "Hold card in frame", (x + 10, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

    # Debug info
    if debug:
        cv2.putText(
            output, scanner.debug_info, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

    return output


class FPSCounter:
    def __init__(self):
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        delta = now - self._last_time
        self._last_time = now
        instant = 1.0 / delta if delta > 0 else 0
        self._fps = self._fps * 0.9 + instant * 0.1
        return self._fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Riftbound card detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--threshold", type=int, default=25,
        help="Hash distance threshold (lower = stricter, max ~64)"
    )
    parser.add_argument(
        "--roi-scale", type=float, default=0.7,
        help="ROI size as fraction of frame height"
    )
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scanner = CardScanner(
        threshold=args.threshold,
        roi_scale=args.roi_scale,
    )

    print(f"[Live] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Live] Resolution: {actual_w}x{actual_h}")
    print("[Live] Press 'q' to quit, 's' screenshot, 'd' debug")

    fps_counter = FPSCounter()
    debug_mode = args.debug

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = scanner.detect(frame)
            fps = fps_counter.update()
            output = draw_overlay(frame, scanner, detection, fps, debug_mode)

            cv2.imshow("Riftbound Card Scanner", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, output)
                print(f"[Live] Saved: {filename}")
            elif key == ord("d"):
                debug_mode = not debug_mode

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Live] Done")


if __name__ == "__main__":
    main()
