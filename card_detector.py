"""
Card detection module using OpenCV and Ultralytics YOLO.

This module provides two detection strategies:
1. YOLO-based detection (fast, requires trained model)
2. Contour-based detection (fallback, no model needed)

Detected card regions are matched against hash_cache.npz for identification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

# Lazy-load hash cache functions to avoid circular imports
_hash_cache = None
_hash_matcher_loaded = False


def _load_hash_cache():
    """Load the hash cache from card_matcher module."""
    global _hash_cache, _hash_matcher_loaded
    if _hash_matcher_loaded:
        return _hash_cache
    try:
        from card_matcher import build_hash_cache
        _hash_cache = build_hash_cache()
        _hash_matcher_loaded = True
    except ImportError:
        _hash_cache = []
        _hash_matcher_loaded = True
    return _hash_cache


def _match_card_region(card_image: np.ndarray) -> dict | None:
    """
    Match a cropped card image against the hash cache.
    Returns the best match or None if no good match found.
    """
    from PIL import Image
    import imagehash

    hash_cache = _load_hash_cache()
    if not hash_cache:
        return None

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Resize to standard card dimensions for hashing
    pil_image = pil_image.resize((300, 420), Image.BICUBIC)

    # Compute perceptual hashes
    phash = imagehash.phash(pil_image, hash_size=8)
    dhash = imagehash.dhash(pil_image, hash_size=8)

    best_match = None
    best_distance = float("inf")

    for item in hash_cache:
        full_phash, full_dhash = item["full_hashes"]
        distance = (phash - full_phash) + (dhash - full_dhash)
        if distance < best_distance:
            best_distance = distance
            best_match = item

    # Threshold for accepting a match
    if best_distance > 40:
        return None

    return {
        "riftbound_id": best_match["riftbound_id"],
        "filename": best_match["filename"],
        "distance": best_distance,
    }


@dataclass
class Detection:
    """Represents a detected card in a frame."""
    card_id: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: np.ndarray | None = None
    match_distance: float | None = None


@dataclass
class CardDetector:
    """
    Detects Riftbound cards in video frames using YOLO or contour detection.

    Args:
        model_path: Path to YOLO model (.pt file). If None, uses contour detection.
        confidence_threshold: Minimum confidence for YOLO detections.
        iou_threshold: IoU threshold for non-max suppression.
        min_card_area: Minimum card area in pixels for contour detection.
        aspect_ratio_range: Valid aspect ratio range for cards (height/width).
        use_hash_matching: Whether to match detected regions against hash cache.
    """
    model_path: Path | None = None
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    min_card_area: int = 5000
    aspect_ratio_range: tuple[float, float] = (1.2, 1.8)
    use_hash_matching: bool = True
    max_detections: int = 5

    _model: "YOLO | None" = field(default=None, init=False, repr=False)
    _model_loaded: bool = field(default=False, init=False, repr=False)

    def _load_model(self) -> "YOLO | None":
        """Lazy-load the YOLO model."""
        if self._model_loaded:
            return self._model
        self._model_loaded = True

        if self.model_path is None:
            return None

        model_file = Path(self.model_path)
        if not model_file.exists():
            print(f"[CardDetector] Model not found: {model_file}")
            return None

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_file))
            print(f"[CardDetector] Loaded YOLO model: {model_file}")
            return self._model
        except ImportError:
            print("[CardDetector] ultralytics not installed, using contour detection")
            return None
        except Exception as e:
            print(f"[CardDetector] Failed to load model: {e}")
            return None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect cards in a video frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of Detection objects.
        """
        model = self._load_model()

        if model is not None:
            return self._detect_yolo(frame, model)
        return self._detect_contours(frame)

    def _detect_yolo(self, frame: np.ndarray, model: "YOLO") -> list[Detection]:
        """Detect cards using YOLO model."""
        results = model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])

                # Get class name if available
                class_name = result.names.get(cls_id, f"class_{cls_id}")

                # Crop the card region for hash matching
                card_region = frame[y1:y2, x1:x2]
                match_result = None
                card_id = class_name

                if self.use_hash_matching and card_region.size > 0:
                    match_result = _match_card_region(card_region)
                    if match_result:
                        card_id = match_result["riftbound_id"]

                detections.append(Detection(
                    card_id=card_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    polygon=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                    match_distance=match_result["distance"] if match_result else None,
                ))

        # Sort by confidence and limit
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:self.max_detections]

    def _detect_contours(self, frame: np.ndarray) -> list[Detection]:
        """Detect card-shaped rectangles using contour detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold for varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        detections = []
        frame_h, frame_w = frame.shape[:2]

        for contour in contours[:20]:  # Check top 20 contours
            area = cv2.contourArea(contour)
            if area < self.min_card_area:
                continue

            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Cards should be quadrilaterals
            if len(approx) != 4:
                continue

            # Check if convex
            if not cv2.isContourConvex(approx):
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Check aspect ratio (cards are taller than wide)
            aspect_ratio = h / w if w > 0 else 0
            min_ar, max_ar = self.aspect_ratio_range
            if not (min_ar <= aspect_ratio <= max_ar):
                continue

            # Check minimum dimensions
            if w < 50 or h < 70:
                continue

            # Calculate confidence based on how rectangular it is
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            confidence = min(rectangularity, 0.99)

            # Extract the card region using perspective transform
            polygon = approx.reshape(4, 2).astype(np.float32)
            card_region = self._extract_card_region(frame, polygon)

            # Match against hash cache
            match_result = None
            card_id = "unknown"

            if self.use_hash_matching and card_region is not None:
                match_result = _match_card_region(card_region)
                if match_result:
                    card_id = match_result["riftbound_id"]
                    confidence = max(confidence, 1.0 - (match_result["distance"] / 100.0))

            if card_id == "unknown" and not match_result:
                # Skip unidentified cards in contour mode
                continue

            detections.append(Detection(
                card_id=card_id,
                confidence=confidence,
                bbox=(x, y, x + w, y + h),
                polygon=polygon.astype(np.int32),
                match_distance=match_result["distance"] if match_result else None,
            ))

            if len(detections) >= self.max_detections:
                break

        return detections

    def _extract_card_region(
        self, frame: np.ndarray, polygon: np.ndarray
    ) -> np.ndarray | None:
        """Extract and warp the card region to a standard size."""
        if polygon.shape[0] != 4:
            return None

        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = self._order_points(polygon)
        (tl, tr, br, bl) = pts

        # Compute dimensions
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        if max_width < 50 or max_height < 70:
            return None

        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype=np.float32)

        # Compute perspective transform and warp
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))

        return warped

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum and diff to find corners
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff

        return rect


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Draw detection boxes and labels on a frame.

    Args:
        frame: BGR image from OpenCV.
        detections: List of Detection objects.

    Returns:
        Annotated frame.
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = (0, 220, 255)  # Yellow-orange

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw polygon if available
        if det.polygon is not None:
            cv2.polylines(output, [det.polygon], True, (0, 255, 0), 2)

        # Build label
        if det.match_distance is not None:
            label = f"{det.card_id} (d={det.match_distance:.0f})"
        else:
            label = f"{det.card_id} ({det.confidence:.2f})"

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            output,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 4, y1),
            color, -1
        )

        # Draw label text
        cv2.putText(
            output,
            label,
            (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output
