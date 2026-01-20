"""
Live card detection using OpenCV and Ultralytics YOLO.

Run this script to detect Riftbound cards from your webcam in real-time.
Cards are matched against the hash_cache.npz for identification.

Usage:
    python live_detect.py                    # Use contour detection (no model)
    python live_detect.py --model card.pt    # Use YOLO model
    python live_detect.py --camera 1         # Use camera index 1
    python live_detect.py --help             # Show all options
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from card_detector import CardDetector, Detection, draw_detections


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Riftbound card detection using YOLO and hash matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model (.pt file). If not provided, uses contour detection.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detections.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=5,
        help="Maximum number of cards to detect per frame.",
    )
    parser.add_argument(
        "--no-hash-match",
        action="store_true",
        help="Disable hash matching (only use YOLO class names).",
    )
    parser.add_argument(
        "--min-card-area",
        type=int,
        default=5000,
        help="Minimum card area in pixels for contour detection.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        default=True,
        help="Display FPS counter.",
    )
    return parser.parse_args()


class FPSCounter:
    """Simple FPS counter with smoothing."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        """Update and return current FPS."""
        now = time.perf_counter()
        delta = now - self._last_time
        self._last_time = now

        instant_fps = 1.0 / delta if delta > 0 else 0.0
        self._fps = self._fps * self.smoothing + instant_fps * (1 - self.smoothing)
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def main() -> None:
    """Main entry point for live detection."""
    args = parse_args()

    # Initialize detector
    model_path = Path(args.model) if args.model else None
    detector = CardDetector(
        model_path=model_path,
        confidence_threshold=args.confidence,
        max_detections=args.max_detections,
        use_hash_matching=not args.no_hash_match,
        min_card_area=args.min_card_area,
    )

    # Determine detection mode
    mode = "YOLO" if model_path and model_path.exists() else "Contour"
    print(f"[LiveDetect] Detection mode: {mode}")
    print(f"[LiveDetect] Hash matching: {'disabled' if args.no_hash_match else 'enabled'}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"[LiveDetect] Failed to open camera {args.camera}")

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[LiveDetect] Camera resolution: {actual_width}x{actual_height}")
    print("[LiveDetect] Press 'q' or ESC to quit")

    fps_counter = FPSCounter()
    window_name = "Riftbound Card Detection"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[LiveDetect] Failed to read frame")
                break

            # Detect cards
            detections = detector.detect(frame)

            # Draw results
            output = draw_detections(frame, detections)

            # Update and draw FPS
            fps = fps_counter.update()
            if args.show_fps:
                fps_text = f"FPS: {fps:.1f} | Mode: {mode} | Cards: {len(detections)}"
                cv2.putText(
                    output,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Show detection details in corner
            y_offset = 60
            for i, det in enumerate(detections[:3]):
                if det.match_distance is not None:
                    info = f"{i+1}. {det.card_id} (dist={det.match_distance:.0f})"
                else:
                    info = f"{i+1}. {det.card_id} (conf={det.confidence:.2f})"
                cv2.putText(
                    output,
                    info,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display
            cv2.imshow(window_name, output)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or 'q'
                break
            elif key == ord("s"):  # Save screenshot
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, output)
                print(f"[LiveDetect] Saved: {filename}")

    except KeyboardInterrupt:
        print("\n[LiveDetect] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[LiveDetect] Shutdown complete")


if __name__ == "__main__":
    main()
