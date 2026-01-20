import argparse
import os
from pathlib import Path

from PIL import Image


def convert_folder(source_dir: Path, dest_dir: Path, quality: int, lossless: bool) -> None:
    source_dir = source_dir.resolve()
    dest_dir = dest_dir.resolve()

    for root, _, files in os.walk(source_dir):
        rel_root = Path(root).relative_to(source_dir)
        output_root = dest_dir / rel_root
        output_root.mkdir(parents=True, exist_ok=True)

        for filename in files:
            if not filename.lower().endswith(".png"):
                continue

            source_path = Path(root) / filename
            output_path = output_root / f"{source_path.stem}.webp"

            with Image.open(source_path) as image:
                image.save(
                    output_path,
                    format="WEBP",
                    quality=quality,
                    lossless=lossless,
                    method=6,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress card images from cards_png to cards_webp."
    )
    parser.add_argument(
        "--source",
        default="cards_png",
        help="Source folder containing PNG images.",
    )
    parser.add_argument(
        "--dest",
        default="cards_webp",
        help="Destination folder for WEBP images.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="WEBP quality (0-100) when not lossless.",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Use lossless WEBP compression.",
    )

    args = parser.parse_args()
    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    if not source_dir.exists():
        raise SystemExit(f"Source folder not found: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    convert_folder(source_dir, dest_dir, args.quality, args.lossless)


if __name__ == "__main__":
    main()
