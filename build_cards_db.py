import argparse
import json
import sqlite3
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_CARDS_PATH = APP_ROOT / "cards.txt"
DEFAULT_DB_PATH = APP_ROOT / "cards.sqlite"
DEFAULT_SCHEMA_PATH = APP_ROOT / "cards_schema.sql"


def _load_cards(cards_path: Path):
    with cards_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    with schema_path.open("r", encoding="utf-8") as handle:
        conn.executescript(handle.read())


def _upsert_set(conn: sqlite3.Connection, set_id: str | None, label: str | None) -> None:
    if not set_id or not label:
        return
    conn.execute(
        "INSERT INTO sets (set_id, label) VALUES (?, ?) "
        "ON CONFLICT(set_id) DO UPDATE SET label = excluded.label",
        (set_id, label),
    )


def _get_or_create_lookup_id(
    conn: sqlite3.Connection,
    table: str,
    name: str,
    cache: dict[str, int],
) -> int:
    cached_id = cache.get(name)
    if cached_id is not None:
        return cached_id
    row = conn.execute(f"SELECT id FROM {table} WHERE name = ?", (name,)).fetchone()
    if row:
        cache[name] = row[0]
        return row[0]
    cursor = conn.execute(f"INSERT INTO {table} (name) VALUES (?)", (name,))
    cache[name] = cursor.lastrowid
    return cursor.lastrowid


def build_database(cards_path: Path, db_path: Path, schema_path: Path) -> None:
    data = _load_cards(cards_path)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _build_schema(conn, schema_path)

    domain_cache: dict[str, int] = {}
    tag_cache: dict[str, int] = {}

    card_count = 0
    set_count = 0

    with conn:
        for entry in data:
            entry_set_id = entry.get("set_id")
            entry_label = entry.get("label")
            if entry_set_id and entry_label:
                _upsert_set(conn, entry_set_id, entry_label)
                set_count += 1

            for card in entry.get("cards", []):
                set_info = card.get("set", {}) or {}
                set_id = set_info.get("set_id") or entry_set_id
                set_label = set_info.get("label") or entry_label
                _upsert_set(conn, set_id, set_label)

                attributes = card.get("attributes", {}) or {}
                classification = card.get("classification", {}) or {}
                text = card.get("text", {}) or {}
                media = card.get("media", {}) or {}
                metadata = card.get("metadata", {}) or {}

                cursor = conn.execute(
                    """
                    INSERT INTO cards (
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
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card.get("riftbound_id"),
                        card.get("name"),
                        card.get("public_code"),
                        card.get("tcgplayer_id"),
                        card.get("collector_number"),
                        classification.get("type"),
                        classification.get("supertype"),
                        classification.get("rarity"),
                        card.get("orientation"),
                        attributes.get("energy"),
                        attributes.get("might"),
                        attributes.get("power"),
                        text.get("plain"),
                        text.get("rich"),
                        media.get("image_url"),
                        media.get("artist"),
                        media.get("accessibility_text"),
                        set_id,
                        metadata.get("clean_name"),
                        int(bool(metadata.get("alternate_art"))),
                        int(bool(metadata.get("overnumbered"))),
                        int(bool(metadata.get("signature"))),
                    ),
                )
                card_id = cursor.lastrowid
                card_count += 1

                for domain in classification.get("domain") or []:
                    domain_id = _get_or_create_lookup_id(conn, "domains", domain, domain_cache)
                    conn.execute(
                        "INSERT OR IGNORE INTO card_domains (card_id, domain_id) VALUES (?, ?)",
                        (card_id, domain_id),
                    )

                for tag in card.get("tags") or []:
                    tag_id = _get_or_create_lookup_id(conn, "tags", tag, tag_cache)
                    conn.execute(
                        "INSERT OR IGNORE INTO card_tags (card_id, tag_id) VALUES (?, ?)",
                        (card_id, tag_id),
                    )

    conn.close()
    print(
        f"Created {db_path} with {card_count} cards, "
        f"{len(domain_cache)} domains, {len(tag_cache)} tags."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SQLite database from cards.txt.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    args = parser.parse_args()

    if not args.cards.exists():
        raise SystemExit(f"Missing cards file: {args.cards}")
    if not args.schema.exists():
        raise SystemExit(f"Missing schema file: {args.schema}")

    build_database(args.cards, args.db, args.schema)


if __name__ == "__main__":
    main()
