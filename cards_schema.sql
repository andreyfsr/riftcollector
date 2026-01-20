PRAGMA foreign_keys = ON;

CREATE TABLE sets (
    id INTEGER PRIMARY KEY,
    set_id TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL
);

CREATE TABLE cards (
    id INTEGER PRIMARY KEY,
    riftbound_id TEXT NOT NULL UNIQUE,
    name TEXT,
    public_code TEXT,
    tcgplayer_id TEXT,
    collector_number INTEGER,
    type TEXT,
    supertype TEXT,
    rarity TEXT,
    orientation TEXT,
    energy INTEGER,
    might INTEGER,
    power INTEGER,
    text_plain TEXT,
    text_rich TEXT,
    image_url TEXT,
    artist TEXT,
    accessibility_text TEXT,
    set_id TEXT,
    clean_name TEXT,
    alternate_art INTEGER,
    overnumbered INTEGER,
    signature INTEGER,
    FOREIGN KEY (set_id) REFERENCES sets(set_id) ON UPDATE CASCADE
);

CREATE TABLE domains (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE card_domains (
    card_id INTEGER NOT NULL,
    domain_id INTEGER NOT NULL,
    PRIMARY KEY (card_id, domain_id),
    FOREIGN KEY (card_id) REFERENCES cards(id) ON DELETE CASCADE,
    FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE CASCADE
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE card_tags (
    card_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (card_id, tag_id),
    FOREIGN KEY (card_id) REFERENCES cards(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

CREATE INDEX idx_cards_set_id ON cards(set_id);
CREATE INDEX idx_cards_name ON cards(name);
CREATE INDEX idx_cards_collector_number ON cards(collector_number);
CREATE INDEX idx_cards_type ON cards(type);
CREATE INDEX idx_cards_rarity ON cards(rarity);
CREATE INDEX idx_card_domains_domain ON card_domains(domain_id);
CREATE INDEX idx_card_tags_tag ON card_tags(tag_id);

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

CREATE INDEX IF NOT EXISTS idx_user_collection_user ON user_collection(user_sub);
