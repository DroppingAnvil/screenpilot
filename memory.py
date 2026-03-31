"""
Memory - SQLite-backed persistent memory.

Stores:
  - completed directives (summary + steps)
  - individual action logs per step
  - searchable by keyword for context injection into new plans
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS directives (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    directive   TEXT NOT NULL,
    summary     TEXT,
    steps       TEXT,           -- JSON array
    completed   INTEGER DEFAULT 0,
    created_at  TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS action_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    directive_id INTEGER,
    step_text    TEXT,
    action_type  TEXT,
    action_data  TEXT,          -- JSON
    result_data  TEXT,          -- JSON
    timestamp    TEXT,
    FOREIGN KEY (directive_id) REFERENCES directives(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS directives_fts
    USING fts5(directive, summary, content=directives, content_rowid=id);

CREATE TRIGGER IF NOT EXISTS directives_ai AFTER INSERT ON directives BEGIN
    INSERT INTO directives_fts(rowid, directive, summary)
    VALUES (new.id, new.directive, COALESCE(new.summary, ''));
END;

CREATE TRIGGER IF NOT EXISTS directives_au AFTER UPDATE ON directives BEGIN
    INSERT INTO directives_fts(directives_fts, rowid, directive, summary)
    VALUES ('delete', old.id, old.directive, COALESCE(old.summary, ''));
    INSERT INTO directives_fts(rowid, directive, summary)
    VALUES (new.id, new.directive, COALESCE(new.summary, ''));
END;
"""


class Memory:
    def __init__(self, db_path: str = "screenpilot.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._current_directive_id = None

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ #

    def store_completion(self, directive: str, steps: list[str]):
        """Mark a directive as fully completed."""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            if self._current_directive_id:
                conn.execute(
                    "UPDATE directives SET completed=1, completed_at=? WHERE id=?",
                    (now, self._current_directive_id)
                )
            else:
                conn.execute(
                    "INSERT INTO directives (directive, steps, completed, created_at, completed_at) VALUES (?,?,1,?,?)",
                    (directive, json.dumps(steps), now, now)
                )

    def log_action(self, directive: str, step: str, action: dict, result: dict):
        """Log a single executed action."""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            # Upsert directive record
            if not self._current_directive_id:
                cur = conn.execute(
                    "INSERT INTO directives (directive, created_at) VALUES (?,?)",
                    (directive, now)
                )
                self._current_directive_id = cur.lastrowid

            conn.execute(
                """INSERT INTO action_log
                   (directive_id, step_text, action_type, action_data, result_data, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (
                    self._current_directive_id,
                    step,
                    action.get("type", "unknown"),
                    json.dumps(action),
                    json.dumps(result),
                    now
                )
            )

    def search_relevant(self, query: str, limit: int = 10) -> list[dict]:
        """FTS search for past directives relevant to the current query."""
        import re
        # Strip non-alphanumeric chars so words like "Goal:" or "Profile:"
        # aren't interpreted as FTS5 column filters (column:term syntax).
        clean = [re.sub(r"[^a-zA-Z0-9]", "", w) for w in query.split()]
        clean = [w for w in clean if len(w) >= 2]
        if not clean:
            return []
        words = " OR ".join(clean[:8])
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT d.directive, d.summary, d.steps, d.completed_at
                       FROM directives_fts f
                       JOIN directives d ON d.id = f.rowid
                       WHERE directives_fts MATCH ? AND d.completed = 1
                       ORDER BY rank LIMIT ?""",
                    (words, limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[Memory] Search error: {e}")
            return []

    def get_all_directives(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM directives ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_action_log(self, directive_id: int = None, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            if directive_id:
                rows = conn.execute(
                    "SELECT * FROM action_log WHERE directive_id=? ORDER BY timestamp DESC LIMIT ?",
                    (directive_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM action_log ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_directive(self, directive_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM action_log WHERE directive_id=?", (directive_id,))
            conn.execute("DELETE FROM directives WHERE id=?", (directive_id,))

    def reset_current_session(self):
        self._current_directive_id = None
