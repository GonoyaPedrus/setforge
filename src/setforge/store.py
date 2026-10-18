"""Cache SQLite des analyses, indexé par emplacement de fichier."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from setforge.models import Track

# À incrémenter quand l'algorithme d'analyse change : les analyses plus anciennes sont refaites.
ANALYSIS_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis (
    location TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    version INTEGER NOT NULL,
    bpm REAL NOT NULL,
    key TEXT NOT NULL,
    key_confidence REAL NOT NULL,
    rms_db REAL NOT NULL,
    onset_rate REAL NOT NULL,
    flux REAL NOT NULL,
    low_ratio REAL NOT NULL,
    energy REAL
);
"""

FEATURE_FIELDS = (
    "bpm",
    "key",
    "key_confidence",
    "rms_db",
    "onset_rate",
    "flux",
    "low_ratio",
)
# Descripteurs combinés dans le score d'énergie, par rang centile sur la bibliothèque.
_ENERGY_COLUMNS = ("rms_db", "onset_rate", "flux")


@dataclass(frozen=True)
class StoredAnalysis:
    location: str
    bpm: float
    key: str
    key_confidence: float
    energy: float | None


@contextmanager
def connect(db: Path) -> Iterator[sqlite3.Connection]:
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db)
    try:
        connection.execute(_SCHEMA)
        with connection:
            yield connection
    finally:
        connection.close()


def is_fresh(connection: sqlite3.Connection, location: str, mtime: float, size: int) -> bool:
    row = connection.execute(
        "SELECT mtime, size, version FROM analysis WHERE location = ?", (location,)
    ).fetchone()
    return row is not None and tuple(row) == (mtime, size, ANALYSIS_VERSION)


def save(
    connection: sqlite3.Connection,
    location: str,
    mtime: float,
    size: int,
    features: Mapping[str, float | str],
) -> None:
    """Enregistre les champs ``FEATURE_FIELDS`` d'une analyse."""
    values = {name: features[name] for name in FEATURE_FIELDS}
    connection.execute(
        f"INSERT OR REPLACE INTO analysis (location, mtime, size, version, {', '.join(values)}) "
        f"VALUES (?, ?, ?, ?, {', '.join('?' for _ in values)})",
        (location, mtime, size, ANALYSIS_VERSION, *values.values()),
    )


def _percentiles(values: list[float]) -> list[float]:
    """Rang centile de chaque valeur dans [0, 1], ex aequo au rang moyen."""
    if len(values) == 1:
        return [0.5]
    order = sorted(values)
    last = len(values) - 1
    ranks: dict[float, float] = {}
    start = 0
    for end in range(1, len(order) + 1):
        if end == len(order) or order[end] != order[start]:
            ranks[order[start]] = (start + end - 1) / 2 / last
            start = end
    return [ranks[value] for value in values]


def refresh_energy(connection: sqlite3.Connection) -> None:
    """Recalcule l'énergie (0-1) de toute la bibliothèque, relative aux autres morceaux."""
    rows = connection.execute(
        f"SELECT location, {', '.join(_ENERGY_COLUMNS)} FROM analysis"
    ).fetchall()
    if not rows:
        return
    columns = [_percentiles([row[i + 1] for row in rows]) for i in range(len(_ENERGY_COLUMNS))]
    connection.executemany(
        "UPDATE analysis SET energy = ? WHERE location = ?",
        [
            (round(sum(col[i] for col in columns) / len(columns), 4), row[0])
            for i, row in enumerate(rows)
        ],
    )


def load_analyses(connection: sqlite3.Connection) -> dict[str, StoredAnalysis]:
    rows = connection.execute(
        "SELECT location, bpm, key, key_confidence, energy FROM analysis"
    ).fetchall()
    return {row[0]: StoredAnalysis(*row) for row in rows}


def enrich(tracks: Iterable[Track], analyses: dict[str, StoredAnalysis]) -> list[Track]:
    """Complète les pistes : Rekordbox reste prioritaire pour BPM et tonalité."""
    enriched = []
    for track in tracks:
        found = analyses.get(track.location)
        if found is None:
            enriched.append(track)
            continue
        enriched.append(
            track.model_copy(
                update={
                    "bpm": track.bpm or found.bpm,
                    "key": track.key or found.key,
                    "energy": found.energy,
                }
            )
        )
    return enriched
