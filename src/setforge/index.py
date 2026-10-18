"""Index vectoriel LanceDB des embeddings de morceaux. Nécessite l'extra ``index``."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import numpy.typing as npt
import pyarrow as pa

TABLE = "tracks"

Fingerprint = tuple[float, int, str]  # mtime, taille, modèle


@dataclass(frozen=True)
class Embedding:
    location: str
    mtime: float
    size: int
    model: str
    vector: npt.NDArray[np.float32]


class TrackIndex:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(path))
        self._table: Any = None
        # Sinon, la table est créée au premier ajout, quand la dimension est connue.
        with contextlib.suppress(FileNotFoundError, ValueError):
            self._table = self._db.open_table(TABLE)

    def __len__(self) -> int:
        return 0 if self._table is None else int(self._table.count_rows())

    def fingerprints(self) -> dict[str, Fingerprint]:
        if self._table is None:
            return {}
        rows = self._table.to_arrow().select(["location", "mtime", "size", "model"]).to_pylist()
        return {r["location"]: (r["mtime"], r["size"], r["model"]) for r in rows}

    def upsert(self, embeddings: Iterable[Embedding]) -> None:
        batch = list(embeddings)
        if not batch:
            return
        rows = [
            {
                "location": e.location,
                "mtime": e.mtime,
                "size": e.size,
                "model": e.model,
                "vector": e.vector.astype(np.float32),
            }
            for e in batch
        ]
        if self._table is None:
            schema = pa.schema(
                [
                    ("location", pa.string()),
                    ("mtime", pa.float64()),
                    ("size", pa.int64()),
                    ("model", pa.string()),
                    ("vector", pa.list_(pa.float32(), batch[0].vector.size)),
                ]
            )
            self._table = self._db.create_table(TABLE, schema=schema)
        (
            self._table.merge_insert("location")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def vectors(self) -> dict[str, npt.NDArray[np.float32]]:
        if self._table is None:
            return {}
        rows = self._table.to_arrow().select(["location", "vector"]).to_pylist()
        return {r["location"]: np.asarray(r["vector"], dtype=np.float32) for r in rows}

    def search(self, vector: npt.NDArray[np.float32], limit: int) -> list[tuple[str, float]]:
        """Plus proches voisins par similarité cosinus, du plus proche au plus lointain."""
        if self._table is None:
            return []
        hits = self._table.search(vector).distance_type("cosine").limit(limit).to_list()
        return [(hit["location"], 1.0 - float(hit["_distance"])) for hit in hits]


def similarities(
    reference: npt.NDArray[np.float32], vectors: Mapping[str, npt.NDArray[np.float32]]
) -> dict[str, float]:
    """Similarité cosinus de ``reference`` avec chaque vecteur (tous normalisés L2)."""
    if not vectors:
        return {}
    locations = list(vectors)
    matrix = np.stack([vectors[location] for location in locations])
    scores = matrix @ reference
    return dict(zip(locations, (float(s) for s in scores), strict=True))
