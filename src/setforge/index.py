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
        dimension = batch[0].vector.size
        if self._table is not None and self._dimension() != dimension:
            # Nouveau modèle d'une autre dimension : les anciens vecteurs ne servent plus.
            self._db.drop_table(TABLE)
            self._table = None
        if self._table is None:
            schema = pa.schema(
                [
                    ("location", pa.string()),
                    ("mtime", pa.float64()),
                    ("size", pa.int64()),
                    ("model", pa.string()),
                    ("vector", pa.list_(pa.float32(), dimension)),
                ]
            )
            self._table = self._db.create_table(TABLE, schema=schema)
        (
            self._table.merge_insert("location")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def _dimension(self) -> int:
        return int(self._table.schema.field("vector").type.list_size)

    def vectors(self, model: str) -> dict[str, npt.NDArray[np.float32]]:
        """Vecteurs calculés avec ``model`` (ceux d'un autre modèle ne sont pas comparables)."""
        if self._table is None:
            return {}
        table = self._table.to_arrow().select(["location", "model", "vector"])
        flat = table.column("vector").combine_chunks().flatten().to_numpy()
        matrix = flat.reshape(table.num_rows, self._dimension()).astype(np.float32)
        locations = table.column("location").to_pylist()
        models = table.column("model").to_pylist()
        return {
            location: matrix[i]
            for i, (location, row_model) in enumerate(zip(locations, models, strict=True))
            if row_model == model
        }

    def search(
        self, vector: npt.NDArray[np.float32], model: str, limit: int | None = None
    ) -> list[tuple[str, float]]:
        """Morceaux du ``model`` par similarité cosinus décroissante (tous sans ``limit``)."""
        if self._table is None:
            return []
        query = (
            self._table.search(vector)
            .distance_type("cosine")
            .where(f"model = '{model.replace("'", "''")}'", prefilter=True)
            .limit(limit or len(self))
        )
        return [(hit["location"], 1.0 - float(hit["_distance"])) for hit in query.to_list()]


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
