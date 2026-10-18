"""Calcul des embeddings CLAP d'une collection, avec cache dans l'index LanceDB."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from setforge.index import Embedding, TrackIndex
from setforge.models import Track
from setforge.rekordbox import location_to_path

# Écrit l'index par paquets : un scan interrompu garde l'essentiel du travail fait.
BATCH_SIZE = 32


class AudioEmbedder(Protocol):
    name: str

    def embed_file(self, path: Path) -> npt.NDArray[np.float32]: ...


@dataclass
class EmbedReport:
    embedded: int = 0
    cached: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def embed_library(
    tracks: Iterable[Track],
    index: TrackIndex,
    embedder: AudioEmbedder,
    progress: Callable[[int, int], None] | None = None,
) -> EmbedReport:
    """Calcule les embeddings absents, obsolètes (fichier modifié) ou d'un autre modèle."""
    report = EmbedReport()
    known = index.fingerprints()
    jobs: list[tuple[str, Path, float, int]] = []
    for track in tracks:
        path = location_to_path(track.location)
        if path is None or not path.is_file():
            continue  # déjà signalé par le scan des descripteurs
        stat = path.stat()
        if known.get(track.location) == (stat.st_mtime, stat.st_size, embedder.name):
            report.cached += 1
        else:
            jobs.append((track.location, path, stat.st_mtime, stat.st_size))

    batch: list[Embedding] = []
    for done, (location, path, mtime, size) in enumerate(jobs, start=1):
        try:
            vector = embedder.embed_file(path)
        except Exception as error:  # un fichier illisible ne bloque pas le scan
            report.failed[location] = str(error)
        else:
            batch.append(Embedding(location, mtime, size, embedder.name, vector))
            report.embedded += 1
        if len(batch) >= BATCH_SIZE:
            index.upsert(batch)
            batch = []
        if progress:
            progress(done, len(jobs))
    index.upsert(batch)
    return report
