"""Analyse en lot d'une collection, avec cache SQLite et calcul parallèle."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from setforge import store
from setforge.analysis.features import AudioFeatures, analyze_file
from setforge.models import Track
from setforge.rekordbox import location_to_path


@dataclass
class ScanReport:
    analyzed: int = 0
    cached: int = 0
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def scan(
    tracks: Iterable[Track],
    db: Path,
    workers: int = 1,
    analyze: Callable[[Path], AudioFeatures] = analyze_file,
    progress: Callable[[int, int], None] | None = None,
) -> ScanReport:
    """Analyse les fichiers absents ou modifiés depuis le dernier scan, puis recalcule l'énergie."""
    report = ScanReport()
    with store.connect(db) as connection:
        todo: dict[str, tuple[Path, float, int]] = {}
        for track in tracks:
            path = location_to_path(track.location)
            if path is None or not path.is_file():
                report.missing.append(track.label)
                continue
            stat = path.stat()
            if store.is_fresh(connection, track.location, stat.st_mtime, stat.st_size):
                report.cached += 1
            else:
                todo[track.location] = (path, stat.st_mtime, stat.st_size)

        def record(location: str, features: AudioFeatures | Exception) -> None:
            if isinstance(features, Exception):
                report.failed[location] = str(features)
            else:
                _, mtime, size = todo[location]
                store.save(connection, location, mtime, size, asdict(features))
                connection.commit()  # un scan interrompu garde ce qui est déjà analysé
                report.analyzed += 1
            if progress:
                progress(report.analyzed + len(report.failed), len(todo))

        if workers <= 1:
            for location, (path, _, _) in todo.items():
                try:
                    features = analyze(path)
                except Exception as error:  # un fichier illisible ne bloque pas le scan
                    record(location, error)
                else:
                    record(location, features)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(analyze, path): loc for loc, (path, _, _) in todo.items()}
                for future in as_completed(futures):
                    location = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        record(location, error)
                    else:
                        record(location, result)

        store.refresh_energy(connection)
    return report
