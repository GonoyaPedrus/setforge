"""Analyse en lot d'une collection, avec cache SQLite et calcul parallèle."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path

from setforge import store
from setforge.analysis.features import AudioFeatures, analyze_file
from setforge.models import Track
from setforge.rekordbox import location_to_path

Analyzer = Callable[[Path], AudioFeatures]
Outcome = AudioFeatures | Exception


@dataclass
class ScanReport:
    analyzed: int = 0
    cached: int = 0
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _attempt(analyze: Analyzer, path: Path) -> Outcome:
    """Un fichier illisible devient un résultat en échec au lieu d'arrêter le scan."""
    try:
        return analyze(path)
    except Exception as error:
        return error


def _run(jobs: dict[str, Path], analyze: Analyzer, workers: int) -> Generator[tuple[str, Outcome]]:
    if workers <= 1:
        for location, path in jobs.items():
            yield location, _attempt(analyze, path)
        return
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_attempt, analyze, path): loc for loc, path in jobs.items()}
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        # Ctrl-C ou erreur : on abandonne la file d'attente au lieu de la terminer.
        pool.shutdown(wait=True, cancel_futures=True)


def scan(
    tracks: Iterable[Track],
    db: Path,
    workers: int = 1,
    analyze: Analyzer = analyze_file,
    progress: Callable[[int, int], None] | None = None,
) -> ScanReport:
    """Analyse les fichiers absents ou modifiés depuis le dernier scan, puis recalcule l'énergie."""
    report = ScanReport()
    with store.connect(db) as connection:
        known = store.fingerprints(connection)
        present: set[str] = set()
        jobs: dict[str, Path] = {}
        stats: dict[str, tuple[float, int]] = {}
        for track in tracks:
            path = location_to_path(track.location)
            if path is None or not path.is_file():
                report.missing.append(track.label)
                continue
            present.add(track.location)
            stat = path.stat()
            if known.get(track.location) == (stat.st_mtime, stat.st_size, store.ANALYSIS_VERSION):
                report.cached += 1
            else:
                jobs[track.location] = path
                stats[track.location] = (stat.st_mtime, stat.st_size)

        with closing(_run(jobs, analyze, workers)) as outcomes:
            for location, outcome in outcomes:
                if isinstance(outcome, Exception):
                    report.failed[location] = str(outcome)
                else:
                    mtime, size = stats[location]
                    store.save(connection, location, mtime, size, asdict(outcome))
                    connection.commit()  # un scan interrompu garde ce qui est déjà analysé
                    report.analyzed += 1
                if progress:
                    progress(report.analyzed + len(report.failed), len(jobs))

        store.refresh_energy(connection, present)
    return report
