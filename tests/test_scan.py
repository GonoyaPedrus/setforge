from pathlib import Path

import pytest

pytest.importorskip("librosa")

from setforge import store
from setforge.analysis.features import AudioFeatures
from setforge.analysis.scan import scan
from setforge.models import Track


def _track(track_id: str, path: Path) -> Track:
    return Track(track_id=track_id, title=track_id, location=path.as_uri())


def _fake_analyze(path: Path) -> AudioFeatures:
    if path.name == "broken.mp3":
        raise RuntimeError("décodage impossible")
    loudness = float(path.read_text())
    return AudioFeatures(124.0, "Am", 0.1, loudness, 2.0, 1.0, 0.3)


def test_scan_analyzes_caches_and_reports(tmp_path: Path) -> None:
    quiet, loud, broken = tmp_path / "quiet.mp3", tmp_path / "loud.mp3", tmp_path / "broken.mp3"
    quiet.write_text("-20")
    loud.write_text("-8")
    broken.write_text("")
    tracks = [
        _track("1", quiet),
        _track("2", loud),
        _track("3", broken),
        _track("4", tmp_path / "absent.mp3"),
    ]
    db = tmp_path / "lib.sqlite"
    calls: list[tuple[int, int]] = []

    report = scan(tracks, db, analyze=_fake_analyze, progress=lambda d, t: calls.append((d, t)))

    assert report.analyzed == 2
    assert report.missing == ["4"]
    assert list(report.failed.values()) == ["décodage impossible"]
    assert calls[-1] == (3, 3)
    with store.connect(db) as connection:
        analyses = store.load_analyses(connection)
    assert analyses[loud.as_uri()].energy > analyses[quiet.as_uri()].energy

    again = scan(tracks, db, analyze=_fake_analyze)
    assert (again.analyzed, again.cached) == (0, 2)
