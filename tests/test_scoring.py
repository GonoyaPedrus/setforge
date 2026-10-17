from pathlib import Path

import pytest

from setforge.models import Track
from setforge.rekordbox import load_collection
from setforge.scoring import bpm_delta_pct, rank_next, score_transition


def _track(track_id: str, bpm: float | None, key: str | None, energy: float | None = None) -> Track:
    return Track(track_id=track_id, title=track_id, location="", bpm=bpm, key=key, energy=energy)


def test_bpm_delta_handles_double_time() -> None:
    assert bpm_delta_pct(124, 62) == 0
    assert bpm_delta_pct(87, 174) == 0
    assert bpm_delta_pct(100, 103) == pytest.approx(3.0)


def test_identical_tracks_score_one() -> None:
    current = _track("a", 124, "Am", 0.5)
    assert score_transition(current, _track("b", 124, "Am", 0.5)).score == 1.0


def test_missing_features_are_ignored() -> None:
    current = _track("a", 124, None)
    assert score_transition(current, _track("b", 124, None)).score == 1.0


def test_rank_next_prefers_harmonic_close_tempo(collection_xml: Path) -> None:
    tracks = load_collection(collection_xml)
    ranked = rank_next(tracks[0], tracks)
    assert [t.candidate.track_id for t in ranked] == ["2", "3", "4"]
    assert ranked[-1].score < 0.2
