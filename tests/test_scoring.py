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


def test_bpm_delta_is_signed() -> None:
    assert bpm_delta_pct(124, 120) < 0
    assert bpm_delta_pct(120, 124) > 0


def test_unknown_key_is_neutral_not_perfect() -> None:
    current = _track("a", 124, "Am")
    unknown = score_transition(current, _track("b", 124, None)).score
    perfect = score_transition(current, _track("c", 124, "Am")).score
    clash = score_transition(current, _track("d", 124, "3A")).score
    assert clash < unknown < perfect


def test_missing_energy_is_ignored() -> None:
    current = _track("a", 124, "Am")
    assert score_transition(current, _track("b", 124, "Am")).score == 1.0


def test_rank_next_without_track_ids() -> None:
    tracks = [_track("", 124, "Am"), _track("", 124, "Em")]
    assert len(rank_next(tracks[0], tracks)) == 1


def test_rank_next_prefers_harmonic_close_tempo(collection_xml: Path) -> None:
    tracks = load_collection(collection_xml)
    ranked = rank_next(tracks[0], tracks)
    assert [t.candidate.track_id for t in ranked] == ["2", "3", "4"]
    assert ranked[-1].score < 0.2
