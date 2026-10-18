from pathlib import Path

from setforge import store
from setforge.models import Track


def _features(rms_db: float, onset_rate: float = 2.0, flux: float = 1.0) -> dict[str, float | str]:
    return {
        "bpm": 124.0,
        "key": "Am",
        "key_confidence": 0.1,
        "rms_db": rms_db,
        "onset_rate": onset_rate,
        "flux": flux,
        "low_ratio": 0.3,
    }


def test_percentiles_handle_ties() -> None:
    assert store._percentiles([3.0, 1.0, 2.0]) == [1.0, 0.0, 0.5]
    assert store._percentiles([1.0, 1.0, 2.0]) == [0.25, 0.25, 1.0]
    assert store._percentiles([5.0]) == [0.5]


def test_energy_is_relative_to_library(tmp_path: Path) -> None:
    db = tmp_path / "lib.sqlite"
    with store.connect(db) as connection:
        store.save(connection, "file://localhost/quiet.mp3", 1.0, 10, _features(-20, 1, 0.5))
        store.save(connection, "file://localhost/loud.mp3", 1.0, 10, _features(-8, 4, 2.0))
        store.save(connection, "file://localhost/deleted.mp3", 1.0, 10, _features(-2, 9, 9.0))
        store.refresh_energy(
            connection, {"file://localhost/quiet.mp3", "file://localhost/loud.mp3"}
        )
        analyses = store.load_analyses(connection)
    assert analyses["file://localhost/quiet.mp3"].energy == 0.0
    assert analyses["file://localhost/loud.mp3"].energy == 1.0
    assert analyses["file://localhost/deleted.mp3"].energy is None


def test_fingerprints(tmp_path: Path) -> None:
    with store.connect(tmp_path / "lib.sqlite") as connection:
        store.save(connection, "loc", 1.0, 10, _features(-10))
        assert store.fingerprints(connection) == {"loc": (1.0, 10, store.ANALYSIS_VERSION)}


def test_enrich_keeps_rekordbox_values() -> None:
    analyses = {"loc": store.StoredAnalysis("loc", 126.0, "Em", 0.1, 0.7)}
    tagged = Track(track_id="1", title="a", location="loc", bpm=124.0, key="Am")
    untagged = Track(track_id="2", title="b", location="loc")
    unknown = Track(track_id="3", title="c", location="elsewhere")
    first, second, third = store.enrich([tagged, untagged, unknown], analyses)
    assert (first.bpm, first.key, first.energy) == (124.0, "Am", 0.7)
    assert (second.bpm, second.key, second.energy) == (126.0, "Em", 0.7)
    assert third == unknown


def test_enrich_ignores_uncertain_key() -> None:
    analyses = {"loc": store.StoredAnalysis("loc", None, "F#m", 0.001, 0.4)}
    (track,) = store.enrich([Track(track_id="1", title="a", location="loc")], analyses)
    assert (track.bpm, track.key, track.energy) == (None, None, 0.4)
