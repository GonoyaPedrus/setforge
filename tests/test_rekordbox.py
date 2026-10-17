from pathlib import Path

from setforge.rekordbox import load_collection, write_playlist


def test_load_collection(collection_xml: Path) -> None:
    tracks = load_collection(collection_xml)
    assert [t.track_id for t in tracks] == ["1", "2", "3", "4"]
    first = tracks[0]
    assert first.bpm == 124.0
    assert str(first.camelot) == "8A"
    assert first.cues[0].name == "Drop"
    assert first.cues[0].start == 61.95


def test_playlist_round_trip(collection_xml: Path, tmp_path: Path) -> None:
    tracks = load_collection(collection_xml)
    out = tmp_path / "set.xml"
    write_playlist(reversed(tracks), "SetForge test", out)

    reloaded = load_collection(out)
    assert [t.track_id for t in reloaded] == ["4", "3", "2", "1"]
    assert reloaded[3].cues == tracks[0].cues
    assert reloaded[3].location == tracks[0].location
    assert 'Name="SetForge test"' in out.read_text(encoding="utf-8")
