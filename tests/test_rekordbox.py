from pathlib import Path

from setforge.rekordbox import load_collection, location_to_path, write_playlist


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


def test_loops_genre_and_duplicates_survive_export(collection_xml: Path, tmp_path: Path) -> None:
    tracks = load_collection(collection_xml)
    loop = tracks[1].cues[0]
    assert (loop.type, loop.end, loop.num) == (4, 40.0, 1)

    out = tmp_path / "set.xml"
    write_playlist([tracks[0], tracks[1], tracks[0]], "Repeat", out)
    reloaded = load_collection(out)
    assert [t.track_id for t in reloaded] == ["1", "2"]
    assert reloaded[1].cues == tracks[1].cues
    assert reloaded[0].genre == "Deep House"
    assert out.read_text(encoding="utf-8").count('Key="1"') == 2


def test_tolerates_malformed_numbers(tmp_path: Path) -> None:
    xml = tmp_path / "bad.xml"
    xml.write_text(
        '<DJ_PLAYLISTS><COLLECTION><TRACK TrackID="1" Name="x" TotalTime="n/a" AverageBpm="124,5">'
        '<POSITION_MARK Start="61,950" Num="0"/></TRACK></COLLECTION></DJ_PLAYLISTS>',
        encoding="utf-8",
    )
    track = load_collection(xml)[0]
    assert (track.duration, track.bpm) == (0, 124.5)
    assert (track.cues[0].start, track.cues[0].num) == (61.95, 0)


def test_location_to_path() -> None:
    assert location_to_path("file://localhost/Users/me/Music/a%20b.mp3") == Path(
        "/Users/me/Music/a b.mp3"
    )
    assert location_to_path("file://localhost/C:/Music/x.flac") == Path("C:/Music/x.flac")
    assert location_to_path("https://example.com/x.mp3") is None
