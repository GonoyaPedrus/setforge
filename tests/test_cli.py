from pathlib import Path

import pytest
from typer.testing import CliRunner

from setforge import store
from setforge.cli import app

runner = CliRunner()


def test_stats(collection_xml: Path) -> None:
    result = runner.invoke(app, ["stats", "--xml", str(collection_xml)])
    assert result.exit_code == 0
    assert "4 pistes" in result.output


def test_next(collection_xml: Path) -> None:
    result = runner.invoke(
        app, ["next", "--xml", str(collection_xml), "night drive", "--limit", "2"]
    )
    assert result.exit_code == 0
    assert "Artist B - Low Tide" in result.output.splitlines()[1]


def test_next_unknown_track(collection_xml: Path) -> None:
    result = runner.invoke(app, ["next", "--xml", str(collection_xml), "introuvable"])
    assert result.exit_code == 1


def test_next_ambiguous_query(collection_xml: Path) -> None:
    result = runner.invoke(app, ["next", "--xml", str(collection_xml), "artist"])
    assert result.exit_code == 1
    assert "ambigu" in result.output


def test_unreadable_xml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<not xml", encoding="utf-8")
    result = runner.invoke(app, ["stats", "--xml", str(bad)])
    assert result.exit_code == 1
    assert "Impossible de lire" in result.output


def test_stats_reads_energy_from_db(collection_xml: Path, tmp_path: Path) -> None:
    db = tmp_path / "lib.sqlite"
    with store.connect(db) as connection:
        store.save(
            connection,
            "file://localhost/Music/night%20drive.mp3",
            1.0,
            1,
            {
                "bpm": 124.0,
                "key": "Am",
                "key_confidence": 0.1,
                "rms_db": -9.0,
                "onset_rate": 2.0,
                "flux": 1.0,
                "low_ratio": 0.3,
            },
        )
        store.refresh_energy(connection, {"file://localhost/Music/night%20drive.mp3"})
    result = runner.invoke(app, ["stats", "--xml", str(collection_xml), "--db", str(db)])
    assert result.exit_code == 0
    assert "1 avec énergie" in result.output


def test_unreadable_db_falls_back_to_rekordbox(collection_xml: Path, tmp_path: Path) -> None:
    db = tmp_path / "lib.sqlite"
    db.write_text("ceci n'est pas une base SQLite", encoding="utf-8")
    result = runner.invoke(app, ["stats", "--xml", str(collection_xml), "--db", str(db)])
    assert result.exit_code == 0
    assert "4 pistes" in result.output
    assert "Analyses ignorées" in result.output


def _index_fixture(tmp_path: Path) -> Path:
    np = pytest.importorskip("numpy")
    pytest.importorskip("lancedb")
    from setforge.embeddings import DEFAULT_MODEL
    from setforge.index import Embedding, TrackIndex

    path = tmp_path / "idx"
    vectors = {
        "night%20drive.mp3": [1.0, 0.0],
        "low%20tide.flac": [0.0, 1.0],
        "sunrise.mp3": [1.0, 0.0],
    }
    TrackIndex(path).upsert(
        Embedding(
            f"file://localhost/Music/{name}", 1.0, 1, DEFAULT_MODEL, np.array(v, dtype=np.float32)
        )
        for name, v in vectors.items()
    )
    return path


def test_next_uses_style_similarity(collection_xml: Path, tmp_path: Path) -> None:
    index = _index_fixture(tmp_path)
    args = ["next", "--xml", str(collection_xml), "night drive", "--index", str(index)]
    result = runner.invoke(app, [*args, "--db", str(tmp_path / "none.sqlite")])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "Artist C - Sunrise" in lines[1]  # même style, malgré une tonalité moins proche
    assert "style 1.00" in lines[1]
    assert "style inconnu" in lines[3]


def test_search_filters_by_bpm(
    collection_xml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    from setforge import embeddings

    class FakeEmbedder:
        def embed_text(self, text: str) -> object:
            return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(embeddings.ClapEmbedder, "load", classmethod(lambda cls: FakeEmbedder()))
    index = _index_fixture(tmp_path)
    base = ["search", "--xml", str(collection_xml), "deep", "--index", str(index)]
    base += ["--db", str(tmp_path / "none.sqlite")]

    result = runner.invoke(app, base)
    assert result.exit_code == 0, result.output
    assert "Artist B - Low Tide" in result.stdout.splitlines()[0]

    filtered = runner.invoke(app, [*base, "--bpm", "126", "--tolerance", "1"])
    assert "Sunrise" in filtered.stdout.splitlines()[0]
    assert "Low Tide" not in filtered.stdout


def test_search_without_index(collection_xml: Path, tmp_path: Path) -> None:
    args = ["search", "--xml", str(collection_xml), "deep", "--index", str(tmp_path / "x")]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "scan --embeddings" in result.output
