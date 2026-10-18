from pathlib import Path

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
        store.refresh_energy(connection)
    result = runner.invoke(app, ["stats", "--xml", str(collection_xml), "--db", str(db)])
    assert result.exit_code == 0
    assert "1 avec énergie" in result.output
