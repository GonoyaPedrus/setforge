from pathlib import Path

from typer.testing import CliRunner

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
