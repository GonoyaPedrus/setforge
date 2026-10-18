"""Interface en ligne de commande : ``setforge --help``."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated

import typer

from setforge import store
from setforge.models import Track
from setforge.rekordbox import load_collection
from setforge.scoring import rank_next

app = typer.Typer(help="SetForge, assistant DJ.", no_args_is_help=True)

XmlOption = Annotated[
    Path,
    typer.Option(
        "--xml", exists=True, dir_okay=False, help="Export Rekordbox XML de la collection"
    ),
]

DEFAULT_DB = Path(os.environ.get("SETFORGE_DB", Path.home() / ".setforge" / "library.sqlite"))
DbOption = Annotated[
    Path,
    typer.Option("--db", dir_okay=False, help="Base SQLite des analyses (env SETFORGE_DB)"),
]


def _load(xml: Path, db: Path | None = None) -> list[Track]:
    """Charge la collection et, si la base existe, la complète avec les analyses."""
    try:
        tracks = load_collection(xml)
    except (ET.ParseError, ValueError) as error:
        typer.echo(f"Impossible de lire {xml} : {error}", err=True)
        raise typer.Exit(code=1) from error
    if db is None or not db.is_file():
        return tracks
    with store.connect(db) as connection:
        return store.enrich(tracks, store.load_analyses(connection))


def _find(tracks: list[Track], query: str) -> Track:
    """Correspondance exacte (titre ou « artiste - titre ») d'abord, puis sous-chaîne unique."""
    needle = query.casefold()
    exact = [t for t in tracks if needle in (t.title.casefold(), t.label.casefold())]
    matches = exact or [t for t in tracks if needle in t.label.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        typer.echo(f"Aucun morceau ne correspond à « {query} ».", err=True)
    else:
        typer.echo(f"« {query} » est ambigu, précise parmi :", err=True)
        for track in matches[:10]:
            typer.echo(f"  {track.label}", err=True)
    raise typer.Exit(code=1)


@app.command()
def stats(xml: XmlOption, db: DbOption = DEFAULT_DB) -> None:
    """Résume la collection : nombre de pistes, BPM, tonalités et énergies disponibles."""
    tracks = _load(xml, db)
    with_bpm = sum(1 for t in tracks if t.bpm)
    with_key = sum(1 for t in tracks if t.camelot)
    with_energy = sum(1 for t in tracks if t.energy is not None)
    typer.echo(
        f"{len(tracks)} pistes, {with_bpm} avec BPM, {with_key} avec tonalité, "
        f"{with_energy} avec énergie"
    )


@app.command()
def scan(
    xml: XmlOption,
    db: DbOption = DEFAULT_DB,
    workers: Annotated[int, typer.Option(min=1, help="Processus d'analyse en parallèle")] = 1,
) -> None:
    """Analyse les fichiers audio de la collection (énergie, BPM et tonalité de secours)."""
    try:
        from setforge.analysis.scan import scan as run_scan
    except ImportError as error:
        typer.echo("L'analyse audio demande l'extra « audio » : uv sync --extra audio", err=True)
        raise typer.Exit(code=1) from error

    def show(done: int, total: int) -> None:
        typer.echo(f"\rAnalyse {done}/{total}", nl=done == total)

    report = run_scan(_load(xml), db, workers=workers, progress=show)
    typer.echo(
        f"{report.analyzed} analysés, {report.cached} déjà à jour, "
        f"{len(report.missing)} fichiers introuvables, {len(report.failed)} en échec"
    )
    for location, message in report.failed.items():
        typer.echo(f"  échec {location} : {message}", err=True)


@app.command("next")
def next_track(
    xml: XmlOption,
    query: Annotated[str, typer.Argument(help="Titre ou « artiste - titre » du morceau joué")],
    limit: Annotated[int, typer.Option(min=1)] = 10,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Propose les morceaux qui enchaînent le mieux après QUERY."""
    tracks = _load(xml, db)
    current = _find(tracks, query)
    typer.echo(f"Après : {current.label}")
    for rank, transition in enumerate(rank_next(current, tracks, limit), start=1):
        reasons = ", ".join(transition.reasons)
        typer.echo(f"{rank:>2}. {transition.score:.2f}  {transition.candidate.label}  [{reasons}]")
