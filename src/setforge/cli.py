"""Interface en ligne de commande : ``setforge --help``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated

import typer

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


def _load(xml: Path) -> list[Track]:
    try:
        return load_collection(xml)
    except (ET.ParseError, ValueError) as error:
        typer.echo(f"Impossible de lire {xml} : {error}", err=True)
        raise typer.Exit(code=1) from error


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
def stats(xml: XmlOption) -> None:
    """Résume la collection : nombre de pistes, BPM et tonalités disponibles."""
    tracks = _load(xml)
    with_bpm = sum(1 for t in tracks if t.bpm)
    with_key = sum(1 for t in tracks if t.camelot)
    typer.echo(f"{len(tracks)} pistes, {with_bpm} avec BPM, {with_key} avec tonalité")


@app.command("next")
def next_track(
    xml: XmlOption,
    query: Annotated[str, typer.Argument(help="Titre ou « artiste - titre » du morceau joué")],
    limit: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Propose les morceaux qui enchaînent le mieux après QUERY."""
    tracks = _load(xml)
    current = _find(tracks, query)
    typer.echo(f"Après : {current.label}")
    for rank, transition in enumerate(rank_next(current, tracks, limit), start=1):
        reasons = ", ".join(transition.reasons)
        typer.echo(f"{rank:>2}. {transition.score:.2f}  {transition.candidate.label}  [{reasons}]")
