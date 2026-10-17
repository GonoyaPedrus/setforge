"""Interface en ligne de commande : ``setforge --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from setforge.rekordbox import load_collection
from setforge.scoring import rank_next

app = typer.Typer(help="SetForge, assistant DJ.", no_args_is_help=True)

XmlOption = Annotated[
    Path,
    typer.Option(
        "--xml", exists=True, dir_okay=False, help="Export Rekordbox XML de la collection"
    ),
]


@app.command()
def stats(xml: XmlOption) -> None:
    """Résume la collection : nombre de pistes, BPM et tonalités disponibles."""
    tracks = load_collection(xml)
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
    tracks = load_collection(xml)
    needle = query.casefold()
    matches = [t for t in tracks if needle in t.label.casefold()]
    if not matches:
        typer.echo(f"Aucun morceau ne correspond à « {query} ».", err=True)
        raise typer.Exit(code=1)
    current = matches[0]
    typer.echo(f"Après : {current.label}")
    for rank, transition in enumerate(rank_next(current, tracks, limit), start=1):
        reasons = ", ".join(transition.reasons)
        typer.echo(f"{rank:>2}. {transition.score:.2f}  {transition.candidate.label}  [{reasons}]")
