"""Interface en ligne de commande : ``setforge --help``."""

from __future__ import annotations

import os
import sqlite3
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from setforge import store
from setforge.models import Track

if TYPE_CHECKING:
    from setforge.embeddings import ClapEmbedder
    from setforge.index import TrackIndex
from setforge.camelot import parse_key
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
DEFAULT_INDEX = Path(os.environ.get("SETFORGE_INDEX", Path.home() / ".setforge" / "index"))
IndexOption = Annotated[
    Path,
    typer.Option(
        "--index", file_okay=False, help="Index LanceDB des embeddings (env SETFORGE_INDEX)"
    ),
]


def _missing_extra(extra: str) -> typer.Exit:
    typer.echo(f"Cette commande demande l'extra « {extra} » : uv sync --extra {extra}", err=True)
    return typer.Exit(code=1)


def _load_clap() -> ClapEmbedder:
    from setforge.embeddings import ClapEmbedder

    typer.echo("Chargement du modèle CLAP (téléchargé au premier lancement)...", err=True)
    try:
        return ClapEmbedder.load()
    except ImportError as error:
        raise _missing_extra("clap") from error


def _open_index(path: Path) -> TrackIndex | None:
    """Index existant, ou None s'il n'a jamais été créé ou si LanceDB n'est pas installé."""
    if not path.is_dir():
        return None
    try:
        from setforge.index import TrackIndex
    except ImportError:
        return None
    return TrackIndex(path)


def _load(xml: Path, db: Path | None = None) -> list[Track]:
    """Charge la collection et, si la base existe, la complète avec les analyses."""
    try:
        tracks = load_collection(xml)
    except (ET.ParseError, ValueError) as error:
        typer.echo(f"Impossible de lire {xml} : {error}", err=True)
        raise typer.Exit(code=1) from error
    if db is None or not db.is_file():
        return tracks
    try:
        with store.connect(db) as connection:
            return store.enrich(tracks, store.load_analyses(connection))
    except sqlite3.Error as error:
        typer.echo(f"Analyses ignorées, base {db} illisible : {error}", err=True)
        return tracks


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
    embeddings: Annotated[
        bool, typer.Option(help="Calcule aussi les embeddings CLAP (extras clap et index)")
    ] = False,
    index: IndexOption = DEFAULT_INDEX,
) -> None:
    """Analyse les fichiers audio de la collection (énergie, BPM et tonalité de secours)."""
    try:
        from setforge.analysis.scan import scan as run_scan
    except ImportError as error:
        raise _missing_extra("audio") from error
    if embeddings:
        try:
            from setforge.analysis.embed import embed_library
            from setforge.index import TrackIndex
        except ImportError as error:
            raise _missing_extra("index") from error
        try:
            # Vérifie l'extra tout de suite, mais ne charge le modèle qu'après l'analyse :
            # inutile de garder CLAP en mémoire à côté des workers.
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as error:
            raise _missing_extra("clap") from error

    def show(step: str) -> Callable[[int, int], None]:
        return lambda done, total: typer.echo(f"\r{step} {done}/{total}", nl=done == total)

    tracks = _load(xml)
    report = run_scan(tracks, db, workers=workers, progress=show("Analyse"))
    typer.echo(
        f"{report.analyzed} analysés, {report.cached} déjà à jour, "
        f"{len(report.missing)} fichiers introuvables, {len(report.failed)} en échec"
    )
    for location, message in report.failed.items():
        typer.echo(f"  échec {location} : {message}", err=True)
    if not embeddings:
        return

    embedder = _load_clap()
    embedded = embed_library(tracks, TrackIndex(index), embedder, progress=show("Embeddings"))
    typer.echo(
        f"{embedded.embedded} embeddings calculés, {embedded.cached} déjà à jour, "
        f"{len(embedded.failed)} en échec"
    )
    for location, message in embedded.failed.items():
        typer.echo(f"  échec {location} : {message}", err=True)


@app.command("next")
def next_track(
    xml: XmlOption,
    query: Annotated[str, typer.Argument(help="Titre ou « artiste - titre » du morceau joué")],
    limit: Annotated[int, typer.Option(min=1)] = 10,
    db: DbOption = DEFAULT_DB,
    index: IndexOption = DEFAULT_INDEX,
) -> None:
    """Propose les morceaux qui enchaînent le mieux après QUERY."""
    tracks = _load(xml, db)
    current = _find(tracks, query)
    scores: dict[str, float] = {}
    track_index = _open_index(index)
    if track_index is not None:
        from setforge.embeddings import DEFAULT_MODEL
        from setforge.index import similarities

        vectors = track_index.vectors(DEFAULT_MODEL)
        if current.location in vectors:
            scores = similarities(vectors[current.location], vectors)
    typer.echo(f"Après : {current.label}")
    for rank, transition in enumerate(rank_next(current, tracks, limit, scores), start=1):
        reasons = ", ".join(transition.reasons)
        typer.echo(f"{rank:>2}. {transition.score:.2f}  {transition.candidate.label}  [{reasons}]")


@app.command()
def search(
    xml: XmlOption,
    query: Annotated[str, typer.Argument(help="Description libre, ex. « deep house sombre »")],
    bpm: Annotated[float | None, typer.Option(min=1, help="Tempo visé")] = None,
    tolerance: Annotated[float, typer.Option(min=0, help="Écart de BPM accepté")] = 3.0,
    key: Annotated[str | None, typer.Option(help="Tonalité, ex. 8A ou Am")] = None,
    limit: Annotated[int, typer.Option(min=1)] = 10,
    db: DbOption = DEFAULT_DB,
    index: IndexOption = DEFAULT_INDEX,
) -> None:
    """Cherche des morceaux par description, filtrés par BPM et tonalité."""
    try:
        from setforge.embeddings import DEFAULT_MODEL
        from setforge.index import TrackIndex  # noqa: F401  (vérifie l'extra index)
    except ImportError as error:
        raise _missing_extra("index") from error
    track_index = _open_index(index)
    if track_index is None or not len(track_index):
        typer.echo("Index vide : lance d'abord « setforge scan --embeddings ».", err=True)
        raise typer.Exit(code=1)
    wanted_key = parse_key(key) if key else None
    if key and wanted_key is None:
        typer.echo(f"Tonalité « {key} » non reconnue.", err=True)
        raise typer.Exit(code=1)

    by_location = {t.location: t for t in _load(xml, db)}
    vector = _load_clap().embed_text(query)
    # Tout l'index est classé (quelques milliers de morceaux) puis filtré :
    # un morceau qui passe les filtres n'est jamais écarté par une coupe arbitraire.
    hits = track_index.search(vector, DEFAULT_MODEL)
    shown = 0
    for location, similarity in hits:
        track = by_location.get(location)
        if track is None:
            continue
        if bpm is not None and (track.bpm is None or abs(track.bpm - bpm) > tolerance):
            continue
        if wanted_key is not None and track.camelot != wanted_key:
            continue
        shown += 1
        details = f"{track.bpm:.1f} BPM" if track.bpm else "BPM inconnu"
        typer.echo(
            f"{shown:>2}. {similarity:.2f}  {track.label}  [{details}, {track.camelot or '?'}]"
        )
        if shown == limit:
            break
    if not shown:
        typer.echo("Aucun morceau ne correspond aux filtres.")
