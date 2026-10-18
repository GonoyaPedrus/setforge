"""Lecture et écriture du format Rekordbox XML (Fichier > Exporter la collection en XML)."""

from __future__ import annotations

import configparser
import contextlib
import os
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlparse

from setforge.models import CuePoint, Track


def _number(value: str | None) -> float | None:
    """Lit un nombre Rekordbox ; accepte la virgule décimale, ignore les valeurs invalides."""
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _int(value: str | None, default: int) -> int:
    number = _number(value)
    return default if number is None else int(number)


def _positive(value: str | None) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _parse_track(node: ET.Element) -> Track:
    cues = [
        CuePoint(
            name=mark.get("Name", ""),
            type=_int(mark.get("Type"), default=0),
            start=max(_number(mark.get("Start")) or 0.0, 0.0),
            end=_positive(mark.get("End")),
            num=_int(mark.get("Num"), default=-1),
        )
        for mark in node.iter("POSITION_MARK")
    ]
    return Track(
        track_id=node.get("TrackID", ""),
        title=node.get("Name", ""),
        artist=node.get("Artist", ""),
        location=node.get("Location", ""),
        duration=_positive(node.get("TotalTime")) or 0,
        bpm=_positive(node.get("AverageBpm")),
        key=node.get("Tonality") or None,
        genre=node.get("Genre", ""),
        cues=cues,
    )


@cache
def running_in_wsl() -> bool:
    """Vrai sous Windows Subsystem for Linux, où les disques Windows sont montés sous Linux."""
    if sys.platform != "linux":
        return False
    return (
        "WSL_DISTRO_NAME" in os.environ
        or Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
        or "microsoft" in platform.uname().release.lower()
    )


@cache
def wsl_mount_root() -> str:
    """Racine des disques Windows : SETFORGE_WSL_MOUNT, sinon /etc/wsl.conf, sinon /mnt."""
    override = os.environ.get("SETFORGE_WSL_MOUNT", "").strip()
    if override.startswith("/"):
        return override.rstrip("/")
    config = configparser.ConfigParser()
    with contextlib.suppress(OSError, configparser.Error):
        config.read("/etc/wsl.conf")
    root = config.get("automount", "root", fallback="/mnt").strip().strip('"')
    return root.rstrip("/") if root.startswith("/") else "/mnt"


def location_to_path(location: str, wsl: bool | None = None) -> Path | None:
    """Convertit ``file://localhost/Users/...`` ou ``file://localhost/C:/...`` en chemin local.

    Un hôte réel (``file://nas/Music/...``) donne un chemin réseau ``//nas/Music/...``.
    Sous WSL (détecté si ``wsl`` est None), ``C:/Music`` devient ``/mnt/c/Music``, ou suit la
    racine de ``wsl_mount_root``.
    """
    parsed = urlparse(location)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if parsed.netloc not in ("", "localhost"):
        return Path(f"//{parsed.netloc}{path}")
    if drive := re.match(r"^/([A-Za-z]):/", path):  # chemin Windows
        if running_in_wsl() if wsl is None else wsl:
            return Path(f"{wsl_mount_root()}/{drive.group(1).lower()}/{path[4:]}")
        path = path[1:]
    return Path(path)


def load_collection(path: Path) -> list[Track]:
    """Charge toutes les pistes de la section COLLECTION."""
    root = ET.parse(path).getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        raise ValueError(f"{path} n'est pas un export Rekordbox XML (COLLECTION absente)")
    return [_parse_track(node) for node in collection.findall("TRACK")]


def write_playlist(tracks: Iterable[Track], name: str, path: Path) -> None:
    """Écrit un XML importable dans Rekordbox contenant une playlist ordonnée.

    Un morceau joué deux fois n'apparaît qu'une fois dans COLLECTION, deux fois dans la playlist.
    """
    ordered = list(tracks)
    unique = list({track.track_id: track for track in ordered}.values())
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="SetForge", Version="0.1.0", Company="")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(unique)))
    for track in unique:
        attrs = {
            "TrackID": track.track_id,
            "Name": track.title,
            "Artist": track.artist,
            "Genre": track.genre,
            "Location": track.location,
            "TotalTime": str(round(track.duration)),
        }
        if track.bpm:
            attrs["AverageBpm"] = f"{track.bpm:.2f}"
        if track.key:
            attrs["Tonality"] = track.key
        node = ET.SubElement(collection, "TRACK", attrs)
        for cue in track.cues:
            mark = {"Name": cue.name, "Type": str(cue.type), "Start": f"{cue.start:.3f}"}
            if cue.end is not None:
                mark["End"] = f"{cue.end:.3f}"
            mark["Num"] = str(cue.num)
            ET.SubElement(node, "POSITION_MARK", mark)
    playlists = ET.SubElement(root, "PLAYLISTS")
    folder = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    playlist = ET.SubElement(
        folder, "NODE", Name=name, Type="1", KeyType="0", Entries=str(len(ordered))
    )
    for track in ordered:
        ET.SubElement(playlist, "TRACK", Key=track.track_id)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
