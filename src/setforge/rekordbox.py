"""Lecture et écriture du format Rekordbox XML (Fichier > Exporter la collection en XML)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from setforge.models import CuePoint, Track


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if number > 0 else None


def _parse_track(node: ET.Element) -> Track:
    cues = [
        CuePoint(
            name=mark.get("Name", ""),
            start=float(mark.get("Start", "0")),
            num=int(mark.get("Num", "-1")),
        )
        for mark in node.iter("POSITION_MARK")
    ]
    return Track(
        track_id=node.get("TrackID", ""),
        title=node.get("Name", ""),
        artist=node.get("Artist", ""),
        location=node.get("Location", ""),
        duration=_float(node.get("TotalTime")) or 0,
        bpm=_float(node.get("AverageBpm")),
        key=node.get("Tonality") or None,
        genre=node.get("Genre", ""),
        cues=cues,
    )


def load_collection(path: Path) -> list[Track]:
    """Charge toutes les pistes de la section COLLECTION."""
    root = ET.parse(path).getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        raise ValueError(f"{path} n'est pas un export Rekordbox XML (COLLECTION absente)")
    return [_parse_track(node) for node in collection.findall("TRACK")]


def write_playlist(tracks: Iterable[Track], name: str, path: Path) -> None:
    """Écrit un XML importable dans Rekordbox contenant une playlist ordonnée."""
    ordered = list(tracks)
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="SetForge", Version="0.1.0", Company="")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(ordered)))
    for track in ordered:
        attrs = {
            "TrackID": track.track_id,
            "Name": track.title,
            "Artist": track.artist,
            "Location": track.location,
            "TotalTime": str(round(track.duration)),
        }
        if track.bpm:
            attrs["AverageBpm"] = f"{track.bpm:.2f}"
        if track.key:
            attrs["Tonality"] = track.key
        node = ET.SubElement(collection, "TRACK", attrs)
        for cue in track.cues:
            ET.SubElement(
                node,
                "POSITION_MARK",
                Name=cue.name,
                Type="0",
                Start=f"{cue.start:.3f}",
                Num=str(cue.num),
            )
    playlists = ET.SubElement(root, "PLAYLISTS")
    folder = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    playlist = ET.SubElement(
        folder, "NODE", Name=name, Type="1", KeyType="0", Entries=str(len(ordered))
    )
    for track in ordered:
        ET.SubElement(playlist, "TRACK", Key=track.track_id)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
