"""Modèle de données commun à l'analyse, la recherche et l'export."""

from __future__ import annotations

from pydantic import BaseModel, Field

from setforge.camelot import CamelotKey, parse_key


class CuePoint(BaseModel):
    name: str = ""
    start: float = Field(ge=0, description="Position en secondes")
    num: int = Field(default=-1, description="-1 = memory cue, 0-7 = hot cue")


class Track(BaseModel):
    track_id: str
    title: str
    artist: str = ""
    location: str = Field(description="URL file:// telle que Rekordbox l'écrit")
    duration: float = Field(default=0, ge=0, description="Durée en secondes")
    bpm: float | None = Field(default=None, gt=0)
    key: str | None = Field(default=None, description="Tonalité brute, ex. « Am » ou « 8A »")
    energy: float | None = Field(default=None, ge=0, le=1)
    genre: str = ""
    cues: list[CuePoint] = Field(default_factory=list)

    @property
    def camelot(self) -> CamelotKey | None:
        return parse_key(self.key) if self.key else None

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist else self.title
