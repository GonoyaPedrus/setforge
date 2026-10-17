"""Conversion des tonalités en notation Camelot et compatibilité harmonique."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tonique -> numéro Camelot. Mineur = lettre A, majeur = lettre B.
_MINOR = {
    "G#": 1, "Ab": 1, "D#": 2, "Eb": 2, "A#": 3, "Bb": 3, "F": 4, "C": 5, "G": 6,
    "D": 7, "A": 8, "E": 9, "B": 10, "F#": 11, "Gb": 11, "C#": 12, "Db": 12,
}  # fmt: skip
_MAJOR = {
    "B": 1, "F#": 2, "Gb": 2, "C#": 3, "Db": 3, "G#": 4, "Ab": 4, "D#": 5, "Eb": 5,
    "A#": 6, "Bb": 6, "F": 7, "C": 8, "G": 9, "D": 10, "A": 11, "E": 12,
}  # fmt: skip

_CAMELOT_RE = re.compile(r"^(1[0-2]|[1-9])([AB])$")
_KEY_RE = re.compile(r"^([A-G][#b]?)\s*(m|min|minor|maj|major)?$")


@dataclass(frozen=True, order=True)
class CamelotKey:
    number: int
    letter: str

    def __str__(self) -> str:
        return f"{self.number}{self.letter}"


def parse_key(value: str) -> CamelotKey | None:
    """Lit une tonalité Rekordbox (« Am », « F#m », « Db ») ou Camelot (« 8A »)."""
    text = value.strip().replace("♯", "#").replace("♭", "b")
    if not text:
        return None
    if match := _CAMELOT_RE.match(text.upper()):
        return CamelotKey(int(match.group(1)), match.group(2))
    match = _KEY_RE.match(text[0].upper() + text[1:])
    if not match:
        return None
    tonic, mode = match.groups()
    minor = mode in ("m", "min", "minor")
    number = (_MINOR if minor else _MAJOR).get(tonic)
    if number is None:  # enharmonies rares (Cb, E#...) non gérées
        return None
    return CamelotKey(number, "A" if minor else "B")


def _wheel_distance(a: int, b: int) -> int:
    diff = abs(a - b) % 12
    return min(diff, 12 - diff)


def compatibility(a: CamelotKey, b: CamelotKey) -> float:
    """Score harmonique entre 0 et 1 pour enchaîner ``a`` puis ``b``.

    1.0 même tonalité, 0.9 voisin direct (±1) ou relatif majeur/mineur,
    0.6 diagonale (±1 et changement de lettre), 0.5 saut d'énergie (+2), sinon 0.
    """
    distance = _wheel_distance(a.number, b.number)
    same_letter = a.letter == b.letter
    if distance == 0:
        return 1.0 if same_letter else 0.9
    if distance == 1:
        return 0.9 if same_letter else 0.6
    if distance == 2 and same_letter:
        return 0.5
    return 0.0
