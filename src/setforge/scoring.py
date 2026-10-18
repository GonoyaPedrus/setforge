"""Score de transition entre deux morceaux."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from setforge.camelot import compatibility
from setforge.models import Track

# Écart de tempo au-delà duquel le pitch devient audible sur la plupart des morceaux.
MAX_BPM_DELTA_PCT = 6.0
# Score attribué à une tonalité ou un BPM inconnu : ni bonus ni élimination.
UNKNOWN_SCORE = 0.5


@dataclass(frozen=True)
class Weights:
    key: float = 0.45
    bpm: float = 0.40
    energy: float = 0.15
    style: float = 0.30


@dataclass
class Transition:
    candidate: Track
    score: float
    reasons: list[str] = field(default_factory=list)


def bpm_delta_pct(current: float, candidate: float) -> float:
    """Écart relatif signé en % (négatif = plus lent), demi et double tempo compris."""
    deltas = ((candidate * factor - current) / current * 100 for factor in (0.5, 1.0, 2.0))
    return min(deltas, key=abs)


def score_transition(
    current: Track,
    candidate: Track,
    weights: Weights | None = None,
    similarity: float | None = None,
    use_style: bool = False,
) -> Transition:
    """Score entre 0 et 1.

    ``similarity`` est la similarité cosinus CLAP. Avec ``use_style``, une similarité inconnue
    (morceau pas encore indexé) reçoit le score neutre au lieu d'être ignorée.
    """
    weights = weights or Weights()
    reasons: list[str] = []
    total = 0.0
    used = 0.0

    a, b = current.camelot, candidate.camelot
    if a and b:
        key_score = compatibility(a, b)
        reasons.append(f"tonalité {a} → {b} ({key_score:.1f})")
    else:
        key_score = UNKNOWN_SCORE
        reasons.append("tonalité inconnue")
    total += weights.key * key_score
    used += weights.key

    if current.bpm and candidate.bpm:
        delta = bpm_delta_pct(current.bpm, candidate.bpm)
        bpm_score = max(0.0, 1 - abs(delta) / MAX_BPM_DELTA_PCT)
        reasons.append(f"BPM {current.bpm:.1f} → {candidate.bpm:.1f} ({delta:+.1f} %)")
    else:
        bpm_score = UNKNOWN_SCORE
        reasons.append("BPM inconnu")
    total += weights.bpm * bpm_score
    used += weights.bpm

    # L'énergie n'est calculée qu'à partir de l'étape 1 : absente, elle est ignorée.

    if current.energy is not None and candidate.energy is not None:
        jump = candidate.energy - current.energy
        energy_score = max(0.0, 1 - abs(jump) * 2)
        total += weights.energy * energy_score
        used += weights.energy
        reasons.append(f"énergie {jump:+.2f}")

    if similarity is not None:
        total += weights.style * max(0.0, similarity)
        used += weights.style
        reasons.append(f"style {similarity:.2f}")
    elif use_style:
        total += weights.style * UNKNOWN_SCORE
        used += weights.style
        reasons.append("style inconnu")

    return Transition(candidate=candidate, score=total / used if used else 0.0, reasons=reasons)


def rank_next(
    current: Track,
    library: Iterable[Track],
    limit: int = 10,
    similarities: Mapping[str, float] | None = None,
) -> list[Transition]:
    """Classe les morceaux de la bibliothèque pour enchaîner après ``current``.

    ``similarities`` associe l'emplacement de chaque morceau à sa similarité avec ``current`` ;
    vide ou absent, le style n'entre pas dans le score.
    """
    similarities = similarities or {}
    transitions = [
        score_transition(
            current,
            track,
            similarity=similarities.get(track.location),
            use_style=bool(similarities),
        )
        for track in library
        if track is not current
    ]
    transitions.sort(key=lambda t: t.score, reverse=True)
    return transitions[:limit]
