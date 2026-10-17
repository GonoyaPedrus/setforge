"""Score de transition entre deux morceaux (MVP sans embeddings)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from setforge.camelot import compatibility
from setforge.models import Track

# Écart de tempo au-delà duquel le pitch devient audible sur la plupart des morceaux.
MAX_BPM_DELTA_PCT = 6.0


@dataclass(frozen=True)
class Weights:
    key: float = 0.45
    bpm: float = 0.40
    energy: float = 0.15


@dataclass
class Transition:
    candidate: Track
    score: float
    reasons: list[str] = field(default_factory=list)


def bpm_delta_pct(current: float, candidate: float) -> float:
    """Écart relatif en %, en tenant compte du demi et du double tempo."""
    return min(abs(candidate * factor - current) / current * 100 for factor in (0.5, 1.0, 2.0))


def score_transition(
    current: Track, candidate: Track, weights: Weights | None = None
) -> Transition:
    weights = weights or Weights()
    reasons: list[str] = []
    total = 0.0
    used = 0.0

    a, b = current.camelot, candidate.camelot
    if a and b:
        key_score = compatibility(a, b)
        total += weights.key * key_score
        used += weights.key
        reasons.append(f"tonalité {a} → {b} ({key_score:.1f})")

    if current.bpm and candidate.bpm:
        delta = bpm_delta_pct(current.bpm, candidate.bpm)
        bpm_score = max(0.0, 1 - delta / MAX_BPM_DELTA_PCT)
        total += weights.bpm * bpm_score
        used += weights.bpm
        reasons.append(f"BPM {current.bpm:.1f} → {candidate.bpm:.1f} ({delta:+.1f} %)")

    if current.energy is not None and candidate.energy is not None:
        jump = candidate.energy - current.energy
        energy_score = max(0.0, 1 - abs(jump) * 2)
        total += weights.energy * energy_score
        used += weights.energy
        reasons.append(f"énergie {jump:+.2f}")

    return Transition(candidate=candidate, score=total / used if used else 0.0, reasons=reasons)


def rank_next(current: Track, library: Iterable[Track], limit: int = 10) -> list[Transition]:
    """Classe les morceaux de la bibliothèque pour enchaîner après ``current``."""
    transitions = [
        score_transition(current, track) for track in library if track.track_id != current.track_id
    ]
    transitions.sort(key=lambda t: t.score, reverse=True)
    return transitions[:limit]
