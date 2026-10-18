"""Descripteurs audio d'un morceau : tempo, tonalité et énergie brute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 22_050
# Plage de tempo DJ : un tempo détecté hors plage est replié par octave (62 -> 124, 248 -> 124).
BPM_RANGE = (70.0, 180.0)
# Fréquence sous laquelle on mesure le poids des basses (kick + basse).
LOW_BAND_HZ = 150.0

# Profils de Krumhansl-Kessler, à partir de la tonique.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

Signal = npt.NDArray[np.float32]


@dataclass(frozen=True)
class AudioFeatures:
    bpm: float
    key: str
    key_confidence: float
    rms_db: float
    onset_rate: float
    flux: float
    low_ratio: float


def fold_bpm(bpm: float, low: float = BPM_RANGE[0], high: float = BPM_RANGE[1]) -> float:
    """Replie un tempo par octave dans [low, high)."""
    if bpm <= 0:
        raise ValueError("tempo nul ou négatif")
    while bpm < low:
        bpm *= 2
    while bpm >= high:
        bpm /= 2
    return bpm


def estimate_key(chroma: npt.NDArray[np.floating]) -> tuple[str, float]:
    """Tonalité par corrélation du chroma moyen avec les 24 profils.

    Renvoie la tonalité au format Rekordbox (« Am », « F# ») et l'écart de corrélation
    entre la meilleure et la deuxième hypothèse, utilisable comme confiance.
    """
    profile = chroma.mean(axis=1)
    scores: list[tuple[float, str]] = []
    for tonic in range(12):
        for template, suffix in ((_MAJOR_PROFILE, ""), (_MINOR_PROFILE, "m")):
            correlation = float(np.corrcoef(profile, np.roll(template, tonic))[0, 1])
            scores.append((correlation, f"{_PITCHES[tonic]}{suffix}"))
    scores.sort(reverse=True)
    (best, key), (second, _) = scores[0], scores[1]
    return key, max(0.0, best - second)


def analyze_signal(y: Signal, sr: int = SAMPLE_RATE) -> AudioFeatures:
    if y.size < sr:
        raise ValueError("signal trop court pour être analysé (moins d'une seconde)")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = fold_bpm(float(np.atleast_1d(tempo)[0]))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, confidence = estimate_key(chroma)

    rms = librosa.feature.rms(y=y)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    spectrum = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(spectrum.sum()) or 1.0

    return AudioFeatures(
        bpm=round(bpm, 2),
        key=key,
        key_confidence=round(confidence, 4),
        rms_db=float(20 * np.log10(max(float(rms.mean()), 1e-10))),
        onset_rate=len(onsets) / (y.size / sr),
        flux=float(onset_env.mean()),
        low_ratio=float(spectrum[freqs < LOW_BAND_HZ].sum()) / total,
    )


def analyze_file(path: Path) -> AudioFeatures:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return analyze_signal(y, int(sr))
