"""Descripteurs audio d'un morceau : tempo, tonalité et énergie brute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt

from setforge.camelot import parse_key

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
    bpm: float | None  # None : aucune pulsation détectée (ambient, intro sans kick)
    key: str | None  # None : chroma plat (silence, bruit)
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


def _camelot_number(key: str) -> int:
    camelot = parse_key(key)
    assert camelot is not None  # les noms de _PITCHES sont tous reconnus
    return camelot.number


def estimate_key(chroma: npt.NDArray[np.floating]) -> tuple[str | None, float]:
    """Tonalité par corrélation du chroma moyen avec les 24 profils.

    Renvoie la tonalité au format Rekordbox (« Am », « F# ») et sa confiance : l'écart de
    corrélation avec la meilleure hypothèse d'un autre numéro Camelot. Le relatif
    majeur/mineur, souvent très proche, est compatible et ne réduit donc pas la confiance.
    """
    profile = chroma.mean(axis=1)
    if not np.isfinite(profile).all() or np.ptp(profile) < 1e-9:
        return None, 0.0
    scores: list[tuple[float, str]] = []
    for tonic in range(12):
        for template, suffix in ((_MAJOR_PROFILE, ""), (_MINOR_PROFILE, "m")):
            correlation = float(np.corrcoef(profile, np.roll(template, tonic))[0, 1])
            scores.append((correlation, f"{_PITCHES[tonic]}{suffix}"))
    scores.sort(reverse=True)
    best, key = scores[0]
    number = _camelot_number(key)
    rival = next(score for score, name in scores if _camelot_number(name) != number)
    return key, max(0.0, best - rival)


def analyze_signal(y: Signal, sr: int = SAMPLE_RATE) -> AudioFeatures:
    if y.size < sr:
        raise ValueError("signal trop court pour être analysé (moins d'une seconde)")
    # Un seul STFT pour l'enveloppe d'onsets, le RMS et le poids des basses.
    magnitude = np.abs(librosa.stft(y))
    power = magnitude**2
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=sr))
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    raw_bpm = float(np.atleast_1d(tempo)[0])
    bpm = round(fold_bpm(raw_bpm), 2) if raw_bpm > 0 else None

    key, confidence = estimate_key(librosa.feature.chroma_cqt(y=y, sr=sr))

    rms = librosa.feature.rms(S=magnitude)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(power.sum()) or 1.0

    return AudioFeatures(
        bpm=bpm,
        key=key,
        key_confidence=round(confidence, 4),
        rms_db=float(20 * np.log10(max(float(rms.mean()), 1e-10))),
        onset_rate=len(onsets) / (y.size / sr),
        flux=float(onset_env.mean()),
        # Gardé pour la détection de structure (étape 2), pas encore dans le score d'énergie.
        low_ratio=float(power[freqs < LOW_BAND_HZ].sum()) / total,
    )


def analyze_file(path: Path) -> AudioFeatures:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return analyze_signal(y, int(sr))
