import numpy as np
import pytest

pytest.importorskip("librosa")

from setforge.analysis.features import SAMPLE_RATE, analyze_signal, estimate_key, fold_bpm

A, C, E = 220.0, 261.63, 329.63


def _synth(bpm: float, chord: tuple[float, ...], seconds: float = 20.0) -> np.ndarray:
    """Accord tenu + kick (sinus 60 Hz amorti) sur chaque temps."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    signal = sum(np.sin(2 * np.pi * f * t) for f in chord) * 0.1
    kick_len = int(0.12 * SAMPLE_RATE)
    kick_t = np.arange(kick_len) / SAMPLE_RATE
    kick = np.sin(2 * np.pi * 60 * kick_t) * np.exp(-kick_t * 30)
    for beat in np.arange(0, seconds, 60 / bpm):
        start = int(beat * SAMPLE_RATE)
        end = min(start + kick_len, t.size)
        signal[start:end] += kick[: end - start]
    return signal.astype(np.float32)


@pytest.mark.parametrize(("raw", "folded"), [(62.0, 124.0), (248.0, 124.0), (174.0, 174.0)])
def test_fold_bpm(raw: float, folded: float) -> None:
    assert fold_bpm(raw) == folded


def test_fold_bpm_rejects_zero() -> None:
    with pytest.raises(ValueError):
        fold_bpm(0)


def test_estimate_key_from_chroma() -> None:
    chroma = np.full((12, 4), 0.1)
    chroma[[9, 0, 4]] = 1.0  # A, C, E
    key, confidence = estimate_key(chroma)
    assert key == "Am"
    assert confidence > 0


def test_analyze_signal_finds_tempo_and_key() -> None:
    features = analyze_signal(_synth(124, (A, C, E)))
    assert features.bpm == pytest.approx(124, abs=2)
    assert features.key == "Am"
    assert 0 < features.low_ratio < 1
    assert features.onset_rate > 1


def test_analyze_signal_rejects_short_audio() -> None:
    with pytest.raises(ValueError):
        analyze_signal(np.zeros(100, dtype=np.float32))
