"""Embeddings audio et texte CLAP, dans un même espace vectoriel.

Nécessite l'extra ``clap`` (torch, transformers). Modèle par défaut : LAION
``larger_clap_music`` (Apache-2.0), téléchargé depuis Hugging Face au premier usage.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_MODEL = "laion/larger_clap_music"
CLAP_SAMPLE_RATE = 48_000
# CLAP lit 10 s d'audio : on en prend plusieurs dans le corps du morceau, loin de l'intro.
WINDOW_SECONDS = 10.0
WINDOW_POSITIONS = (0.25, 0.5, 0.75)

Vector = npt.NDArray[np.float32]


def window_offsets(
    duration: float,
    seconds: float = WINDOW_SECONDS,
    positions: tuple[float, ...] = WINDOW_POSITIONS,
) -> list[float]:
    """Débuts (en secondes) des extraits centrés sur chaque position relative du morceau."""
    if duration <= seconds:
        return [0.0]
    return sorted(
        {min(max(p * duration - seconds / 2, 0.0), duration - seconds) for p in positions}
    )


def windows(
    y: npt.NDArray[np.float32],
    sr: int,
    seconds: float = WINDOW_SECONDS,
    positions: tuple[float, ...] = WINDOW_POSITIONS,
) -> list[npt.NDArray[np.float32]]:
    """Extraits de ``seconds`` secondes centrés sur chaque position relative du morceau."""
    size = int(seconds * sr)
    if y.size <= size:
        return [y]
    starts = sorted({min(max(int(p * y.size) - size // 2, 0), y.size - size) for p in positions})
    return [y[start : start + size] for start in starts]


def normalize(vectors: npt.NDArray[np.floating]) -> Vector:
    """Moyenne des vecteurs, normalisée L2 (similarité cosinus = produit scalaire)."""
    mean = np.asarray(vectors, dtype=np.float32).reshape(-1, vectors.shape[-1]).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        raise ValueError("embedding nul")
    return (mean / norm).astype(np.float32)


def _pooled(output: Any) -> Any:
    # transformers 5 renvoie un ModelOutput, la 4.x directement le tenseur projeté.
    return getattr(output, "pooler_output", output)


class ClapEmbedder:
    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        tokenizer: Callable[..., Any],
        name: str = DEFAULT_MODEL,
    ) -> None:
        import torch

        self.name = name
        self._torch = torch
        self._model = model.eval()
        self._device = next(model.parameters()).device
        self._feature_extractor = feature_extractor
        self._tokenizer = tokenizer

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL) -> ClapEmbedder:
        """Charge le modèle (téléchargé depuis Hugging Face au premier appel)."""
        import torch
        from transformers import ClapModel, ClapProcessor

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        processor: Any = ClapProcessor.from_pretrained(name)
        model: Any = ClapModel.from_pretrained(name)
        model = model.to(device)
        return cls(model, processor.feature_extractor, processor.tokenizer, name)

    def embed_audio(self, y: npt.NDArray[np.float32]) -> Vector:
        """Embedding d'un signal mono à 48 kHz."""
        return self._embed_windows(windows(y, CLAP_SAMPLE_RATE))

    def _embed_windows(self, parts: list[npt.NDArray[np.float32]]) -> Vector:
        inputs = self._feature_extractor(parts, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        with self._torch.no_grad():
            output = self._model.get_audio_features(**inputs.to(self._device))
        return normalize(_pooled(output).cpu().numpy())

    def embed_file(self, path: Path) -> Vector:
        """Ne décode que les extraits utiles, pas le morceau entier."""
        import librosa

        duration = librosa.get_duration(path=path)
        parts = [
            librosa.load(
                path, sr=CLAP_SAMPLE_RATE, mono=True, offset=offset, duration=WINDOW_SECONDS
            )[0]
            for offset in window_offsets(duration)
        ]
        return self._embed_windows(parts)

    def embed_text(self, text: str) -> Vector:
        inputs = self._tokenizer([text], padding=True, return_tensors="pt")
        with self._torch.no_grad():
            output = self._model.get_text_features(**inputs.to(self._device))
        return normalize(_pooled(output).cpu().numpy())
