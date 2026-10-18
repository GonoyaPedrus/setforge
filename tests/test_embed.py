from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lancedb")

from setforge.analysis.embed import embed_library
from setforge.index import TrackIndex
from setforge.models import Track


class FakeEmbedder:
    name = "fake"

    def embed_file(self, path: Path) -> np.ndarray:
        if path.name == "broken.mp3":
            raise RuntimeError("décodage impossible")
        return np.array([1.0, 0.0], dtype=np.float32)


def test_embed_library_caches_by_file_and_model(tmp_path: Path) -> None:
    files = [tmp_path / name for name in ("a.mp3", "b.mp3", "broken.mp3")]
    for path in files:
        path.write_text("x")
    tracks = [Track(track_id=p.stem, title=p.stem, location=p.as_uri()) for p in files]
    tracks.append(Track(track_id="absent", title="absent", location="file:///nulle/part.mp3"))
    tracks.append(tracks[0].model_copy(update={"track_id": "doublon"}))
    index = TrackIndex(tmp_path / "idx")

    report = embed_library(tracks, index, FakeEmbedder())
    assert (report.embedded, report.cached) == (2, 0)
    assert list(report.failed.values()) == ["décodage impossible"]
    assert len(index) == 2

    again = embed_library(tracks, index, FakeEmbedder())
    assert (again.embedded, again.cached) == (0, 3)  # le doublon compte comme à jour

    other = FakeEmbedder()
    other.name = "autre-modele"
    assert embed_library(tracks, index, other).embedded == 2
