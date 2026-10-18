from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lancedb")

from setforge.index import Embedding, TrackIndex, similarities


def _unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_empty_index(tmp_path: Path) -> None:
    index = TrackIndex(tmp_path / "idx")
    assert len(index) == 0
    assert index.fingerprints() == {}
    assert index.search(_unit(1, 0), 5) == []


def test_upsert_search_and_reopen(tmp_path: Path) -> None:
    index = TrackIndex(tmp_path / "idx")
    index.upsert(
        [
            Embedding("a", 1.0, 10, "m", _unit(1, 0)),
            Embedding("b", 1.0, 10, "m", _unit(1, 1)),
            Embedding("c", 1.0, 10, "m", _unit(0, 1)),
        ]
    )
    index.upsert([Embedding("a", 2.0, 11, "m", _unit(1, 0.1))])

    reopened = TrackIndex(tmp_path / "idx")
    assert len(reopened) == 3
    assert reopened.fingerprints()["a"] == (2.0, 11, "m")
    hits = reopened.search(_unit(1, 0), 2)
    assert [location for location, _ in hits] == ["a", "b"]
    assert hits[0][1] == pytest.approx(0.995, abs=1e-3)


def test_similarities() -> None:
    scores = similarities(_unit(1, 0), {"a": _unit(1, 0), "c": _unit(0, 1)})
    assert scores == pytest.approx({"a": 1.0, "c": 0.0})
    assert similarities(_unit(1, 0), {}) == {}
