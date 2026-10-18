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
    assert index.search(_unit(1, 0), "m", 5) == []
    assert index.vectors("m") == {}


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
    hits = reopened.search(_unit(1, 0), "m", 2)
    assert [location for location, _ in hits] == ["a", "b"]
    assert hits[0][1] == pytest.approx(0.995, abs=1e-3)
    assert len(reopened.search(_unit(1, 0), "m")) == 3
    assert np.allclose(reopened.vectors("m")["c"], _unit(0, 1))


def test_other_models_are_kept_apart(tmp_path: Path) -> None:
    index = TrackIndex(tmp_path / "idx")
    index.upsert([Embedding("a", 1.0, 10, "m", _unit(1, 0))])
    index.upsert([Embedding("b", 1.0, 10, "o'ther", _unit(1, 0))])
    assert list(index.vectors("m")) == ["a"]
    assert [loc for loc, _ in index.search(_unit(1, 0), "o'ther")] == ["b"]


def test_new_dimension_replaces_the_index(tmp_path: Path) -> None:
    index = TrackIndex(tmp_path / "idx")
    index.upsert([Embedding("a", 1.0, 10, "small", _unit(1, 0))])
    index.upsert([Embedding("b", 1.0, 10, "big", _unit(1, 0, 0))])
    assert len(index) == 1
    assert list(index.vectors("big")) == ["b"]


def test_similarities() -> None:
    scores = similarities(_unit(1, 0), {"a": _unit(1, 0), "c": _unit(0, 1)})
    assert scores == pytest.approx({"a": 1.0, "c": 0.0})
    assert similarities(_unit(1, 0), {}) == {}
