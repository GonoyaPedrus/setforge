import numpy as np
import pytest

from setforge.embeddings import CLAP_SAMPLE_RATE, normalize, windows


def test_windows_cover_the_body_of_the_track() -> None:
    sr = 10
    y = np.arange(1000, dtype=np.float32)  # 100 s à 10 Hz
    parts = windows(y, sr, seconds=10)
    assert [len(p) for p in parts] == [100, 100, 100]
    assert [int(p[0]) for p in parts] == [200, 450, 700]


def test_windows_short_track_is_kept_whole() -> None:
    y = np.ones(50, dtype=np.float32)
    assert [len(p) for p in windows(y, 10, seconds=10)] == [50]


def test_normalize_averages_and_scales() -> None:
    vector = normalize(np.array([[3.0, 0.0], [3.0, 8.0]]))
    assert np.allclose(vector, [0.6, 0.8])


def test_normalize_rejects_zero() -> None:
    with pytest.raises(ValueError):
        normalize(np.zeros((1, 3)))


class _Tokens(dict):  # type: ignore[type-arg]
    def to(self, device: object) -> "_Tokens":
        return self


def test_clap_embedder_on_tiny_random_model() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from setforge.embeddings import ClapEmbedder

    config = transformers.ClapConfig(
        text_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 37,
            "vocab_size": 100,
        },
        audio_config={
            "hidden_size": 32,
            "depths": [1, 1],
            "num_attention_heads": [2, 2],
            "patch_embeds_hidden_size": 16,
            "window_size": 4,
            "enable_fusion": False,
        },
        projection_dim=16,
    )
    model = transformers.ClapModel(config)
    extractor = transformers.ClapFeatureExtractor(truncation="rand_trunc")

    def tokenizer(texts: list[str], **_: object) -> _Tokens:
        ids = torch.tensor([[1, 5, 7, 2]] * len(texts))
        return _Tokens(input_ids=ids, attention_mask=torch.ones_like(ids))

    embedder = ClapEmbedder(model, extractor, tokenizer, name="tiny")
    audio = embedder.embed_audio(np.random.default_rng(0).standard_normal(CLAP_SAMPLE_RATE * 30))
    text = embedder.embed_text("deep house sombre")
    assert audio.shape == text.shape == (16,)
    assert np.linalg.norm(audio) == pytest.approx(1, abs=1e-5)
    assert np.linalg.norm(text) == pytest.approx(1, abs=1e-5)
