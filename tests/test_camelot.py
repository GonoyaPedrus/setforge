import pytest

from setforge.camelot import CamelotKey, compatibility, parse_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Am", "8A"),
        ("C", "8B"),
        ("F#m", "11A"),
        ("Gbm", "11A"),
        ("Db", "3B"),
        ("Bb minor", "3A"),
        ("8a", "8A"),
        ("12B", "12B"),
        ("E♭", "5B"),
    ],
)
def test_parse_key(raw: str, expected: str) -> None:
    assert str(parse_key(raw)) == expected


@pytest.mark.parametrize("raw", ["", "H", "13A", "Cb", "not a key"])
def test_parse_key_rejects_unknown(raw: str) -> None:
    assert parse_key(raw) is None


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("8A", "8A", 1.0),
        ("8A", "8B", 0.9),
        ("8A", "9A", 0.9),
        ("12A", "1A", 0.9),
        ("8A", "9B", 0.6),
        ("8A", "10A", 0.5),
        ("8A", "3A", 0.0),
    ],
)
def test_compatibility(a: str, b: str, expected: float) -> None:
    key_a, key_b = parse_key(a), parse_key(b)
    assert isinstance(key_a, CamelotKey)
    assert isinstance(key_b, CamelotKey)
    assert compatibility(key_a, key_b) == expected
