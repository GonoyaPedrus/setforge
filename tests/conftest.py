from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def collection_xml() -> Path:
    return FIXTURES / "collection.xml"
