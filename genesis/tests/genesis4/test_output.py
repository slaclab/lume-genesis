from typing import Dict, List

import pytest
from ...version4.parsers import extract_aliases


@pytest.mark.parametrize(
    "keys, expected_aliases",
    [
        pytest.param(
            ["Beam/Global/xsize", "Beam/xsize", "Field/Global/xsize", "Field/xsize"],
            {
                "beam_global_xsize": "Beam/Global/xsize",
                "beam_xsize": "Beam/xsize",
                "field_global_xsize": "Field/Global/xsize",
                "field_xsize": "Field/xsize",
            },
            id="xsize",
        ),
        pytest.param(
            [
                "Beam/Global/xsize",
                "Beam/xsize",
                "Field/Global/xsize",
                "Field/xsize",
                "test/unique",
                "foobar",
            ],
            {
                "beam_global_xsize": "Beam/Global/xsize",
                "beam_xsize": "Beam/xsize",
                "field_global_xsize": "Field/Global/xsize",
                "field_xsize": "Field/xsize",
                "unique": "test/unique",
                "test_unique": "test/unique",
            },
            id="xsize-and-unique",
        ),
    ],
)
def test_extract_aliases(keys: List[str], expected_aliases: Dict[str, str]) -> None:
    assert extract_aliases(dict.fromkeys(keys)) == expected_aliases
