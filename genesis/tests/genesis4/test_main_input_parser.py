import pathlib
import pprint

import lark
import pytest

from ...version4.input.grammar import new_main_input_parser
from ...version4.input import MainInput


test_v4_root = pathlib.Path(__file__).resolve().parent
genesis4_input_files = tuple(test_v4_root.glob("*.in"))


@pytest.fixture(scope="module")
def main_input_parser() -> lark.Lark:
    return new_main_input_parser()


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("&namelist\n&end"),
        pytest.param("&namelist\nvar=value\n&end"),
        pytest.param("&namelist\n  var=value\n&end"),
        pytest.param("&namelist\n  var=value \n&end\n"),
    ],
)
def test_elements(main_input_parser: lark.Lark, source: str) -> None:
    print(main_input_parser.parse(source))


@pytest.mark.parametrize(
    "filename", [pytest.param(fn, id=fn.name) for fn in genesis4_input_files]
)
def test_load_file(main_input_parser: lark.Lark, filename: pathlib.Path) -> None:
    with open(filename) as fp:
        contents = fp.read()
    print(f"\n\nInput ({filename})")
    print(contents)

    print("\n\nAs a lark Tree:")
    print(main_input_parser.parse(contents))

    print("\n\nAs dataclasses:")
    inp = MainInput.from_file(filename)
    pprint.pprint(inp, width=1)

    print("\n\nSerialized back to a file:")
    round_tripped = str(inp)
    print(round_tripped)

    print("\n\nChecking file output vs initial dataclasses..")
    second_inp = MainInput.from_contents(round_tripped)
    assert str(inp) == str(second_inp)
