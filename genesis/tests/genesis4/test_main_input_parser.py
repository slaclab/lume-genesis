import pathlib
import pprint

import lark
import pytest

from ...version4 import Genesis4, MainInput
from ...version4.input.core import new_main_input_parser

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
        pytest.param("&namelist\n  # comment \n  var=value \n&end\n"),
    ],
)
def test_elements(main_input_parser: lark.Lark, source: str) -> None:
    print(main_input_parser.parse(source))


def test_inline_comment(main_input_parser: lark.Lark) -> None:
    namelist = main_input_parser.parse(
        "&namelist\n  # comment \n  var=value# comment\n&end\n"
    )
    parameter_set = list(namelist.find_data("parameter_set"))[0]
    var, value = parameter_set.children
    assert str(var) == "var"
    assert str(value) == "value"


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
    round_tripped = inp.to_genesis()
    print(round_tripped)

    print("\n\nChecking file output vs initial dataclasses..")
    second_inp = MainInput.from_contents(round_tripped)
    assert inp.to_genesis() == second_inp.to_genesis()


@pytest.mark.parametrize(
    "filename", [pytest.param(fn, id=fn.name) for fn in genesis4_input_files[:1]]
)
def test_load_file_with_input_parser_helper(filename: pathlib.Path) -> None:
    assert isinstance(Genesis4.input_parser(filename), MainInput)


@pytest.mark.parametrize(
    "filename", [pytest.param(fn, id=fn.name) for fn in genesis4_input_files]
)
def test_serialize_file(filename: pathlib.Path) -> None:
    inp = MainInput.from_file(filename)

    print(filename)
    pprint.pprint(inp)

    print("Is serialized as follows:")
    serialized = inp.to_dicts()
    pprint.pprint(serialized)

    print("Deserialized back to dataclasses:")
    deserialized = MainInput.from_dicts(serialized, filename=filename)
    pprint.pprint(deserialized)
    try:
        assert inp == deserialized
    except AssertionError:
        for idx, (inp_namelist, deserialized_namelist) in enumerate(
            zip(inp.namelists, deserialized.namelists)
        ):
            if type(inp_namelist) is not type(deserialized_namelist):
                print(
                    f"Namelist {idx} differ by type: {type(inp_namelist)}, {type(deserialized_namelist)}"
                )
                continue
            for attr, value1 in inp_namelist.model_dump().items():
                value2 = deserialized_namelist.model_dump()[attr]
                if value1 != value2:
                    print(f"Namelist {idx} differs with attribute {attr}:")
                    print("Input:\n", value1)
                    print("Deserialized:\n", value2)
        raise
