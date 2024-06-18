import pathlib
import pprint

import lark
import pytest

from ...version4 import Genesis4, Lattice
from ...version4.input.parsers import new_lattice_parser

test_v4_root = pathlib.Path(__file__).resolve().parent
genesis4_lattices = tuple(test_v4_root.glob("*.lat"))


@pytest.fixture(scope="module")
def lattice_parser() -> lark.Lark:
    return new_lattice_parser()


@pytest.mark.parametrize(
    "element_source",
    [
        pytest.param("D1: DRIFT = { l = 0.44};"),
        pytest.param("D2: DRIFT = { l = 0.24};"),
        pytest.param("QF: QUADRUPOLE = { l = 0.080000, k1= 2.000000 };"),
        pytest.param("QD: QUADRUPOLE = { l = 0.080000, k1= -2.000000 };"),
        pytest.param("QD: QUAD= { l = 0.080000, k1= -2.000000 };"),
        pytest.param(
            "UND: UNDULATOR = { lambdau=0.015000, nwig=266, aw=0.84853, helical= True};"
        ),
        pytest.param("FODO: LINE={UND,D1,QF,D2,UND,D1,QD,D2};"),
        pytest.param("FEL: LINE={6*FODO};"),
        pytest.param("FODO: Line = {UND,F@0.2,D@1.2,M@2.0}; "),
        pytest.param("Line1: LINE= {UND}; Line2: Line = {Line1};"),
    ],
)
def test_elements(lattice_parser: lark.Lark, element_source: str) -> None:
    print(lattice_parser.parse(element_source))


@pytest.mark.parametrize(
    "filename", [pytest.param(fn, id=fn.name) for fn in genesis4_lattices]
)
def test_load_file(lattice_parser: lark.Lark, filename: pathlib.Path) -> None:
    with open(filename) as fp:
        contents = fp.read()
    print(f"\n\nLattice ({filename})")
    print(contents)

    print("\n\nAs a lark Tree:")
    print(lattice_parser.parse(contents))

    print("\n\nAs dataclasses:")
    lattice = Lattice.from_file(filename)
    pprint.pprint(lattice, width=1)

    print("\n\nSerialized back to a file:")
    round_tripped = lattice.to_genesis()
    print(round_tripped)

    print("\n\nChecking file output vs initial dataclasses..")
    second_lattice = Lattice.from_contents(round_tripped)
    assert lattice.to_genesis() == second_lattice.to_genesis()


@pytest.mark.parametrize(
    "filename", [pytest.param(fn, id=fn.name) for fn in genesis4_lattices[:1]]
)
def test_load_file_with_input_parser_helper(filename: pathlib.Path) -> None:
    assert isinstance(Genesis4.lattice_parser(filename), Lattice)


@pytest.mark.parametrize(
    "filename", [pytest.param(fn, id=fn.name) for fn in genesis4_lattices]
)
def test_serialize(filename: pathlib.Path) -> None:
    lattice = Lattice.from_file(filename)

    print("This lattice:")
    pprint.pprint(lattice)

    print("Is serialized as follows:")
    serialized = lattice.model_dump()
    pprint.pprint(serialized)

    print("Deserialized back to dataclasses:")
    deserialized = Lattice.model_validate(serialized)
    assert lattice == deserialized
