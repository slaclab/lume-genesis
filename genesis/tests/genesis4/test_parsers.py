import pathlib
import pprint

import lark
import pytest

from ...version4.grammar import new_lattice_parser, Lattice


TEST_V4_ROOT = pathlib.Path(__file__).resolve().parent
genesis4_lattices = tuple(TEST_V4_ROOT.glob("*.lat"))


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
    print(str(lattice))
