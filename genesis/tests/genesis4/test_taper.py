import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..conftest import test_artifacts
from ...errors import NotFlatError, RecursiveLineError
from ...version4 import Drift, DuplicatedLineItem, Lattice, Undulator, Line, Quadrupole


@pytest.fixture
def flat_line():
    return Lattice(
        elements={
            "U1": Undulator(lambdau=0.1, nwig=1, aw=0.1),
            "U2": Undulator(lambdau=0.2, nwig=1, aw=0.2),
            "U3": Undulator(lambdau=0.3, nwig=1, aw=0.3),
            "LN": Line(elements=["U1", "U2", "U3"]),
        }
    )


@pytest.fixture
def non_flat_line():
    return Lattice(
        elements={
            "U1": Undulator(),
            "U2": Undulator(),
            "U3": Undulator(),
            "L1": Line(elements=["U1", "U2", "U3"]),
            "LN": Line(elements=["L1", "L1", "L1"]),
        }
    )


def test_taper_array():
    lat = Lattice(
        elements={
            "U1": Undulator(),
            "U2": Undulator(),
            "U3": Undulator(),
            "LN": Line(elements=["U1", "U2", "U3"]),
        }
    )
    lat.taper_array("LN", [1, 2, 3])
    assert lat.elements["U1"].aw == 1
    assert lat.elements["U2"].aw == 2
    assert lat.elements["U3"].aw == 3


def test_taper_linear_non_flat(non_flat_line: Lattice):
    with pytest.raises(NotFlatError):
        non_flat_line.taper_linear(
            "LN",
            aw0=1.0,
            zstart=1,
            zend=3,
            taper_start=1,
            taper_end=0,
        )


def test_taper_array_non_flat(non_flat_line: Lattice):
    with pytest.raises(NotFlatError):
        non_flat_line.taper_array("LN", [1, 2, 3])
    non_flat_line.taper_array("L1", [1, 2, 3])


def test_taper_custom_non_flat(non_flat_line: Lattice):
    with pytest.raises(NotFlatError):
        non_flat_line.taper_custom("LN", aw0=0, zstart=0, zend=1)


def test_by_z_location_located():
    lat = Lattice(
        elements={
            "U1": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U2": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U3": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "LN": Line(elements=["U1@10.0", "U2@5.", "U3@1"]),
        }
    )

    zs = lat.by_z_location("LN")
    assert np.allclose([z for z, _ in zs], [1, 5, 10])


def test_by_z_location_located_nested():
    lat = Lattice(
        elements={
            "U1": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U2": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U3": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U4": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U5": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "L1": Line(elements=["U1@30", "U2@20", "U3@10"]),
            "L2": Line(elements=["L1", "U4@2.5", "U5@5"]),
            "LN": Line(elements=["L2"]),
        }
    )

    by_z = lat.by_z_location("LN")
    assert np.allclose([z for z, _ in by_z], [2.5, 5, 10, 20, 30])
    assert [elem.label for _, elem in by_z] == ["U4", "U5", "U3", "U2", "U1"]


def test_taper_linear():
    lat = Lattice(
        elements={
            "U1": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U2": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "U3": Undulator(
                nwig=1,
                lambdau=1.0,
            ),
            "LN": Line(elements=["U1", "U2", "U3"]),
        }
    )

    zs = lat.by_z_location("LN")
    assert np.allclose([z for z, _ in zs], [1, 2, 3])
    tapers = lat.taper_linear(
        "LN",
        aw0=1.0,
        zstart=1,
        zend=3,
        taper_start=1,
        taper_end=0,
    )

    assert np.allclose(tapers, [1, 0.5, 0.0])


def test_is_flat_recursive() -> None:
    lat = Lattice(
        elements={
            "U1": Undulator(),
            "U2": Undulator(),
            "U3": Undulator(),
            "LN": Line(elements=["U1", "U2", "U3", "LN"]),
        }
    )

    with pytest.raises(RecursiveLineError):
        lat.is_flat("LN")


@pytest.mark.parametrize(
    ("lattice", "is_flat"),
    [
        pytest.param(
            Lattice(
                elements={
                    "U1": Undulator(),
                    "U2": Undulator(),
                    "U3": Undulator(),
                    "LN": Line(elements=["U1", "U2", "U3"]),
                }
            ),
            True,
            id="basic",
        ),
        pytest.param(
            Lattice(
                elements={
                    "U1": Undulator(),
                    "U2": Undulator(),
                    "U3": Undulator(),
                    "L1": Line(elements=["U1"]),
                    "L2": Line(elements=["U2"]),
                    "L3": Line(elements=["U3"]),
                    "LN": Line(elements=["L1"]),
                }
            ),
            True,
            id="flat-complicated",
        ),
        pytest.param(
            Lattice(
                elements={
                    "U1": Undulator(),
                    "U2": Undulator(),
                    "U3": Undulator(),
                    "L1": Line(elements=["U1", "U2", "U3"]),
                    "LN": Line(elements=["L1", "L1"]),
                }
            ),
            False,
            id="reuse",
        ),
    ],
)
def test_is_flat(lattice: Lattice, is_flat: bool) -> None:
    assert lattice.is_flat("LN") == is_flat


@pytest.mark.parametrize(
    ("show_labels", "normalize_aw"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_plot(
    request: pytest.FixtureRequest,
    flat_line: Lattice,
    show_labels: bool,
    normalize_aw: bool,
) -> None:
    flat_line.plot(
        "LN",
        show=False,
        show_labels=show_labels,
        normalize_aw=normalize_aw,
    )
    plt.savefig(test_artifacts / f"{request.node.name}.png")


def test_flatten_doc_example() -> None:
    UND = Undulator(label="UND")
    UNDA = Undulator(label="UNDA")
    QUAD = Quadrupole(label="QUAD")
    LA = Line(elements=["UND"], label="LA")
    LB = Line(elements=["QUAD"], label="LB")
    LN = Line(elements=["LA", "LB"], label="LN")

    lattice = Lattice.from_elements([LA, LB, LN, UND, QUAD, UNDA])
    flattened = lattice.flatten_line(
        "LN",
        start=0,
        count=1,
        in_place=False,
    )
    assert flattened == [
        [
            Undulator(label="LN0_LA0_UND0"),
            Quadrupole(label="LN0_LB0_QUAD0"),
        ]
    ]

    flattened = lattice.flatten_line(
        "LN",
        start=1,
        count=2,
        in_place=False,
    )
    assert flattened == [
        [
            Undulator(label="LN1_LA1_UND1"),
            Quadrupole(label="LN1_LB1_QUAD1"),
        ],
        [
            Undulator(label="LN2_LA1_UND1"),
            Quadrupole(label="LN2_LB1_QUAD1"),
        ],
    ]


def test_flatten_example() -> None:
    UND = Undulator(label="UND")
    UNDA = Undulator(label="UNDA")
    QUAD = Quadrupole(label="QUAD")
    LA = Line(elements=["UND"], label="LA")
    LB = Line(elements=["QUAD"], label="LB")
    LN = Line(elements=["LA", "LB", "UNDA"], label="LN")
    lattice = Lattice.from_elements([LA, LB, LN, UND, QUAD, UNDA])
    flattened = lattice.flatten_line(
        "LN",
        count=1,
        start=1,
        in_place=False,
        # line_format="{name}_{index}",
        # label_format="{name}_{index}",
    )
    assert flattened == [
        [
            Undulator(label="LN1_LA1_UND1"),
            Quadrupole(label="LN1_LB1_QUAD1"),
            Undulator(label="LN1_UNDA1"),
        ]
    ]

    flattened = lattice.flatten_line("LN", start=1, count=2, in_place=False)
    assert flattened == [
        [
            Undulator(label="LN1_LA1_UND1"),
            Quadrupole(label="LN1_LB1_QUAD1"),
            Undulator(label="LN1_UNDA1"),
        ],
        [
            Undulator(label="LN2_LA1_UND1"),
            Quadrupole(label="LN2_LB1_QUAD1"),
            Undulator(label="LN2_UNDA1"),
        ],
    ]

    lattice = Lattice.from_elements([LA, LB, LN, UND, QUAD, UNDA])
    lattice.flatten_line("LN", count=1, in_place=True)
    assert lattice.elements["LN"].elements == [
        "LN0_LA0_UND0",
        "LN0_LB0_QUAD0",
        "LN0_UNDA0",
    ]

    lattice = Lattice.from_elements([LA, LB, LN, UND, QUAD, UNDA])
    lattice.flatten_line("LN", start=1, count=2, in_place=True)
    assert lattice.elements["LN"].elements == [
        "LN1_LA1_UND1",
        "LN1_LB1_QUAD1",
        "LN1_UNDA1",
        "LN2_LA1_UND1",
        "LN2_LB1_QUAD1",
        "LN2_UNDA1",
    ]


def test_flatten_example1():
    fodo_count = 6

    lat0 = Lattice(
        elements={
            "D1": Drift(label="D1", L=0.44),
            "D2": Drift(label="D2", L=0.24),
            "QF": Quadrupole(label="QF", L=0.08, k1=2.0),
            "QD": Quadrupole(label="QD", L=0.08, k1=-2.0),
            "UND": Undulator(
                label="UND", aw=0.84853, lambdau=0.015, nwig=266, helical=True
            ),
            "FODO": Line(
                label="FODO",
                elements=["UND", "D1", "QF", "D2", "UND", "D1", "QD", "D2"],
            ),
            "FEL": Line(
                label="FEL",
                elements=[DuplicatedLineItem(label="FODO", count=fodo_count)],
            ),
        },
    )

    flattened = lat0.flatten_line(
        "FEL",
        count=1,
        start=1,
        in_place=False,
        line_format="{name}{index}",
        label_format="{name}_{index}",
    )
    flattened_names = [ele.label for section in flattened for ele in section]
    for ele in lat0.elements["FODO"].elements:
        assert f"FEL1_FODO1_{ele}_1" in flattened_names

    assert "FEL1_FODO1_D1_1" in flattened_names
    assert "FEL2_FODO1_UND_1" not in flattened_names

    assert len(flattened_names) == fodo_count * len(lat0.elements["FODO"].elements)

    print("Flattened:")
    for name in flattened_names:
        print(name)

    flattened = lat0.flatten_line(
        "FEL",
        count=2,
        start=1,
        in_place=False,
        line_format="{name}{index}",
        label_format="{name}_{index}",
    )
    flattened_names = [ele.label for section in flattened for ele in section]
    assert "FEL1_FODO1_UND_1" in flattened_names
    assert "FEL2_FODO1_UND_1" in flattened_names
    assert "FEL2_FODO1_UND_2" in flattened_names
    assert "FEL2_FODO2_UND_1" in flattened_names

    for ele in lat0.elements["FODO"].elements:
        for idx in range(1, 6):
            assert f"FEL1_FODO{idx}_{ele}_1" in flattened_names
            assert f"FEL2_FODO{idx}_{ele}_1" in flattened_names

    assert len(flattened_names) == fodo_count * len(lat0.elements["FODO"].elements) * 2
