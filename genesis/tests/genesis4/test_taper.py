import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..conftest import test_artifacts
from ...errors import NotFlatError, RecursiveLineError
from ...version4 import Lattice, Undulator, Line


@pytest.fixture
def flat_line():
    return Lattice(
        elements={
            "U1": Undulator(lambdau=0.1, nwig=1),
            "U2": Undulator(lambdau=0.1, nwig=1),
            "U3": Undulator(lambdau=0.1, nwig=1),
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


def test_plot_taper(flat_line: Lattice) -> None:
    flat_line.plot_beamline_taper("LN", show=False)
    plt.savefig(test_artifacts / "test_plot_taper.png")
