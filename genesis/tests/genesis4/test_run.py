import pathlib
from typing import Optional

import pytest

from ..conftest import test_root

from ...version4 import Genesis4
from ...version4.input import Beam, Lattice, MainInput
from .conftest import run_basic
from .util import run_with_instances, run_with_source


def test_compare_lattice_to_file(lattice: Lattice):
    original_lattice = Lattice.from_file(run_basic / "hxr.lat")
    print("The original lattice file is parsed as follows:")
    print(repr(original_lattice))
    print()

    print("The new, Python-based lattice looks like this:")
    print(lattice)

    assert lattice.to_genesis() == original_lattice.to_genesis()
    print("Verified they are the same when made into Genesis 4 format.")

    print("The LCLS2_HXR_U2 element is made up of the following:")
    for item in lattice.elements["LCLS2_HXR_U2"].elements:
        print("-", item)


def test_check_references(main_input: MainInput):
    beam: Beam = main_input.by_namelist[Beam][0]
    assert str(beam.current) == "@beamcurrent"
    assert str(beam.gamma) == "@beamgamma"
    assert "current = @beamcurrent" in str(main_input)
    assert "gamma = @beamgamma" in str(main_input)


@pytest.mark.parametrize(
    "timeout",
    [
        pytest.param(30, id="no-timeout"),
        pytest.param(None, id="with-timeout"),
    ],
)
def test_run_with_instances(
    main_input: MainInput,
    lattice: Lattice,
    timeout: Optional[float],
) -> None:
    run_with_instances(main_input, lattice, timeout=timeout)


def test_run_with_instances_check_output(
    main_input: MainInput,
    lattice: Lattice,
) -> None:
    genesis = run_with_instances(main_input, lattice)
    fig = genesis.plot(return_figure=True)
    assert fig is not None
    fig.savefig(test_root / "test_output_plot_smoke.png")
    genesis.stat("beam_xsize")


def test_run_with_source(
    lattice: Lattice,
    main_input: MainInput,
    tmp_path: pathlib.Path,
):
    run_with_source(
        lattice=lattice, main_input=main_input, workdir=tmp_path, source_path=run_basic
    )


@pytest.mark.parametrize(
    "mpi",
    [
        False,
        True,
    ],
)
def test_get_executable_smoke(genesis4: Genesis4, mpi: bool) -> None:
    genesis4.use_mpi = mpi
    assert genesis4.get_executable()


@pytest.mark.parametrize(
    "mpi, nproc",
    [
        (False, 0),
        (False, 1),
        (False, 4),
        (True, 0),
        (True, 1),
        (True, 4),
    ],
)
def test_get_run_prefix_smoke(genesis4: Genesis4, nproc: int, mpi: bool) -> None:
    genesis4.nproc = nproc
    genesis4.use_mpi = mpi
    assert genesis4.get_run_prefix()
