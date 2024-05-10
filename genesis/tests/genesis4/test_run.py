import pathlib


from ...version4.input import (
    Beam,
    Lattice,
    MainInput,
)
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


def test_run_with_instances(main_input: MainInput, lattice: Lattice) -> None:
    run_with_instances(main_input, lattice)


def test_run_with_source(
    lattice: Lattice,
    main_input: MainInput,
    tmp_path: pathlib.Path,
):
    run_with_source(
        lattice=lattice, main_input=main_input, workdir=tmp_path, source_path=run_basic
    )
