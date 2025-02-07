import pathlib

import pydantic
import pydantic.alias_generators
import pytest

from ...errors import NoSetupNamelistError

from ... import version4 as g4
from ...version4 import AnyBeamlineElement, AnyNameList, Lattice, MainInput, Setup


def test_namelist_output():
    setup = Setup(
        rootname="Benchmark",
        lattice="Aramis.lat",
        beamline="ARAMIS",
        lambda0=1e-10,
        gamma0=11357.82,
        delz=0.045,
        shotnoise=False,
        beam_global_stat=True,
        field_global_stat=True,
    )
    assert (
        setup.to_genesis()
        == """
&setup
  rootname = Benchmark
  lattice = Aramis.lat
  beamline = ARAMIS
  gamma0 = 11357.82
  delz = 0.045
  shotnoise = false
  beam_global_stat = true
  field_global_stat = true
&end
""".strip()
    )


def test_main_input_helpers(namelist: AnyNameList):
    main = MainInput(namelists=[namelist])

    attr_base = pydantic.alias_generators.to_snake(namelist.__class__.__name__)
    attr = f"{attr_base}s" if not attr_base.endswith("s") else attr_base

    if attr_base in {"setup", "initial_particles"}:
        assert getattr(main, attr_base) == namelist
    elif attr_base in {"lattice_namelist"}:
        assert main.lattice == namelist
        assert main.lattices == [namelist]
        return
    elif attr_base == "profile_gauss":
        assert getattr(main, "profile_gausses") == [namelist]
    else:
        assert getattr(main, attr) == [namelist]

    assert getattr(main, attr_base) == namelist


def test_main_input_helpers_plural(namelist: AnyNameList):
    main = MainInput(namelists=[namelist, namelist])

    attr = pydantic.alias_generators.to_snake(namelist.__class__.__name__)

    if attr == "lattice_namelist":
        attr = "lattice"

    if attr in {"setup", "initial_particles"}:
        ...
    else:
        with pytest.raises(ValueError) as ex:
            getattr(main, attr)
            assert "Please use" in str(ex.exconly())

        suggested_attr = str(ex.exconly()).split("Please use .")[1].split()[0]
        assert getattr(main, suggested_attr) == [namelist, namelist]


def test_lattice_helpers(element: AnyBeamlineElement):
    main = Lattice(elements={"a": element})

    attr_base = pydantic.alias_generators.to_snake(element.__class__.__name__)
    attr = f"{attr_base}s" if not attr_base.endswith("s") else attr_base

    assert getattr(main, attr) == [element]


def test_main_input_positional_instantiation():
    base = MainInput(namelists=[Setup()])
    positional = MainInput(namelists=[Setup()])
    assert positional == base


def test_lattice_positional_instantiation():
    base = Lattice(elements={"a": g4.Quadrupole()})
    positional = Lattice({"a": g4.Quadrupole()})
    assert positional == base

    base.fix_labels()
    list_positional = Lattice([g4.Quadrupole(label="a")])
    assert list_positional == base


def test_lattice_empty_label():
    with pytest.raises(ValueError):
        Lattice(
            [
                g4.Quadrupole(label=""),
            ]
        )


def test_lattice_duplicate_label():
    with pytest.raises(ValueError):
        Lattice(
            [
                g4.Quadrupole(label="a"),
                g4.Quadrupole(label="a"),
            ]
        )


@pytest.mark.xfail(reason="pydantic union discriminator + custom __init__", strict=True)
def test_line_positional_instantiation():
    base = g4.Line(elements=["a"])
    # We want to support this eventually, but it appears that after adding a
    # custom __init__ to Line, this causes the type-based discriminator to get
    # confused and start trying to instantiate other beamline elements using
    # the class.
    positional = g4.Line(["a"])
    assert positional == base


def test_setup_missing_setup(tmp_path: pathlib.Path):
    main = g4.MainInput()
    with pytest.raises(NoSetupNamelistError):
        main.write_files(tmp_path)


def test_setup_rootname_fix(tmp_path: pathlib.Path):
    main = g4.MainInput([g4.Setup()])
    assert main.setup.rootname == ""

    main.write_files(tmp_path)
    assert main.setup.rootname == "output"


def test_lattice_list_setattr():
    lat = Lattice(
        [
            g4.Quadrupole(L=0.0, label="a"),
            g4.Quadrupole(L=0.0, label="b"),
        ]
    )

    assert lat.quadrupoles.label == ["a", "b"]
    lat.quadrupoles.label = ["c", "d"]
    assert lat.quadrupoles.label == ["c", "d"]
    assert lat.quadrupoles[0].label == "c"
    assert lat.quadrupoles[1].label == "d"

    assert lat.quadrupoles.L == [0.0, 0.0]

    # broadcasting a single value to all
    lat.quadrupoles.L = 1.0
    assert lat.quadrupoles.L == [1.0, 1.0]
    assert lat.quadrupoles[0].L == 1.0
    assert lat.quadrupoles[1].L == 1.0

    # setting a list of values
    lat.quadrupoles.L = [3.0, 4.0]
    assert lat.quadrupoles.L == [3.0, 4.0]
    assert lat.quadrupoles[0].L == 3.0
    assert lat.quadrupoles[1].L == 4.0
