import pydantic
import pydantic.alias_generators

from ...version4 import MainInput
from ...version4.input import Lattice, Setup
from ...version4.input.core import AnyBeamlineElement, AnyNameList


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

    if attr_base == "setup":
        assert getattr(main, attr_base) == namelist
    else:
        assert getattr(main, attr) == [namelist]


def test_lattice_helpers(element: AnyBeamlineElement):
    main = Lattice(elements={"a": element})

    attr_base = pydantic.alias_generators.to_snake(element.__class__.__name__)
    attr = f"{attr_base}s" if not attr_base.endswith("s") else attr_base

    assert getattr(main, attr) == [element]
