from typing import Union
import pytest

from ...version4.types import BeamlineElement, NameList

from ...tools import DisplayOptions

from ...version4.input import (
    Chicane,
    Corrector,
    Drift,
    Marker,
    PhaseShifter,
    Quadrupole,
    Undulator,
    AlterSetup,
    Beam,
    Efield,
    Field,
    ImportBeam,
    ImportDistribution,
    ImportField,
    ImportTransformation,
    ProfileConst,
    ProfileFile,
    ProfileGauss,
    ProfilePolynom,
    ProfileStep,
    SequenceConst,
    SequencePolynom,
    SequencePower,
    SequenceRandom,
    Setup,
    Sponrad,
    Time,
    Track,
    Wake,
    Write,
    InitialParticlesFile,
    Lattice,
    Line,
    ProfileArray,
)


@pytest.fixture(
    params=[
        ("html", True),
        ("markdown", True),
        ("genesis", True),
        ("html", False),
        ("markdown", False),
        ("genesis", False),
    ],
    ids=[
        "html-with-desc",
        "markdown-with-desc",
        "genesis-with-desc",
        "html-no-desc",
        "markdown-no-desc",
        "genesis-no-desc",
    ],
)
def display_options(request: pytest.FixtureRequest) -> DisplayOptions:
    mode, desc = request.param
    return DisplayOptions(
        jupyter_render_mode=mode,
        console_render_mode=mode if mode != "html" else "genesis",
        include_description=desc,
    )


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(Chicane(), id="Chicane"),
        pytest.param(Corrector(), id="Corrector"),
        pytest.param(Drift(), id="Drift"),
        pytest.param(Marker(), id="Marker"),
        pytest.param(PhaseShifter(), id="PhaseShifter"),
        pytest.param(Quadrupole(), id="Quadrupole"),
        pytest.param(Undulator(), id="Undulator"),
        pytest.param(AlterSetup(), id="AlterSetup"),
        pytest.param(Beam(), id="Beam"),
        pytest.param(Efield(), id="Efield"),
        pytest.param(Field(), id="Field"),
        pytest.param(ImportBeam(), id="ImportBeam"),
        pytest.param(ImportDistribution(), id="ImportDistribution"),
        pytest.param(ImportField(), id="ImportField"),
        pytest.param(ImportTransformation(), id="ImportTransformation"),
        pytest.param(ProfileConst(label="label"), id="ProfileConst"),
        pytest.param(ProfileFile(label="label"), id="ProfileFile"),
        pytest.param(ProfileGauss(label="label"), id="ProfileGauss"),
        pytest.param(ProfilePolynom(label="label"), id="ProfilePolynom"),
        pytest.param(ProfileStep(label="label"), id="ProfileStep"),
        pytest.param(SequenceConst(label="label"), id="SequenceConst"),
        pytest.param(SequencePolynom(label="label"), id="SequencePolynom"),
        pytest.param(SequencePower(label="label"), id="SequencePower"),
        pytest.param(SequenceRandom(label="label"), id="SequenceRandom"),
        pytest.param(Setup(), id="Setup"),
        pytest.param(Sponrad(), id="Sponrad"),
        pytest.param(Time(), id="Time"),
        pytest.param(Track(), id="Track"),
        pytest.param(Wake(), id="Wake"),
        pytest.param(Write(), id="Write"),
        # pytest.param(InitialParticlesData(data={}), id="InitialParticlesData"),
        pytest.param(InitialParticlesFile(), id="InitialParticlesFile"),
        pytest.param(Lattice(), id="Lattice"),
        pytest.param(Line(), id="Line"),
        pytest.param(
            ProfileArray(label="label", xdata=[0], ydata=[0]), id="ProfileArray"
        ),
    ],
)
def test_render(
    display_options: DisplayOptions, obj: Union[BeamlineElement, NameList]
) -> None:
    print("Render options", display_options)
    print("Using __str__:")
    print(str(obj))
    print("Using _repr_html_:")
    print(obj._repr_html_())
