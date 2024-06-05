import pathlib
import time
from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pmd_beamphysics import ParticleGroup
from scipy.constants import c

from ...version4 import Genesis4, Genesis4Input, Lattice, MainInput, Reference
from ...version4.input import (
    Beam,
    Drift,
    Line,
    ProfileArray,
    ProfileGauss,
    Setup,
    Time,
    Track,
    Write,
)
from ..conftest import test_root


def test_profile_gauss():
    # Create a simple drift lattice
    D1 = Drift(L=1)
    lattice = Lattice(elements={"D1": D1, "LAT": Line(elements=[D1])})

    # This profile will make a Gaussian distribition. Here we do some
    # calculations to make the correct bunch length for a given bunch charge to
    # provide a peak current.
    peak_current = 1000
    bunch_charge = 100e-12
    sigma_t = bunch_charge / (sqrt(2 * pi) * peak_current)
    sigma_z = sigma_t * c
    slen = 6 * sigma_z
    S0 = 3 * sigma_z

    main = MainInput(
        namelists=[
            Setup(
                rootname="drift_test",
                # lattice="LATFILE",
                beamline="LAT",
                gamma0=1000,
                lambda0=1e-07,
                delz=0.026,
                seed=123456,
                npart=128,
            ),
            Time(slen=slen),
            ProfileGauss(
                label="beamcurrent",
                c0=peak_current,
                s0=S0,
                sig=sigma_z,
            ),
            Beam(
                gamma=1000,
                delgam=1,
                current=Reference("beamcurrent"),
            ),
            Track(zstop=1),
            Write(beam="end"),
        ],
    )

    input = Genesis4Input(
        main=main,
        lattice=lattice,
    )
    G = Genesis4(input=input, verbose=True)
    output = G.run(raise_on_error=True)

    print("delz=", G.input.main.setup.delz)

    output.load_particles()
    P1 = output.particles["end"]
    P1.drift_to_z()
    P1.plot("t", "energy", return_figure=True)

    print("End particles:", output.particles["end"])
    assert isinstance(output.particles["end"], ParticleGroup)
    print("Charge is", P1.charge)


def test_profile_array(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
) -> None:
    # Create a simple drift lattice
    D1 = Drift(L=1)
    lattice = Lattice(elements={"D1": D1, "LAT": Line(elements=[D1])})

    # LUME-Genesis automatically makes an HDF5 file with `ProfileArray`.
    npts = 100
    slen = 100e-6
    S = np.linspace(0, slen, npts)
    current = np.linspace(1, 1000.0, npts)
    fig = plt.figure()
    plt.plot(S, current)
    fig.savefig(test_root / f"{request.node.name}_0.png")

    main = MainInput(
        namelists=[
            Setup(
                rootname="drift_test",
                # lattice=lattice,
                beamline="LAT",
                gamma0=1000,
                lambda0=1e-07,
                delz=0.026,
                seed=123456,
                npart=128,
            ),
            Time(slen=slen),
            ProfileArray(label="beamcurrent", xdata=S, ydata=current),
            Beam(
                gamma=1000,
                delgam=1,
                current=Reference("beamcurrent"),
                ex=1e-06,
                ey=1e-06,
                betax=7.910909406464387,
                betay=16.881178621346898,
                alphax=-0.7393217413918415,
                alphay=1.3870723536888105,
            ),
            Track(zstop=1),
            Write(beam="end"),
        ]
    )

    G = Genesis4(
        input=Genesis4Input(
            main=main,
            lattice=lattice,
        ),
        verbose=True,
        workdir=tmp_path,
        use_temp_dir=False,
    )
    output = G.run(raise_on_error=True)

    print("Main", main)
    print("Lattice", lattice)
    print("Output log", output.run.output_log)
    print("Meta", output.meta)

    output.load_particles()
    P1 = output.particles["end"]
    P1.drift_to_z()
    fig = P1.plot("t", "energy", return_figure=True)
    assert fig is not None
    fig.savefig(test_root / f"{request.node.name}_1.png")

    # Resample particles for equal weights. This is neccessary when reading
    # from a distribution file.
    P1r = P1.resample(len(P1))
    fig = P1r.plot("t", "energy", return_figure=True)
    assert fig is not None
    fig.savefig(test_root / f"{request.node.name}_2.png")

    # Make a more interesting distribution from this:
    P1r.pz[0 : len(P1) // 2] *= 1.1
    fig = P1r.plot("t", "energy", return_figure=True)
    assert fig is not None
    fig.savefig(test_root / f"{request.node.name}_3.png")

    # ParticleGroup can write to a file for Genesis4.
    dist_file = tmp_path / "genesis4_distribution.h5"
    P1r.write_genesis4_distribution(str(dist_file), verbose=True)

    # Use this file as the input to a new simulation.

    main = MainInput(
        namelists=[
            Setup(
                rootname="drift_test",
                # lattice=full_path(LATFILE),
                beamline="LAT",
                gamma0=1000,
                lambda0=1e-07,
                delz=0.026,
                seed=123456,
                npart=512,
            ),
            Time(),
            Track(zstop=1),
            Write(beam="end"),
        ],
    )

    input = Genesis4Input(
        main=main,
        lattice=lattice,
    )
    G1 = Genesis4(
        input=input,
        verbose=True,
        workdir=tmp_path,
        use_temp_dir=False,
        initial_particles=P1r,
    )

    assert G1.input.main.import_distribution is not None
    output = G1.run(raise_on_error=True)
    print(output.run.output_log)

    output.load_particles()
    P2 = output.particles["end"]
    print("p2.z=", P2.z)

    P2.drift_to_z()
    fig = P2.plot("t", "energy", return_figure=True)
    fig.savefig(test_root / f"{request.node.name}_4.png")
    print("P2=", P2)

    fig = P2.plot("weight", bins=100, return_figure=True)
    fig.savefig(test_root / f"{request.node.name}_5.png")
    # Notice that `importdistribution` is filled in:
    list(output.beam)
    G1.input

    t0 = time.monotonic()
    G1.archive(tmp_path / "archive.h5")
    t1 = time.monotonic()
    loaded_g1 = Genesis4.from_archive(tmp_path / "archive.h5")
    t2 = time.monotonic()
    print("Took", t1 - t0, "s to archive")
    print("Took", t2 - t1, "s to restore")

    assert loaded_g1.input.initial_particles is not None
    for key, value in loaded_g1.input.initial_particles.data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, P1r.data[key])
        else:
            assert value == P1r.data[key]
