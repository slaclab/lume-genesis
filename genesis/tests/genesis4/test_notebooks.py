"""
This file contains conversions of example notebooks to regular Python.

The goal here is to run those example notebooks relatively quickly so that
the majority of their functionality can be covered in the test suite.

Running the notebooks themselves (as in `jupyter execute`) will be performed
as part of the documentation generation process.
"""

import pathlib
import pprint
import string
import textwrap
from math import pi, sqrt
from typing import Optional

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pmd_beamphysics import ParticleGroup

from ... import version4 as g4
from ...version4 import Genesis4, Lattice, MainInput, Track, Write
from ...version4.output import FieldFile
from ...version4.types import Reference
from ..conftest import (
    genesis4_examples,
    genesis4_example1_path,
    genesis4_example2_path,
)


@pytest.fixture(scope="function")
def _close_plots(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    try:
        yield
    finally:
        print("Closing plots...")
        plt.close()


@pytest.fixture(scope="function")
def _shorten_zstop(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def short_run(self, *args, **kwargs):
        for track in self.input.main.by_namelist[Track]:
            print(f"Reducing simulation range from {track.zstop}")
            track.zstop *= 0.05
            track.zstop = min((1, track.zstop))
            print(f"... now {track.zstop}")
        res = orig_run(self, *args, **kwargs)
        return res

    orig_run = Genesis4.run
    monkeypatch.setattr(Genesis4, "run", short_run)


def test_example1(_shorten_zstop) -> None:
    G = Genesis4(genesis4_example1_path / "Example1.in")
    G.verbose = True
    output = G.run(raise_on_error=True)
    G.plot(["beam_xsize", "beam_ysize", "field_xsize", "field_ysize"])

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel(r"$z$ (m)")
    ax1.set_ylabel(r"$a_w$", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.step(output.lattice.z, output.lattice.aw, color=color, where="post")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel(r"$k_1$ (m$^{-2}$)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.step(output.lattice.z, output.lattice.qf, color=color, where="post")
    plt.show()

    # plot the beam sizes
    zplot = output.lattice.zplot
    plt.plot(zplot, output.beam.xsize * 1e6, label=r"Beam: $\sigma_x$")
    plt.plot(zplot, output.beam.ysize * 1e6, label=r"Beam: $\sigma_y$")
    plt.plot(zplot, output.field.xsize * 1e6, label=r"Field: $\sigma_x$")
    plt.plot(zplot, output.field.ysize * 1e6, label=r"Field: $\sigma_y$")
    plt.legend()
    plt.xlabel(r"$z$ (m)")
    plt.ylabel(r"$\sigma_{x,y}$ ($\mu$m)")
    plt.ylim([0, 60])
    plt.show()

    # plot power and bunching
    z = output.lattice.zplot
    b = output.beam.bunching
    p = output.field.power

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel(r"$z$ (m)")
    ax1.set_ylabel(r"$P$ (W)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.semilogy(z, p, color=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel(r"$<\exp(i\theta)>$", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.semilogy(z, b, color=color)
    ax2.set_ylim([1e-3, 0.5])
    plt.show()


def test_example2() -> None:
    G = Genesis4(genesis4_example2_path / "Example2.in")
    output = G.run(raise_on_error=True)
    G.plot(["beam_xsize", "beam_ysize", "field_xsize", "field_ysize"])

    print("Loaded fields:", output.load_fields())
    print("Loaded particles:", output.load_particles())

    def get_slice(field: FieldFile, slc: int) -> np.ndarray:
        return field.dfl[:, :, slc]

    def getWF(field: FieldFile, slice=0):
        ng = field.param.gridpoints
        dg = field.param.gridsize
        # inten = np.reshape(fre * fre + fim * fim, (ng, ng))
        inten = np.abs(get_slice(field, slice) ** 2)
        return inten, dg * (ng - 1) * 0.5 * 1e3

    def getPS(particles: ParticleGroup):
        gamma = particles.gamma * 0.511e-3
        theta = np.mod(particles.theta - np.pi * 0.5, 2 * np.pi)
        return theta, gamma

    def getTS(particles: ParticleGroup):
        x = particles.x * 1e6
        theta = particles.theta
        theta = np.mod(theta - np.pi * 0.5, 2 * np.pi)
        return theta, x

    # plot wavefront
    istep = 184
    _, axs = plt.subplots(2, 2)
    color = "yellow"
    for i1 in range(2):
        for i2 in range(2):
            i = (i2 * 2 + i1 + 1) * istep
            inten, dg = getWF(output.field3d[i], slice=0)
            axs[i2, i1].imshow(inten, extent=(-dg, dg, -dg, dg))
            txt = r"$z$ = %3.1f m" % (9.5 * (i2 * 2 + i1 + 1))
            axs[i2, i1].text(-0.15, 0.15, txt, color=color)

    axs[1, 0].set_xlabel(r"$x$ (mm)")
    axs[1, 1].set_xlabel(r"$x$ (mm)")
    axs[0, 0].set_ylabel(r"$y$ (mm)")
    axs[1, 0].set_ylabel(r"$y$ (mm)")
    plt.show()

    # get range for phase space plots
    emin = np.min(output.beam.emin) * 0.511e-3
    emax = np.max(output.beam.emax) * 0.511e-3
    xmin = np.min(output.beam.xmin) * 1e6
    xmax = np.max(output.beam.xmax) * 1e6

    # plot final phase space
    t, g = getPS(output.particles[700])
    plt.scatter(t, g, s=0.2)
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$E$ (GeV)")
    plt.xlim([0, 2 * np.pi])
    plt.ylim([emin, emax])
    plt.show()

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2 * np.pi), ylim=(emin, emax))
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$E$ (GeV)")
    scat = ax.scatter([], [], s=0.2)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def animate(i):
        t, g = getPS(output.particles[2 * i])
        scat.set_offsets(np.hstack((t[:, np.newaxis], g[:, np.newaxis])))
        return (scat,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, blit=False, interval=20, frames=500
    )
    anim.save("Animation1.mp4")

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2 * np.pi), ylim=(xmin, xmax))
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$x$ ($\mu$m)")
    scat = ax.scatter([], [], s=0.2)

    def animate2(i):
        t, g = getTS(output.particles[2 * i])
        scat.set_offsets(np.hstack((t[:, np.newaxis], g[:, np.newaxis])))
        return (scat,)

    anim = animation.FuncAnimation(
        fig, animate2, init_func=init, blit=False, interval=20, frames=500
    )
    anim.save("Animation2.mp4")


def test_fodo(_shorten_zstop, tmp_path: pathlib.Path) -> None:
    """fodo_scan with *reduced* zstop and scan parameters."""
    NUM_STEPS = 2

    LATFILE = str(tmp_path / "genesis4_fodo.lat")

    def make_lat(k1=2, latfile=LATFILE):
        lat = textwrap.dedent(
            f"""\
            D1: DRIFT = {{ l = 0.445}};
            D2: DRIFT = {{ l = 0.24}};
            QF: QUADRUPOLE = {{ l = 0.080000, k1= {k1} }};
            QD: QUADRUPOLE = {{ l = 0.080000, k1= {-k1} }};

            UND: UNDULATOR = {{ lambdau=0.015000,nwig=266,aw=0.84853}};

            FODO: LINE= {{UND,D1,QF,D2,UND,D1,QD,D2}};

            ARAMIS: LINE= {{13*FODO}};
            """
        )
        with open(latfile, "w") as f:
            f.write(lat)

        return latfile

    make_lat()

    INPUT0 = [
        {
            "type": "setup",
            "rootname": "Benchmark",
            "lattice": make_lat(k1=2),
            "beamline": "ARAMIS",
            "lambda0": 1e-10,
            "gamma0": 11357.82,
            "delz": 0.045,
            "shotnoise": 0,
            "beam_global_stat": True,
            "field_global_stat": True,
        },
        {
            "type": "lattice",
            "zmatch": 9.5,
        },
        {
            "type": "field",
            "power": 5000,
            "dgrid": 0.0002,
            "ngrid": 255,
            "waist_size": 3e-05,
        },
        {"type": "beam", "current": 3000, "delgam": 1, "ex": 4e-07, "ey": 4e-07},
        {"type": "track", "zstop": 123.5},
    ]

    main = MainInput.from_dicts(INPUT0)

    G = Genesis4(main)
    G.nproc = 0  # Auto-select
    G.run()
    G.plot(
        "field_peak_power",
        yscale="log",
        y2=["beam_xsize", "beam_ysize"],
        ylim2=(0, 50e-6),
    )

    def run1(k):
        make_lat(k)
        main.setup.lattice = LATFILE
        main.by_namelist[Track][0].zstop = 20
        G = Genesis4(main)
        G.nproc = 0
        G.run()
        return G

    G2 = run1(4)
    G2.plot(
        "field_peak_power",
        yscale="log",
        y2=["beam_xsize", "beam_ysize"],
        ylim2=(0, 50e-6),
    )

    klist = np.linspace(1, 3, NUM_STEPS)
    Glist = [run1(k) for k in klist]

    fig, ax = plt.subplots()
    for k, g in zip(klist, Glist):
        x = g.stat("zplot")
        y = g.stat("field_peak_power")
        ax.plot(x, y / 1e6, label=f"{k:0.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel("power (MW)")
    plt.legend(title=r"$k$ (1/m$^2$)")

    fig, ax = plt.subplots()
    y = np.array([g.stat("field_peak_power")[-1] for g in Glist])
    ixbest = y.argmax()
    _Gbest = Glist[ixbest]
    kbest = klist[ixbest]
    ybest = y[ixbest]
    ax.plot(klist, y / 1e6)
    ax.scatter(kbest, ybest / 1e6, marker="*", label=rf"$k$= {kbest:0.1f} 1/m$^2$")
    ax.set_ylabel("end power (MW)")
    ax.set_xlabel(r"$k$ (1/m$^2$)")
    plt.legend()

    fig, ax = plt.subplots()
    for k, g in zip(klist, Glist):
        x = g.stat("zplot")
        y = g.stat("beam_xsize")
        if k == kbest:
            color = "black"
        else:
            color = None
        ax.plot(x, y * 1e6, label=f"{k:0.1f}", color=color)

    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel("Beam xsize (µm)")
    # ax.set_ylim(0, None)
    plt.legend(title=r"$k$ (1/m$^2$)")


def test_fodo_scan_model(_shorten_zstop, tmp_path: pathlib.Path) -> None:
    NUM_STEPS = 2

    import string
    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import numpy as np

    def make_fodo(
        Lcell=9.5,
        kL=None,
        lambdau=15e-3,
        Ltot=150.0,
        Lpad=0.685 / 2,  # Will be adjusted slightly to make Lcell exact
        Lquad=0.08,
        aw=0.84853,
    ):
        if kL is None:
            #  Optimal for flat beam
            kL = 1 / (sqrt(2) * Lcell / 2)

            # Optimal for round beam (90 deg phase advance)
            # k1L_optimal = 2*sqrt(2) / Lcell

        k1 = kL / Lquad

        # Length for single wiggler
        Lwig = (Lcell - 4 * Lpad - 2 * Lquad) / 2
        nwig = round(Lwig / lambdau)

        # Set padding exactly
        Lwig = lambdau * nwig
        Lpad = round((Lcell - 2 * Lwig - 2 * Lquad) / 4, 9)

        ncell = round(Ltot // Lcell)

        lat = string.Template(
            """
            D1: DRIFT = { l = ${Lpad} };
            D2: DRIFT = { l = ${Lpad} };
            QF: QUADRUPOLE = { l = ${Lquad}, k1=  ${k1} };
            QD: QUADRUPOLE = { l = ${Lquad}, k1= -${k1} };
            UND: UNDULATOR = { lambdau=${lambdau}, nwig=${nwig}, aw=${aw} };
            FODO: LINE= {UND, D1,QF,D2, UND, D1,QD,D2};
            LAT: LINE= { ${ncell}*FODO };
            """
        ).substitute(
            Lpad=Lpad,
            Lquad=Lquad,
            k1=k1,
            lambdau=lambdau,
            nwig=nwig,
            aw=aw,
            ncell=ncell,
            kL=kL,
        )
        return lat

    print(make_fodo(lambdau=30e-3, Lcell=9.5))

    def Krof(*, lambdar, lambdau, gamma):
        """
        K to make lambdar resonant
        """
        Ksq = 2 * (2 * gamma**2 * lambdar / lambdau - 1)
        if Ksq <= 0:
            raise ValueError(
                f"No resonance available, lambdau must be < {2*gamma**2*lambdar*1e3:0.1f}1e-3 m"
            )

        return sqrt(Ksq)

    print(Krof(lambdar=1e-10, lambdau=25e-3, gamma=11357.82) / sqrt(2))

    @dataclass
    class FODOModel:
        # Lengths
        Lcell: float = 9.5
        Ltot: float = 125
        Lquad: float = 0.08
        Lpad: float = 0.685 / 2

        kL: Optional[float] = None  # Will be picked automatically

        lambdar: float = 1e-10
        lambdau: float = 15e-3
        gamma: float = 11357.82

        current: float = 3000
        delgam: float = 1.0
        norm_emit_x: float = 0.4e-6
        norm_emit_y: float = 0.4e-6

        nproc = 0  # Auto-select

        seed: Optional[int] = None  # None will pick a random seed

        def make_lattice(self):
            """
            Returns the lattice string,
            setting aw for resonance
            """

            aw = Krof(
                lambdar=self.lambdar, lambdau=self.lambdau, gamma=self.gamma
            ) / sqrt(2)

            return make_fodo(
                Lcell=self.Lcell,
                kL=self.kL,
                lambdau=self.lambdau,
                Ltot=self.Ltot,
                Lpad=self.Lpad,  # Will be adjusted
                Lquad=self.Lquad,
                aw=aw,
            )

        def make_genesis(self):
            if self.seed is None:
                seed = np.random.randint(0, 1e10)
            else:
                seed = self.seed

            input = MainInput(
                namelists=[
                    g4.Setup(
                        rootname="Benchmark",
                        beamline="LAT",
                        lambda0=self.lambdar,
                        gamma0=self.gamma,
                        delz=0.045,
                        seed=seed,
                        shotnoise=False,
                        beam_global_stat=True,
                        field_global_stat=True,
                    ),
                    g4.LatticeNamelist(zmatch=self.Lcell),
                    g4.Field(
                        power=5000,
                        dgrid=0.0002,
                        ngrid=255,
                        waist_size=3e-05,
                    ),
                    g4.Beam(
                        current=self.current,
                        delgam=self.delgam,
                        ex=self.norm_emit_x,
                        ey=self.norm_emit_y,
                    ),
                    g4.Track(zstop=self.Ltot),
                ]
            )
            lattice = self.make_lattice()
            G = Genesis4(input, lattice, verbose=True)
            return G

        def run(self):
            G = self.make_genesis()
            G.nproc = self.nproc
            G.run()
            return G

    fodo_model = FODOModel(Ltot=20)

    G = fodo_model.run()

    G.plot(
        "field_peak_power",
        yscale="log",
        y2=["beam_xsize", "beam_ysize"],
        ylim2=(0, 50e-6),
    )

    def run2(kL):
        G = FODOModel(kL=kL).run()
        return G

    G2 = run2(0.136)  # = 1.7 * .08

    G2.plot(
        "field_peak_power",
        yscale="log",
        y2=["beam_xsize", "beam_ysize"],
        ylim2=(0, 200e-6),
    )

    kLlist = np.linspace(0.1, 0.4, NUM_STEPS)
    Glist = [run2(kL) for kL in kLlist]

    fig, ax = plt.subplots()
    for k, g in zip(kLlist, Glist):
        x = g.stat("zplot")
        y = g.stat("field_peak_power")
        ax.plot(x, y / 1e6, label=f"{k:0.3f}")
    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel("power (MW)")
    plt.legend(title=r"$k_1L$ (1/m)")

    fig, ax = plt.subplots()
    y = np.array([g.stat("field_peak_power")[-1] for g in Glist])
    ixbest = y.argmax()
    _Gbest = Glist[ixbest]
    kbest = kLlist[ixbest]
    ybest = y[ixbest]
    ax.plot(kLlist, y / 1e6)
    ax.scatter(kbest, ybest / 1e6, marker="*", label=rf"$k_1L$= {kbest:0.1f} 1/m")
    ax.set_ylabel("end power (MW)")
    ax.set_xlabel(r"$k_1L$ (1/m)")
    plt.legend()

    fig, ax = plt.subplots()
    for k, g in zip(kLlist, Glist):
        x = g.stat("zplot")
        y = g.stat("beam_xsize")
        if k == kbest:
            color = "black"
        else:
            color = None
        ax.plot(x, y * 1e6, label=f"{k:0.3f}", color=color)
    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel("Beam xsize (µm)")
    ax.set_ylim(0, None)
    plt.legend(title=r"$k_1L$ (1/m)")

    def run3(lambdau):
        p = {}
        p["lambdau"] = lambdau
        G = FODOModel(**p).make_genesis()
        G.run()
        return G

    G2 = run3(10e-3)
    G2.plot(
        "field_peak_power",
        yscale="log",
        y2=["beam_xsize", "beam_ysize"],
        ylim2=(0, None),
    )

    lambdaulist = np.linspace(10e-3, 25e-3, NUM_STEPS)
    Glist = [run3(lambdau) for lambdau in lambdaulist]

    fig, ax = plt.subplots()
    for k, g in zip(lambdaulist, Glist):
        x = g.stat("zplot")
        y = g.stat("field_peak_power")
        ax.plot(x, y / 1e6, label=f"{k*1e3:0.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel("power (MW)")
    plt.legend(title=r"$\lambda_u$ (mm)")


@pytest.mark.filterwarnings("ignore:Attempt to set non-positive")
@pytest.mark.filterwarnings("ignore:More than 20")
def test_genesis4_example(_shorten_zstop, tmp_path: pathlib.Path) -> None:
    G = Genesis4(
        genesis4_examples / "data/basic4/cu_hxr.in",
        genesis4_examples / "data/basic4/hxr.lat",
        verbose=True,
    )
    G.input.main.by_namelist[Track]
    for track in G.input.main.by_namelist[Track]:
        track.zstop = 92
    G.input.main.by_namelist[Track]
    G.input.main.by_namelist[Track][0]
    # Add writing a field file
    G.input.main.namelists.append(Write(field="end"))
    # Add writing a beam (particle) file
    G.input.main.namelists.append(Write(beam="end"))
    # This is tested elsewhere
    # genesis.global_display_options.include_description = False
    #
    # print("Viewed as an HTML table:")
    #
    # for write in G.input.main.by_namelist[Write]:
    #     IPython.display.display(write)
    print("Or viewed as a markdown table:")
    for write in G.input.main.by_namelist[Write]:
        print(write)
        print()
    G.nproc = 0
    G.run()
    assert G.output
    if G.output.run.error:
        print(G.output.run.error_reason)
    else:
        print("No error")
    # The main output is an HDF5. The Genesis4 object loads all array data into a flat dict
    list(G.output)
    # This is the output file that was loaded
    print("Took", G.output.run.run_time, "sec")
    print(list(G.output.keys())[:10], "...")
    print(G.output.alias["alphax"])
    # There are many outputs. `.output_info()` gives a convenient table describing what was read in.
    G.output.info()
    # Field files can be very large and are made readily available for lazy loading.
    # Loaded fields are present in `.field` in the output:
    list(G.output.field3d)
    # For convenience, fields and particles may be automatically loaded after a run by using `run(load_fields=True, load_particles=True)` instead.
    # Otherwise, these can be manually loaded individually or all at once:
    G.output.load_fields()
    list(G.output.field3d)
    # This field data has two parts: basic parameters `param`, and the raw 3D complex array `dfl`:
    print(G.output.field3d["end"].param)
    print(G.output.field3d["end"].dfl.shape)
    # Sum over y and compute the absolute square
    dfl = G.output.field3d["end"].dfl
    param = G.output.field3d["end"].param
    dat2 = np.abs(np.sum(dfl, axis=1)) ** 2
    plt.imshow(dat2)

    def plot_slice(i=0):
        dat = np.angle(dfl[:, :, i])
        dx = param.gridsize * 1e6
        plt.xlabel("x (µm)")
        plt.xlabel("y (µm)")
        plt.title(f"Phase for slize {i}")
        plt.imshow(dat.T, origin="lower", extent=[-dx, dx, -dx, dx])

    plot_slice(i=100)
    # # Particles
    #
    # Particle files can be read in as [openPMD-beamphysics](https://christophermayes.github.io/openPMD-beamphysics/) `ParticleGroup` objects.
    # These are lazily loaded by default (`run(load_particles=False)`). They may also be loaded all at once, with `load_particles()`. `output.particle_files` will only show not-yet-loaded particle files.
    G.output.particle_files
    G.output.load_particles()
    G.output.particles
    P = G.output.particles["end"]
    P.plot("z", "energy")
    # Change to z coordinates to see the current. Note that the head of the bunch is now on the left.
    P.drift_to_z()
    P.plot("t", "energy")
    # Check some statistics
    print(P["norm_emit_x"], P["norm_emit_y"], P["mean_gamma"])
    wavelength = G.input.main.setup.lambda0
    bunching_key = f"bunching_{wavelength}"
    P.drift_to_t()
    P.slice_plot(bunching_key, n_slice=1000)
    # Genesis4 data
    final_bunching = G.output.beam.bunching[-1, :]
    _current = G.output.beam.current[-1, :]
    s = G.output.globals.s
    # ParticleGroup data
    ss = P.slice_statistics(bunching_key, n_slice=len(s))
    ss.keys()
    x = ss["mean_z"]
    y = ss[bunching_key]
    # Compare
    fig, ax = plt.subplots()
    ax.plot(x * 1e6, y, label="ParticleGroup")
    ax.plot(s * 1e6, final_bunching, "--", label="Genesis4 output")
    ax.set_xlabel("s (µm)")
    ax.set_ylabel("bunching")
    plt.legend()
    # This is the average bunching from the ParticleGroup:
    P.bunching(wavelength)
    # That agrees with the appropriate averaging of Genesis4's bunching calc:
    G.stat("bunching")[-1]
    G.plot("bunching")
    # Check the total charge in pC:
    print(P["charge"] / 1e-12)  # pC
    # Each item in the output dict has a corresponding units
    G.output.units("beam_betax")
    # # Plotting
    #
    # Convenient plotting of the data in `.output` is provided by `.plot`. The default is to plot the power. Depending on the key these statistics are averaged or integrated over the slices. Some keys like `power` are converted to `peak_power`, while `field_energy` is the integral over `field_power`.
    print(G.output.alias["field_peak_power"])
    G.plot()
    # Left and right axes can be set this way:
    G.plot(
        "field_energy", yscale="log", y2=["beam_xsize", "beam_ysize"], ylim2=(0, 100e-6)
    )
    # By default, these plots average over slices. In the case of beam sizes, simply averaging these does not take into account the effect of misaligned slices. To plot this, LUME-Genesis provides additional `beam_sigma_x`, `beam_sima_y`, `beam_sigma_energy` keys that properly project these quantities. The difference is noticable in the energy spread calculation:
    G.plot(["beam_sigma_energy", "beam_energyspread"], ylim=(0, 100))
    G.plot(["field_xsize", "field_ysize"])
    plt.imshow(G.output.field.power, aspect="auto")
    G.archive(tmp_path / "archived.h5")
    fp = h5py.File(tmp_path / "archived.h5")
    # Grestored = Genesis4.from_archive("archived.h5")
    Grestored = Genesis4.from_archive(fp)
    assert Grestored.output is not None
    Grestored.output.plot()
    # # Manual loading of Genesis4 data
    #
    # Sometimes it is necessary to run Genesis4 manually, and load the output into LUME-Genesis for further analysis.
    #
    # First, let's create some input to run in a local directory `temp/`:
    new_path = tmp_path / "manual-loading-test"
    G.write_input(new_path)
    # Now run on the command line:
    # Using the `use_temp_dir=False` and `workdir` options, the input and output data can be loaded into a new Genesis4 object:
    G2 = Genesis4("genesis4.in", use_temp_dir=False, workdir=new_path, verbose=True)
    G2.run()
    G2.plot()


def test_parsing(_shorten_zstop):
    FILE = genesis4_examples / "data/basic4/cu_hxr.in"
    input = MainInput.from_file(FILE)
    input.namelists
    print(input.setup)
    input = MainInput.from_contents(
        """
    &setup
      rootname = LCLS2_HXR_9keV
      lattice = hxr.lat
      beamline = HXR
      gamma0 = 19174.0776
      lambda0 = 1.3789244869952112e-10
      delz = 0.026
      seed = 84672
      npart = 1024
    &end
    """
    )

    FILE = genesis4_examples / "data/basic4/hxr.lat"
    lat = Lattice.from_file(FILE)
    lat.elements
    print(lat.to_genesis())
    lat = Lattice.from_contents(
        """
    CORR32: corrector = {l=0.001};
    CORR33: corrector = {l=0.001};
    """
    )


def test_genesis4_particles(_shorten_zstop, tmp_path: pathlib.Path):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.constants import c

    from ...version4 import (
        Beam,
        Drift,
        Lattice,
        Line,
        MainInput,
        ProfileArray,
        ProfileGauss,
        Setup,
        Time,
        Track,
        Write,
    )

    D1 = Drift(L=1)
    lattice = Lattice(elements={"D1": D1, "LAT": Line(elements=[D1])})
    PEAK_CURRENT = 1000
    BUNCH_CHARGE = 100e-12
    SIGMA_T = BUNCH_CHARGE / (sqrt(2 * pi) * PEAK_CURRENT)
    SIGMA_Z = SIGMA_T * c
    SLEN = 6 * SIGMA_Z
    S0 = 3 * SIGMA_Z
    print(SIGMA_T, SIGMA_Z, SLEN)
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
            Time(slen=SLEN),
            ProfileGauss(
                label="beamcurrent",
                c0=PEAK_CURRENT,
                s0=S0,
                sig=SIGMA_Z,
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
    G = Genesis4(main, lattice, verbose=True)
    output = G.run()
    G.input.main.setup.delz
    print(G.output.run.output_log)
    output.load_particles()
    P1 = output.particles["end"]
    P1.drift_to_z()
    P1.plot("t", "energy")
    P1
    output.particles["end"]
    P1.charge
    NPTS = 100
    SLEN = 100e-6
    S = np.linspace(0, SLEN, NPTS)
    CURRENT = np.linspace(1, 1000.0, NPTS)
    plt.plot(S, CURRENT)
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
            Time(slen=SLEN),
            ProfileArray(label="beamcurrent", xdata=S, ydata=CURRENT),
            Beam(
                gamma=1000,
                delgam=1,
                current="beamcurrent",
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
    G = Genesis4(main, lattice, verbose=True)
    output = G.run()
    print(main.to_genesis())
    print(lattice.to_genesis())
    print(output.run.output_log)
    output.meta
    output.load_particles()
    P1 = output.particles["end"]
    P1.drift_to_z()
    P1.plot("t", "energy")
    print(P1)
    NSAMPLE = len(P1)
    P1r = P1.resample(NSAMPLE)
    P1r.plot("t", "energy")
    print(P1r)
    P1r.pz[0 : len(P1) // 2] *= 1.1
    P1r.plot("t", "energy")

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
    G1 = Genesis4(main, lattice, verbose=True, initial_particles=P1r)
    assert G1.input.main.import_distribution.file

    output = G1.run()
    pprint.pprint(output.run)
    print(output.run.output_log)
    output.load_particles()
    P2 = output.particles["end"]
    P2.z
    P2.drift_to_z()
    P2.plot("t", "energy")
    print(P2)
    P2.plot("weight", bins=100)
    list(output.beam)
    G1.input

    G1.archive(tmp_path / "archive.h5")
    loaded = Genesis4.from_archive(tmp_path / "archive.h5")
    assert loaded.initial_particles is not None
    assert np.isclose(loaded.initial_particles.charge, P1r.charge)


def test_example1_lattice_plot() -> None:
    G = Genesis4(genesis4_example1_path / "Example1.in")
    with pytest.raises(ValueError):
        G.input.lattice.plot()  # need to specify beamline name
    with pytest.raises(ValueError):
        G.input.lattice.plot("invalid_beamline_name")
    with pytest.raises(ValueError):
        G.input.lattice.plot("QF")  # quadrupole, not line
    G.input.lattice.plot("FODO")
    G.input.lattice.plot("FEL")

    G.input.lattice.elements.pop("FEL")
    G.input.lattice.plot()  # only a single beamline now, plot it


def test_migration():
    MAIN = [
        {
            "type": "setup",
            "rootname": "Benchmark",
            "lattice": "lattice.lat",
            "beamline": "ARAMIS",
            "lambda0": 1e-10,
            "gamma0": 11357.82,
            "delz": 0.045,
            "shotnoise": 0,
            "beam_global_stat": True,
            "field_global_stat": True,
        },
        {"type": "lattice", "zmatch": 9.5},
        {
            "type": "field",
            "power": 5000,
            "dgrid": 0.0002,
            "ngrid": 255,
            "waist_size": 3e-05,
        },
        {"type": "beam", "current": 3000, "delgam": 1, "ex": 4e-07, "ey": 4e-07},
        {"type": "track", "zstop": 123.5},
    ]

    main = g4.MainInput.from_dicts(MAIN)

    print(main)
    assert main.setup.beamline == "ARAMIS"
    assert np.isclose(main.lattice.zmatch, 9.5)
    assert main.field.power == 5000
    assert main.beam.current == 3000
    assert np.isclose(main.track.zstop, 123.5)
    assert len(main.namelists) == len(MAIN)

    with pytest.raises(ValueError):
        g4.MainInput.from_dicts([{}])

    with pytest.raises(ValueError):
        g4.MainInput.from_dicts([{"type": "invalid_type", "zstop": 123.5}])

    with pytest.raises(ValueError):
        g4.MainInput.from_dicts([{"type": "beam", "INVALID_ITEM": 123.5}])

    def make_lat_orig(k1=2):
        return string.Template(
            """
    D1: DRIFT = { l = 0.445};
    D2: DRIFT = { l = 0.24};
    QF: QUADRUPOLE = { l = 0.080000, k1= ${my_k1} };
    QD: QUADRUPOLE = { l = 0.080000, k1= -${my_k1} };

    UND: UNDULATOR = { lambdau=0.015000,nwig=266,aw=0.84853};

    FODO: LINE= {UND,D1,QF,D2,UND,D1,QD,D2};

    ARAMIS: LINE= {13*FODO};
        """
        ).substitute(my_k1=k1)

    orig_lattice = g4.Lattice.from_contents(make_lat_orig())

    def make_lat_new(k1=2):
        return g4.Lattice(
            {
                "D1": g4.Drift(L=0.445),
                "D2": g4.Drift(L=0.24),
                "QF": g4.Quadrupole(L=0.08, k1=k1),
                "QD": g4.Quadrupole(L=0.08, k1=-k1),
                "UND": g4.Undulator(aw=0.84853, lambdau=0.015, nwig=266),
                "FODO": g4.Line(
                    elements=["UND", "D1", "QF", "D2", "UND", "D1", "QD", "D2"]
                ),
                "ARAMIS": g4.Line(
                    elements=[g4.DuplicatedLineItem(label="FODO", count=13)]
                ),
            },
        )

    new_lattice = make_lat_new()

    new_lattice.filename = orig_lattice.filename
    assert orig_lattice == new_lattice

    G = g4.Genesis4(main, make_lat_new())

    print(G.input)

    new_main = g4.MainInput.from_dicts(MAIN)

    G.input.main = new_main

    _G1 = g4.Genesis4(main, make_lat_new())
