import pathlib
import textwrap

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics import ParticleGroup

from genesis.version4.input import MainInput, Track, Write
from genesis.version4.types import FieldFile

from ...version4 import Genesis4
from ..conftest import genesis4_examples

example_data = genesis4_examples / "data"


def test_example1() -> None:
    workdir = example_data / "example1-steadystate"
    G = Genesis4(workdir / "Example1.in")
    G.verbose = True
    output = G.run(raise_on_error=True)
    G.plot(["beam_xsize", "beam_ysize", "field_xsize", "field_ysize"])

    z = output.lattice["z"]
    aw = output.lattice["aw"]
    qf = output.lattice["qf"]

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel(r"$z$ (m)")
    ax1.set_ylabel(r"$a_w$", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.step(z, aw, color=color, where="post")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel(r"$k_1$ (m$^{-2}$)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.step(z, qf, color=color, where="post")
    plt.show()

    # plot the beam sizes
    z = output.lattice["zplot"]
    bx = output.beam["xsize"]
    by = output.beam["ysize"]
    fx = output.field_info["xsize"]
    fy = output.field_info["ysize"]
    plt.plot(z, bx * 1e6, label=r"Beam: $\sigma_x$")
    plt.plot(z, by * 1e6, label=r"Beam: $\sigma_y$")
    plt.plot(z, fx * 1e6, label=r"Field: $\sigma_x$")
    plt.plot(z, fy * 1e6, label=r"Field: $\sigma_y$")
    plt.legend()
    plt.xlabel(r"$z$ (m)")
    plt.ylabel(r"$\sigma_{x,y}$ ($\mu$m)")
    plt.ylim([0, 60])
    plt.show()

    # plot power and bunching
    z = output.lattice["zplot"]
    b = output.beam["bunching"]
    p = output.field_info["power"]

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
    G = Genesis4(example_data / "example2-dumps" / "Example2.in")
    output = G.run(raise_on_error=True)
    G.plot(["beam_xsize", "beam_ysize", "field_xsize", "field_ysize"])

    print("Loaded fields:", output.load_fields())
    print("Loaded particles:", output.load_particles())

    def get_slice(field: FieldFile, slc: int) -> np.ndarray:
        return field.dfl[:, :, slc]

    def getWF(field: FieldFile, slice=0):
        ng = field.param["gridpoints"]
        dg = field.param["gridsize"]
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
            inten, dg = getWF(output.fields[f"Example2.{i}"], slice=0)
            axs[i2, i1].imshow(inten, extent=(-dg, dg, -dg, dg))
            txt = r"$z$ = %3.1f m" % (9.5 * (i2 * 2 + i1 + 1))
            axs[i2, i1].text(-0.15, 0.15, txt, color=color)

    axs[1, 0].set_xlabel(r"$x$ (mm)")
    axs[1, 1].set_xlabel(r"$x$ (mm)")
    axs[0, 0].set_ylabel(r"$y$ (mm)")
    axs[1, 0].set_ylabel(r"$y$ (mm)")
    plt.show()

    # get range for phase space plots
    emin = np.min(output.beam["emin"]) * 0.511e-3
    emax = np.max(output.beam["emax"]) * 0.511e-3
    xmin = np.min(output.beam["xmin"]) * 1e6
    xmax = np.max(output.beam["xmax"]) * 1e6

    # plot final phase space
    t, g = getPS(output.particles["Example2.700"])
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
        t, g = getPS(output.particles[f"Example2.{2 * i}"])
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
        t, g = getTS(output.particles[f"Example2.{2 * i}"])
        scat.set_offsets(np.hstack((t[:, np.newaxis], g[:, np.newaxis])))
        return (scat,)

    anim = animation.FuncAnimation(
        fig, animate2, init_func=init, blit=False, interval=20, frames=500
    )
    anim.save("Animation2.mp4")


# TODO: fodo_scan_model


def test_fodo(tmp_path: pathlib.Path) -> None:
    """fodo_scan with *reduced* zstop and scan parameters."""
    ZSTOP_SHORTEN_FACTOR = 0.1

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
        {"type": "track", "zstop": 123.5 * ZSTOP_SHORTEN_FACTOR},
    ]

    main = MainInput.from_dicts(INPUT0)

    G = Genesis4(main)
    G.nproc = 0  # Auto-select
    G.run()
    G.plot("power", yscale="log", y2=["beam_xsize", "beam_ysize"], ylim2=(0, 50e-6))

    def run1(k):
        make_lat(k)
        main.setup.lattice = LATFILE
        main.by_namelist[Track][0].zstop = 20
        G = Genesis4(main)
        G.nproc = 8
        G.run()
        return G

    G2 = run1(4)
    G2.plot("power", yscale="log", y2=["beam_xsize", "beam_ysize"], ylim2=(0, 50e-6))

    klist = np.linspace(1, 3, int(10 * ZSTOP_SHORTEN_FACTOR))
    Glist = [run1(k) for k in klist]

    fig, ax = plt.subplots()
    for k, g in zip(klist, Glist):
        x = g.stat("zplot")
        y = g.stat("power")
        ax.plot(x, y / 1e6, label=f"{k:0.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel("power (MW)")
    plt.legend(title=r"$k$ (1/m$^2$)")

    fig, ax = plt.subplots()
    y = np.array([g.stat("power")[-1] for g in Glist])
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


def test_genesis4_example(tmp_path: pathlib.Path) -> None:
    """genesis4_example with *significantly shortened* zstop."""
    ZSTOP_SHORTEN_FACTOR = 0.005

    G = Genesis4(
        genesis4_examples / "data/basic4/cu_hxr.in",
        genesis4_examples / "data/basic4/hxr.lat",
        verbose=True,
    )
    G.input.main.by_namelist[Track]
    for track in G.input.main.by_namelist[Track]:
        track.zstop = 92 * ZSTOP_SHORTEN_FACTOR
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
    list(G.output.field)
    # For convenience, fields and particles may be automatically loaded after a run by using `run(load_fields=True, load_particles=True)` instead.
    # Otherwise, these can be manually loaded individually or all at once:
    G.output.load_fields()
    list(G.output.field)
    # This field data has two parts: basic parameters `param`, and the raw 3D complex array `dfl`:
    G.output.field["end"]["param"]
    G.output.field["end"]["dfl"].shape
    # Sum over y and compute the absolute square
    dfl = G.output.field["end"]["dfl"]
    param = G.output.field["end"]["param"]
    dat2 = np.abs(np.sum(dfl, axis=1)) ** 2
    plt.imshow(dat2)

    def plot_slice(i=0):
        dat = np.angle(dfl[:, :, i])
        dx = param["gridsize"] * 1e6
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
    final_bunching = G.output.beam["bunching"][-1, :]
    _current = G.output.beam["current"][-1, :]
    s = G.output.global_["s"]
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
    G.output.units("Beam/betax")
    # # Plotting
    #
    # Convenient plotting of the data in `.output` is provided by `.plot`. The default is to plot the power. Depending on the key these statistics are averaged or integrated over the slices. Some keys like `power` are converted to `peak_power`, while `field_energy` is the integral over `field_power`.
    print(G.output.alias["power"])
    G.plot()
    # Left and right axes can be set this way:
    G.plot(
        "field_energy", yscale="log", y2=["beam_xsize", "beam_ysize"], ylim2=(0, 100e-6)
    )
    # By default, these plots average over slices. In the case of beam sizes, simply averaging these does not take into account the effect of misaligned slices. To plot this, LUME-Genesis provides additional `beam_sigma_x`, `beam_sima_y`, `beam_sigma_energy` keys that properly project these quantities. The difference is noticable in the energy spread calculation:
    G.plot(["beam_sigma_energy", "Beam/energyspread"], ylim=(0, 100))
    G.plot(["field_xsize", "field_ysize"])
    plt.imshow(G.output.field_info["power"], aspect="auto")
    G.archive("archived.h5")
    Grestored = Genesis4.from_archive("archived.h5")
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
