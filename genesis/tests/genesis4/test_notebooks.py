import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pmd_beamphysics import ParticleGroup

from genesis.version4.types import FieldFileDict

from ...version4 import Genesis4
from ..conftest import genesis4_examples


example_data = genesis4_examples / "data"


def test_example1():
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


def test_example2():
    G = Genesis4(example_data / "example2-dumps" / "Example2.in")
    output = G.run(raise_on_error=True)
    G.plot(["beam_xsize", "beam_ysize", "field_xsize", "field_ysize"])

    print("Loaded fields:", output.load_fields())
    print("Loaded particles:", output.load_particles())

    def get_slice(field: FieldFileDict, slc: int) -> np.ndarray:
        return field["dfl"][:, :, slc]

    def getWF(field: FieldFileDict, slice=0):
        ng = field["param"]["gridpoints"]
        dg = field["param"]["gridsize"]
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
