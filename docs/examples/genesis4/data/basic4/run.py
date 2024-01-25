import h5py
import numpy as np
import matplotlib.pyplot as plt

from genesis.version4.genesis4 import Genesis4Python
from genesis.version4.input import (
    Genesis4CommandInput,
    Setup,
    Time,
    Field,
    ProfileArray,
    Beam,
    Lattice,
    MainInput,
    Track,
    Reference,
)

with h5py.File("beam_current.h5") as fp:
    current_x = np.asarray(fp["s"])
    current = np.asarray(fp["current"])

with h5py.File("beam_gamma.h5") as fp:
    gamma_x = np.asarray(fp["s"])
    gamma = np.asarray(fp["gamma"])


main = MainInput(
    namelists=[
        Setup(
            rootname="LCLS2_HXR_9keV",
            outputdir="",
            # lattice="hxr.lat",
            beamline="HXR",
            gamma0=19174.0776,
            lambda0=1.3789244869952112e-10,
            delz=0.026,
            seed=84672,
            npart=1024,
        ),
        Time(slen=0.0000150, sample=200),
        Field(
            dgrid=0.0001,
            ngrid=101,
            harm=1,
            accumulate=True,
        ),
        ProfileArray(
            label="beamcurrent",
            xdata=current_x,
            ydata=current,
        ),
        ProfileArray(
            label="beamgamma",
            xdata=gamma_x,
            ydata=gamma,
        ),
        Beam(
            # TODO: allow usage of beam_current instance directly
            current=Reference("beamcurrent"),
            gamma=Reference("beamgamma"),
            delgam=3.97848,
            ex=4.000000e-7,
            ey=4.000000e-7,
            betax=7.910909406464387,
            betay=16.8811786213468987,
            alphax=-0.7393217413918415,
            alphay=1.3870723536888105,
        ),
        Track(
            zstop=1.0,  # stop quickly
            field_dump_at_undexit=False,
        ),
    ],
)

input = Genesis4CommandInput(
    # main=MainInput.from_file("cu_hxr.in"),
    main=main,
    lattice=Lattice.from_file("hxr.lat"),
    output_path="foobar",
)

cmd = Genesis4Python(input=input, verbose=True)
cmd.path = "/tmp/genesis4-test"
cmd.write_input()
print(cmd.get_run_script())
res = cmd.run()

res.plot()
plt.ioff()
plt.show()
