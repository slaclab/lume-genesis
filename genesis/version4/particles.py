import pathlib
from typing import Union

import h5py
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import genesis4_par_to_data

from .types import AnyPath


def load_particle_group(
    h5: Union[AnyPath, h5py.File], smear: bool = True
) -> ParticleGroup:
    """
    Load a ParticleGroup from the provided h5py File instance.

    Parameters
    ----------
    h5 : h5py.File
    smear : bool

    Returns
    -------
    ParticleGroup
    """
    if isinstance(h5, (str, pathlib.Path)):
        with h5py.File(h5, "r") as fp:
            return load_particle_group(fp, smear=smear)

    if "beamletsize" in h5:
        # Genesis 4 format.
        return ParticleGroup(data=genesis4_par_to_data(h5, smear=smear))
    return ParticleGroup(h5=h5)
