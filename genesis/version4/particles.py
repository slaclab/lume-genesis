from __future__ import annotations
import logging
import pathlib
from typing import Dict, List, Optional, Type, TypeVar, Union

import h5py
import numpy as np
import pydantic
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import genesis4_par_to_data

from .types import BaseModel, AnyPath, NDArray

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Genesis4Slice(BaseModel):
    """Raw Genesis 4 particle file slice data."""

    index: int
    current: NDArray
    gamma: NDArray
    px: NDArray
    py: NDArray
    theta: NDArray
    x: NDArray
    y: NDArray
    units: Dict[str, str]

    @classmethod
    def from_h5(cls, h5: h5py.Group, index: int) -> Genesis4Slice:
        """
        Create a Genesis4Slice from an h5py group.

        Parameters
        ----------
        h5 : h5py.File
        index : int
            The slice index.

        Returns
        -------
        Genesis4Slice
        """
        units = {
            key: h5[key].attrs["unit"].decode() for key in h5 if "unit" in h5[key].attrs
        }
        return cls(
            index=index,
            current=np.asarray(h5["current"]),
            gamma=np.asarray(h5["gamma"]),
            px=np.asarray(h5["px"]),
            py=np.asarray(h5["py"]),
            theta=np.asarray(h5["theta"]),
            x=np.asarray(h5["x"]),
            y=np.asarray(h5["y"]),
            units=units,
        )

    def write_h5(self, h5: h5py.Group) -> None:
        """
        Write the raw particle data back to an opened h5py file.

        Parameters
        ----------
        h5 : h5py.File
        """

        def write_array(key: str):
            h5[key] = getattr(self, key)
            units = self.units.get(key, None)
            if units is not None:
                h5[key].attrs["unit"] = np.bytes_(units)

        to_write = set(type(self).model_fields) - {"index", "units"}
        for attr in to_write:
            write_array(attr)


class Genesis4ParticleData(BaseModel):
    """
    Raw Genesis 4 particle data, consisting of a number of slices.

    Note that we recommend in general that you use ParticleGroup from
    OpenPMD-beamphysics.

    This should primarily be used in the case of specifying the initial
    particle condition based on prior Genesis4-written particle data.
    ParticleGroup is not supported for this use case.
    """

    slicelength: float = 0.0
    slicespacing: float = 0.0  # slicelength internally
    refposition: float = 0.0  # s0 internally
    beamletsize: int = 0  # nbins internally
    one4one: bool = False

    slices: List[Genesis4Slice] = pydantic.Field(default_factory=dict)
    slicerange_min: Optional[float] = None
    slicerange_max: Optional[float] = None
    slicerange_inc: Optional[float] = None
    units: Dict[str, str] = pydantic.Field(default_factory=dict)

    @classmethod
    def from_h5(cls, h5: h5py.File) -> Genesis4ParticleData:
        """
        Create a Genesis4ParticleData from an h5py File.

        Parameters
        ----------
        h5 : h5py.File

        Returns
        -------
        Genesis4ParticleData
        """

        def get_scalar(key: str, cast_type: Type[T]) -> T:
            if key not in h5:
                return None
            value = np.asarray(h5[key])
            return cast_type(value[0])

        slicecount = get_scalar("slicecount", int)
        assert isinstance(slicecount, int)
        slices = [
            Genesis4Slice.from_h5(h5.require_group(f"slice{slice:06d}"), slice)
            for slice in range(1, slicecount + 1)
        ]
        units = {
            key: h5[key].attrs["unit"].decode() for key in h5 if "unit" in h5[key].attrs
        }
        return cls(
            slices=slices,
            slicelength=get_scalar("slicelength", float),
            slicespacing=get_scalar("slicespacing", float),
            refposition=get_scalar("refposition", float),
            beamletsize=get_scalar("beamletsize", int),
            one4one=get_scalar("one4one", bool),
            slicerange_min=get_scalar("slicerange_min", float),
            slicerange_max=get_scalar("slicerange_max", float),
            slicerange_inc=get_scalar("slicerange_inc", float),
            units=units,
        )

    @classmethod
    def from_filename(cls, path: AnyPath) -> Genesis4ParticleData:
        """
        Load raw Genesis4ParticleData from a filename.

        Parameters
        ----------
        path : pathlib.Path or str

        Returns
        -------
        Genesis4ParticleData
        """
        with h5py.File(path, "r") as h5:
            return cls.from_h5(h5)

    def write_h5(self, h5: h5py.File) -> None:
        """
        Write the raw particle data back to an opened h5py file.

        Parameters
        ----------
        h5 : h5py.File
        """

        def write_scalar(key: str):
            value = getattr(self, key)
            if value is None:
                return
            h5[key] = np.asarray([value])
            units = self.units.get(key, None)
            if units is not None:
                h5[key].attrs["unit"] = np.bytes_(units)

        to_write = set(type(self).model_fields) - {"slices", "units"}
        for key in to_write:
            write_scalar(key)

        h5["slicecount"] = np.asarray([len(self.slices)])
        for slice in self.slices:
            slice.write_h5(h5.create_group(f"slice{slice.index:06d}"))

    def write_genesis4_distribution(self, path: AnyPath, verbose: bool = True) -> None:
        if verbose:
            logger.info(
                "Writing Genesis4 particle data to %s (%d slices)",
                path,
                len(self.slices),
            )
        with h5py.File(path, "w") as h5:
            return self.write_h5(h5)


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
