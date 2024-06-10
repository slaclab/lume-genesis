from __future__ import annotations

import pathlib
from typing import Any

import h5py
from pmd_beamphysics import ParticleGroup

from .field import FieldFile
from .particles import load_particle_group
from .types import BaseModel, FileKey

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class HDF5ReferenceFile(BaseModel):
    """An externally-referenced HDF5 file."""

    key: FileKey
    filename: pathlib.Path


class LoadableFieldH5File(HDF5ReferenceFile):
    type: Literal["field"] = "field"

    def load(self, **kwargs: Any) -> FieldFile:
        with h5py.File(self.filename) as h5:
            if self.type == "field":
                return FieldFile.from_file(
                    h5,
                    **kwargs,
                )
        raise NotImplementedError(self.type)


class LoadableParticleGroupH5File(HDF5ReferenceFile):
    type: Literal["particle_group"] = "particle_group"

    def load(self, **kwargs: Any) -> ParticleGroup:
        with h5py.File(self.filename) as h5:
            return load_particle_group(h5, **kwargs)
