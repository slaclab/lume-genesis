from __future__ import annotations

import pathlib
import pydantic
from typing import Union

import h5py
import numpy as np

from . import readers
from .types import (
    AnyPath,
    BaseModel,
    FieldFileParams,
    FileKey,
    NDArray,
)


def get_key_from_filename(fn: str) -> FileKey:
    """
    Get the user-facing key from a given particle/field filename.

    For named files such as "Example2.fld.h5", this would be the unique
    portion: "Example2".

    For integration step labeled files such as "Example2.123.fld.h5", this
    would be the integration step (as an integer): 123.

    Parameters
    ----------
    fn : str

    Returns
    -------
    str or int
    """
    # basename.STEP.par.h5
    parts = fn.split(".")
    try:
        return int(parts[-3])
    except ValueError:
        return parts[-3]
    except IndexError:
        return fn


class FieldFile(BaseModel):
    label: FileKey = ""
    dfl: NDArray = pydantic.Field(default_factory=lambda: np.zeros((1, 1, 1)))
    param: FieldFileParams = pydantic.Field(default_factory=FieldFileParams)

    @property
    def hdf5_filename(self) -> str:
        """
        The label-based HDF5 filename which Genesis would write.

        For a label of `'end'` this would be `'end.fld.h5'`.

        Returns
        -------
        str
        """
        return f"{self.label}.fld.h5"

    @classmethod
    def from_file(
        cls,
        file: Union[AnyPath, h5py.File],
    ) -> FieldFile:
        """
        Load a single Genesis 4-format field file (``*.fld.h5``).

        Returns
        -------
        FieldFile
        """
        if isinstance(file, h5py.File):
            dfl, param = readers.load_genesis4_fields(file)
            filename = file.filename
        else:
            filename = pathlib.Path(file)
            if not h5py.is_hdf5(filename):
                raise ValueError(f"Field file {filename} is not an HDF5 file")

            with h5py.File(filename, "r") as h5:
                dfl, param = readers.load_genesis4_fields(h5)

        return cls(
            label=get_key_from_filename(pathlib.Path(filename).name),
            dfl=dfl,
            param=FieldFileParams(**param),
        )

    def _write_genesis4(
        self,
        h5: h5py.File,
    ) -> None:
        """
        Write the field data in the Genesis 4 format.

        Parameters
        ----------
        h5 : h5py.File
        """
        _nx, _ny, nz = self.dfl.shape
        h5["gridpoints"] = np.asarray([self.param.gridpoints])
        h5["gridsize"] = np.asarray([self.param.gridsize])
        h5["refposition"] = np.asarray([self.param.refposition])
        h5["wavelength"] = np.asarray([self.param.wavelength])
        h5["slicecount"] = np.asarray([nz])
        h5["slicespacing"] = np.asarray([self.param.slicespacing])
        for key, value in (self.param.model_extra or {}).items():
            if not isinstance(value, np.ndarray):
                value = np.asarray([value])
            h5[key] = value

        # Note from Sven:
        #   The order of the 1D array of the wavefront is with the x
        #   coordinates as the inner loop.
        #   So the order is (x1,y1),(x2,y1), ... (xn,y1),(x1,y2),(x2,y2),.....
        #   This is done in the routine getLLGridpoint in the field class.
        # Therefore the transpose is needed below
        for z in range(nz):
            slice_index = z + 1
            slice_group = h5.create_group(f"slice{slice_index:06}")
            slice_group["field-real"] = self.dfl[:, :, z].real.T.flatten()
            slice_group["field-imag"] = self.dfl[:, :, z].imag.T.flatten()

    def write_genesis4(
        self,
        dest: Union[AnyPath, h5py.File],
    ) -> None:
        """
        Write field data from memory to a file.

        Parameters
        ----------
        dest : h5py.File, str or pathlib.Path
            Destination to write to.  May be an open HDF5 file handle or a
            filename.
        """

        if isinstance(dest, (str, pathlib.Path)):
            with h5py.File(dest, "w") as h5:
                return self._write_genesis4(h5)
        if isinstance(dest, h5py.Group):
            return self._write_genesis4(dest)

        raise ValueError(
            f"Unsupported destination: {dest}. It should be a path or an h5py Group"
        )

    def write_openpmd_wavefront(
        self,
        dest: Union[AnyPath, h5py.Group],
        verbose: bool = True,
    ) -> None:
        """
        Write the field file information to the given HDF5 file in
        OpenPMD-wavefront format.

        Parameters
        ----------
        dest : str, pathlib.Path, or h5py.Group
            Filename or already-open h5py.Group to write to.
        """
        from .writers import write_openpmd_wavefront, write_openpmd_wavefront_h5

        if isinstance(dest, (str, pathlib.Path)):
            write_openpmd_wavefront(
                str(dest), self.dfl, self.param.model_dump(), verbose=verbose
            )
            return
        if isinstance(dest, h5py.Group):
            write_openpmd_wavefront_h5(dest, self.dfl, self.param.model_dump())
            return

        raise ValueError(type(dest))  # type: ignore[unreachable]
