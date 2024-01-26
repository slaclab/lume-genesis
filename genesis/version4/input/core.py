from __future__ import annotations
import logging
from contextlib import contextmanager

import dataclasses
import pathlib
from typing import (
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import lark
import h5py
import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import c_light

from . import util
from .generated_lattice import BeamlineElement
from .generated_main import Importdistribution, NameList, Reference, ProfileFile, Setup
from .types import AnyPath, ArrayType, Float, ValueType

LATTICE_GRAMMAR = pathlib.Path("version4") / "input" / "lattice.lark"
MAIN_INPUT_GRAMMAR = pathlib.Path("version4") / "input" / "main_input.lark"

logger = logging.getLogger(__name__)

LineItem = Union[str, "DuplicatedLineItem", "PositionedLineItem", BeamlineElement]


@dataclasses.dataclass
class DuplicatedLineItem:
    """
    A Genesis 4 lattice Line item which is at a certain position.

    Attributes
    ----------
    label : str
        The name/label of the line item.
    count : int
        The number of times to repeat the line item.
    """

    label: str
    count: int

    @classmethod
    def from_string(cls, value: str) -> DuplicatedLineItem:
        count, label = value.split("*", 1)
        return cls(
            label=label.strip(),
            count=int(count),
        )

    def __str__(self) -> str:
        return f"{self.count}*{self.label}"


@dataclasses.dataclass
class PositionedLineItem:
    """
    A Genesis 4 lattice Line item which is at a certain position.

    Attributes
    ----------
    label : str
        The name/label of the line item.
    position : Float
        The position of the element.
    """

    label: str
    position: Float

    @classmethod
    def from_string(cls, value: str) -> PositionedLineItem:
        label, position = value.split("@", 1)
        return cls(
            label=label.strip(),
            position=float(position.strip()),
        )

    def __str__(self) -> str:
        return f"{self.label}@{self.position}"


def _fix_line_item(line_item: LineItem) -> LineItem:
    """Make the appropriate dataclass for a serialized line item, if necessary."""
    if isinstance(line_item, (DuplicatedLineItem, PositionedLineItem)):
        return line_item
    if isinstance(line_item, BeamlineElement):
        return line_item
    if "@" in line_item:
        return PositionedLineItem.from_string(line_item)
    if "*" in line_item:
        return DuplicatedLineItem.from_string(line_item)
    return line_item


@dataclasses.dataclass(repr=False)
class Line(BeamlineElement):
    """
    A Genesis 4 beamline Line.

    Attributes
    ----------
    elements : list of LineItem
        Elements contained in the line.
    label : str, optional
        An optional label to attach to the line.
    """

    _genesis_name_: ClassVar[str] = "line"
    elements: List[LineItem] = dataclasses.field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        self.elements = [_fix_line_item(item) for item in self.elements]

    @property
    def _string_elements(self) -> List[str]:
        return [
            elem.label if isinstance(elem, BeamlineElement) else str(elem)
            for elem in self.elements
        ]

    def to_dict(self) -> Dict:
        """
        Get a serialized (dictionary representation) of this beamline element.
        """
        return {
            "type": "line",
            "elements": self._string_elements,
        }

    def __str__(self) -> str:
        elements = ", ".join(self._string_elements)
        return "".join(
            (
                self.label,
                ": LINE = {",
                elements,
                "};",
            )
        )

    def __repr__(self) -> str:
        return util.get_non_default_repr(self)

    @classmethod
    def from_labels(
        cls,
        elements: Dict[str, BeamlineElement],
        *element_labels: str,
        label: str = "",
    ) -> Line:
        """
        Create a `Line` from labeled beamline elements.

        Parameters
        ----------
        elements : Dict[str, BeamlineElement]
            The element dictionary.
        *element_labels : str
            Labels of elements.  May be either a single label per argument or a
            whitespace-delimited string of labels.
        label : str
            Label name for the line.

        Returns
        -------
        Line
        """
        try:
            return cls(
                elements=[
                    elements[label]
                    for labels in element_labels
                    for label in labels.strip().split()
                ],
                label=label,
            )
        except KeyError as ex:
            raise ValueError(
                f"Label {ex} is not present in the beamline element dictionary. "
                f"The following are valid: {tuple(elements.keys())}"
            )


@dataclasses.dataclass(repr=False)
class Lattice:
    """
    A Genesis 4 beamline Lattice configuration.

    Attributes
    ----------
    elements : list of BeamlineElement or Line
        Elements contained in the lattice.
    filename : pathlib.Path or None
        The filename from which this lattice was loaded.
    """

    elements: Dict[str, Union[BeamlineElement, Line]] = dataclasses.field(
        default_factory=dict
    )
    filename: Optional[pathlib.Path] = None

    def __str__(self) -> str:
        return self.to_genesis()

    def to_genesis(self) -> str:
        self.fix_labels()
        return "\n".join(str(element) for element in self.elements.values())

    def __repr__(self) -> str:
        return util.get_non_default_repr(self)

    def fix_labels(self) -> None:
        for label, element in self.elements.items():
            if element.label != label:
                if element.label:
                    logger.warning(
                        "Renaming beamline element in lattice from %s to %s",
                        element.label,
                        label,
                    )
                element.label = label

    def to_dict(self) -> Dict[str, Dict]:
        """Serialize this lattice to a list of dictionaries."""
        self.fix_labels()
        return {label: element.to_dict() for label, element in self.elements.items()}

    @classmethod
    def from_dict(
        cls, contents: Dict[str, Dict], filename: Optional[pathlib.Path] = None
    ) -> Lattice:
        """
        Load a Lattice instance from a serialized dictionary.

        Parameters
        ----------
        contents : dict of label to element dict
            The serialized contents of the lattice.

        Returns
        -------
        Lattice
        """
        elements = {}
        for label, dct in contents.items():
            elements[label] = BeamlineElement.from_dict(dct)
            elements[label].label = label
        return cls(
            elements=elements,
            filename=filename,
        )

    @classmethod
    def from_contents(
        cls, contents: str, filename: Optional[AnyPath] = None
    ) -> Lattice:
        """
        Load a lattice from its file contents.

        Parameters
        ----------
        contents : str
            The contents of the lattice file.
        filename : AnyPath or None, optional
            The filename of the lattice, if known.

        Returns
        -------
        Lattice
        """
        parser = new_lattice_parser()
        tree = parser.parse(contents)
        filename = filename or "unknown"
        return _LatticeTransformer(filename).transform(tree)

    @classmethod
    def from_file(cls, filename: AnyPath) -> Lattice:
        """
        Load a lattice file from disk.

        Parameters
        ----------
        filename : AnyPath
            The filename to load.

        Returns
        -------
        Lattice
        """
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)


@dataclasses.dataclass
class ProfileArray(NameList):
    r"""
    ProfileArray is a lume-genesis convenience class for generating
    ``profile_file`` namelists.

    Attributes
    ----------
    x_label : str
        Name of the profile, which is used to refer to it in later calls of namelists
    xdata : list of float or np.ndarray, default=''
        The `s`-position for the look-up table.
    ydata : list of float or np.ndarray, default=''
        The function values of the look-up table.
    isTime : bool, default=False
        If true the `s`-position is a time variable and therefore multiplied with the
        speed of light `c` to get the position in meters.
    reverse : bool, default=False
        if true the order in the look-up table is reverse. This is sometimes needed
        because time and spatial coordinates differ sometimes by a minus sign.
    _filename : str, optional
        By default, this is a randomly-generated filename that lume-genesis
        manages for you.  If desirable, you may set a fixed filename relative
        to the main input file.  Path delimiters (such as ``/``) are not
        allowed.
    """

    _genesis_name_: ClassVar[str] = ""
    label: str
    xdata: ArrayType
    ydata: ArrayType
    isTime: bool = False
    reverse: bool = False
    autoassign: bool = False
    _filename: str = dataclasses.field(
        default_factory=util.get_temporary_filename,
    )
    _x_label: str = "x"
    _y_label: str = "y"

    def write(self, base_path: AnyPath) -> pathlib.Path:
        if "/" in self._filename:
            raise ValueError(
                "Filename is not allowed to contain the path separator "
                "forward slash (/).  Genesis 4 interprets these as part of "
                "the HDF group."
            )
        path = pathlib.Path(base_path) / self._filename
        with h5py.File(path, "w") as fp:
            # for key, value in self.hdf_data.items():
            #     fp[key] = value
            fp.update(self.get_hdf_data())
        logger.info("Saved %s to %s", self.label, path)
        return path

    def get_hdf_data(self) -> Dict[str, ArrayType]:
        """Get all HDF5 data to be written for Genesis 4."""
        return {
            self._x_label: self.xdata,
            self._y_label: self.ydata,
        }

    def to_profile_file(self) -> ProfileFile:
        """Convert this ProfileArray into a 'profile_file' namelist for Genesis 4."""
        return ProfileFile(
            label=self.label,
            xdata=f"{self._filename}/{self._x_label}",
            ydata=f"{self._filename}/{self._y_label}",
            isTime=self.isTime,
            reverse=self.reverse,
            autoassign=self.autoassign,
        )

    def to_genesis(self) -> str:
        return self.to_profile_file().to_genesis()


@dataclasses.dataclass
class InitialParticles(NameList):
    r"""
    InitialParticles is a lume-genesis convenience class for generating
    ``importdistribution`` namelists.

    Attributes
    ----------
    particles : ParticleGroup
        Initial particles.
    _filename : str, optional
        By default, this is a randomly-generated filename that lume-genesis
        manages for you.  If desirable, you may set a fixed filename relative
        to the main input file.  Path delimiters (such as ``/``) are not
        allowed.
    """

    _genesis_name_: ClassVar[str] = ""
    particles: ParticleGroup
    _filename: str = dataclasses.field(
        default_factory=util.get_temporary_filename,
    )

    @property
    def slen(self) -> float:
        """
        Length of the time window in meters.

        Note that for parallel jobs this might be adjusted towards larger
        values.

        This can be used to update the Time namelist with information
        from this initial particle distribution.
        """
        return max(
            c_light * self.particles.t.ptp(),
            self.particles.z.ptp(),
        )

    def write(self, base_path: AnyPath) -> pathlib.Path:
        """
        Write the particles to disk for usage with Genesis 4.

        Lume-genesis will assign a filename using this base path. Except for
        expert-level usage, you should not need to configure the specific
        filename as it is transparent to the user.

        Parameters
        ----------
        base_path : AnyPath
            The directory to write to.

        Returns
        -------
        pathlib.Path
            [TODO:description]
        """
        if "/" in self._filename:
            raise ValueError(
                "Filename is not allowed to contain the path separator "
                "forward slash (/).  Genesis 4 interprets these as part of "
                "the HDF group."
            )
        path = pathlib.Path(base_path) / self._filename
        self.particles.write_genesis4_distribution(str(path))
        logger.info("Saved particles to %s", path)
        return path

    def to_import_distribution(self) -> Importdistribution:
        return Importdistribution(
            file=self._filename,
            charge=self.particles.charge,
        )

    def to_genesis(self) -> str:
        return self.to_import_distribution().to_genesis()


T_NameList = TypeVar("T_NameList", bound=NameList)


@dataclasses.dataclass
class MainInput:
    """
    A Genesis 4 main input configuration file.

    Attributes
    ----------
    namelists : list of NameList
        Elements contained in the lattice.
    filename : pathlib.Path or None
        The filename from which this was loaded.
    """

    namelists: List[NameList] = dataclasses.field(default_factory=list)
    filename: Optional[pathlib.Path] = None

    def __str__(self) -> str:
        return self.to_genesis()

    def to_genesis(self) -> str:
        return "\n\n".join(str(namelist) for namelist in self.namelists)

    @property
    def by_namelist(self) -> Dict[Type[T_NameList], List[T_NameList]]:
        """Get namelists organized by their class."""
        by_namelist = {}
        for namelist in self.namelists:
            by_namelist.setdefault(type(namelist), [])
            by_namelist[type(namelist)].append(namelist)
        return by_namelist

    def get_setup(self) -> Setup:
        """Get the required setup namelist."""
        setups = self.by_namelist.get(Setup, [])
        if len(setups) == 0:
            raise ValueError("Setup is not defined in the input.")
        if len(setups) > 1:
            raise ValueError("Multiple setup namelists were defined in the input.")
        return setups[0]

    def to_dicts(self) -> List[Dict]:
        """Serialize this main input to a list of dictionaries."""
        return [namelist.to_dict() for namelist in self.namelists]

    @classmethod
    def from_dicts(
        cls, contents: Sequence[Dict], filename: Optional[pathlib.Path] = None
    ) -> MainInput:
        """
        Load main input from a list of serialized dictionaries.

        Parameters
        ----------
        contents : sequence of dict
            The serialized contents of the main input file.

        Returns
        -------
        MainInput
        """
        return cls(
            namelists=[NameList.from_dict(dct) for dct in contents],
            filename=filename,
        )

    @classmethod
    def from_contents(
        cls, contents: str, filename: Optional[AnyPath] = None
    ) -> MainInput:
        """
        Load main input from its file contents.

        Parameters
        ----------
        contents : str
            The contents of the main input file.
        filename : AnyPath or None, optional
            The filename, if known.

        Returns
        -------
        MainInput
        """
        parser = new_main_input_parser()
        tree = parser.parse(contents)
        filename = filename or "unknown"
        return _MainInputTransformer(filename).transform(tree)

    @classmethod
    def from_file(cls, filename: AnyPath) -> MainInput:
        """
        Load a main input file from disk.

        Parameters
        ----------
        filename : AnyPath
            The filename to load.

        Returns
        -------
        MainInput
        """
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)


def _split_file(namelist: NameList, attr: str, file: str) -> str:
    """Get a filename from the namelist."""
    if attr in ("xdata", "ydata"):
        # We can't handle paths here, although perhaps lume-genesis
        # could fix this for users... hmm.
        return file.split("/")[0]
    if attr in ("file",):
        return file
    raise NotImplementedError(attr)


@dataclasses.dataclass
class Genesis4Input:
    """
    All Genesis 4-related command input.

    Attributes
    ----------
    main : MainInput
        The main input. This contains all the namelists, starting with
        `&setup`.
    lattice : Lattice
        The lattice definition.
    beamline : str, optional
        Optional override for `beamline` in the setup namelist.
    lattice_name : str, optional
        Optional override for `lattice` in the setup namelist.
    seed : str, optional
        Optional override for `seed` in the setup namelist.
    output_path : Optional[AnyPath] = None
        Optional override for `rootname` in the setup namelist.
    source_path : Optional[AnyPath] = None
        When using Genesis 4-compatible input as strings, this is the directory
        where we expect to find other input HDF5 files.
    lattice_filename : str = "genesis.lat"
        The filename to use when writing the lattice.  As a user,
        you should not need to worry about this as Genesis4Input will
        handle it for you.
    input_filename : str = "input.in"
        The filename to use when writing the main input file. As a user, you
        should not need to worry about this as Genesis4Input will handle it for
        you.
    """

    main: MainInput
    lattice: Lattice
    beamline: Optional[str] = None
    lattice_name: Optional[str] = None
    seed: Optional[str] = None
    source_path: pathlib.Path = pathlib.Path(".")
    output_path: Optional[AnyPath] = None
    lattice_filename: str = "genesis.lat"
    input_filename: str = "input.in"

    def get_arguments(self, workdir: AnyPath = pathlib.Path(".")) -> List[str]:
        """
        Get all of the command-line arguments for running Genesis 4.

        Parameters
        ----------
        workdir : str or pathlib.Path
            The working directory for Genesis 4. This will be where all
            input and output files are written.

        Returns
        -------
        list of str
            Individual arguments to pass to Genesis 4.
        """
        path = pathlib.Path(workdir)
        optional_args = []
        if self.beamline:
            optional_args.extend(["-b", self.beamline])
        if self.lattice_name:
            optional_args.extend(["-l", self.lattice_name])
        if self.seed is not None:
            optional_args.extend(["-s", self.seed])
        if self.output_path is not None:
            optional_args.extend(["-o", str(path / self.output_path)])
        return [
            *optional_args,
            "-l",
            str(path / self.lattice_filename),
            str(path / self.input_filename),
        ]

    def write(self, workdir: AnyPath = pathlib.Path(".")) -> List[pathlib.Path]:
        """
        Write all input files for executing Genesis 4.

        Parameters
        ----------
        workdir : AnyPath
            The working directory for Genesis 4. This will be where all
            input and output files are written.

        Returns
        -------
        List[pathlib.Path]
            Paths written.
        """
        path = pathlib.Path(workdir)
        path.mkdir(parents=True, mode=0o755, exist_ok=True)

        extra_paths = []

        for cls in (ProfileArray, InitialParticles):
            for idx, namelist in enumerate(self.main.by_namelist.get(cls, [])):
                assert isinstance(namelist, cls)
                namelist._filename = f"{cls.__name__}_{idx}.h5"
                extra_paths.append(namelist.write(workdir))

        for idx, namelist in enumerate(self.main.namelists):
            for attr in ("xdata", "ydata", "file"):
                file = getattr(namelist, attr, None)
                if file is not None and not isinstance(file, np.ndarray):
                    file = _split_file(namelist, attr, file)
                    if file:
                        source_file = self.source_path / file
                        dest_file = path / file
                        if not dest_file.is_symlink():
                            dest_file.symlink_to(source_file)
                        extra_paths.append(dest_file)

        main_config = self.main.to_genesis()
        with open(path / self.input_filename, "wt") as fp:
            print(main_config, file=fp)

        logger.debug(
            "Wrote main config to %s:\n%s",
            main_config,
            path / self.input_filename,
        )

        lattice = self.lattice.to_genesis()
        with open(path / self.lattice_filename, "wt") as fp:
            print(lattice, file=fp)

        logger.debug(
            "Wrote lattice to %s:\n%s",
            lattice,
            path / self.lattice_filename,
        )

        return [path / self.input_filename, path / self.lattice_filename, *extra_paths]

    @contextmanager
    def write_context(
        self, workdir: AnyPath
    ) -> Generator[List[pathlib.Path], None, None]:
        """
        Write all input files for executing Genesis 4.

        This is a context manager. When the context manager exits, all input
        files will be cleaned up (i.e., deleted from disk).

        Parameters
        ----------
        workdir : AnyPath
            The working directory for Genesis 4. This will be where all
            input and output files are written.

        Yields
        ------
        List[pathlib.Path]
            Paths written.
        """
        paths = self.write(workdir)
        yield paths
        for path in paths:
            path.unlink(missing_ok=True)

    @classmethod
    def from_strings(
        cls,
        main: str,
        lattice: str,
        *,
        source_path: pathlib.Path = pathlib.Path("."),
    ) -> Genesis4Input:
        """
        Work directly with Genesis 4-compatible inputs.

        If the input refers to files that already exist on disk, ensure
        that `source_path` is set correctly.
        """
        return cls(
            main=MainInput.from_contents(main),
            lattice=Lattice.from_contents(lattice),
            source_path=source_path,
        )


def new_parser(filename: AnyPath, **kwargs) -> lark.Lark:
    """
    Get a new parser for one of the packaged grammars.

    Parameters
    ----------
    filename : str
        The packaged grammar filename.
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return lark.Lark.open_from_package(
        "genesis",
        str(filename),
        parser="lalr",
        maybe_placeholders=True,
        propagate_positions=True,
        **kwargs,
    )


def new_lattice_parser(**kwargs) -> lark.Lark:
    """
    Get a new parser for the packaged Lattice input grammar.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return new_parser(LATTICE_GRAMMAR, **kwargs)


def new_main_input_parser(**kwargs) -> lark.Lark:
    """
    Get a new parser for the packaged main input file grammar.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return new_parser(MAIN_INPUT_GRAMMAR, **kwargs)


def _fix_parameters(
    cls: Union[Type[BeamlineElement], Type[NameList]],
    params: Dict[str, lark.Token],
) -> Tuple[Dict[str, ValueType], Dict[str, str]]:
    """
    Fix parameters to beamline elements when transforming with
    :class:``_LatticeTransformer`.

    Parameters
    ----------
    cls : Type[BeamlineElement]
        The dataclass associated with the beamline element.  This is used to
        determine the attribute name and data type associated with the
        parameter.
    params : Dict[str, lark.Token]
        Parameter name map to lark Token value.

    Returns
    -------
    Dict[str, ValueType]
        Keyword arguments for the dataclass.
    Dict[str, str]
        Unexpected arguments for the dataclass; unsure what to do with them.
    """
    kwargs: Dict[str, ValueType] = {}
    extra: Dict[str, str] = {}
    mapping = cls._parameter_to_attr_
    for name, value in params.items():
        name = mapping.get(name, name)
        dtype = cls.__annotations__.get(name, None)
        allow_reference = "| Reference" in dtype
        dtype = dtype.replace("| Reference", "").strip()
        if dtype is None:
            extra[name] = str(value)
        elif value.startswith("@") and allow_reference:
            kwargs[name] = Reference(str(value[1:]).strip())
        elif dtype == "int":
            kwargs[name] = int(value)
        elif dtype == "float":
            kwargs[name] = float(value)
        elif dtype == "bool":
            kwargs[name] = value.lower() == "true"
        elif dtype == "str":
            kwargs[name] = value.strip()
        else:
            raise RuntimeError(f"Unexpected type annotation hit for {name}: {dtype}")
    return kwargs, extra


class _LatticeTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`Lattice`.

    Attributes
    ----------
    _filename : str
        Filename source of the input.
    """

    _filename: Optional[pathlib.Path]
    # This maps, e.g., "UNDU" -> Undulator
    type_map: Dict[str, Type[BeamlineElement]] = {
        cls.__name__.upper()[:4]: cls for cls in BeamlineElement.__subclasses__()
    }

    def __init__(self, filename: AnyPath) -> None:
        super().__init__()
        self._filename = pathlib.Path(filename)

    @lark.v_args(inline=True)
    def line(
        self,
        label: lark.Token,
        _: lark.Token,  # line
        element_list: List[LineItem],
    ) -> Tuple[str, Line]:
        return str(label), Line(elements=element_list)

    @lark.v_args(inline=True)
    def parameter_set(
        self,
        parameter: lark.Token,
        value: lark.Token,
    ) -> Tuple[str, lark.Token]:
        return str(parameter), value

    def parameter_list(
        self, sets: List[Tuple[str, ValueType]]
    ) -> List[Tuple[str, ValueType]]:
        return list(sets)

    @lark.v_args(inline=True)
    def beamline_element(
        self,
        label: lark.Token,
        type_: lark.Token,
        parameter_list: List[Tuple[str, lark.Token]],
    ) -> Tuple[str, BeamlineElement]:
        cls = self.type_map[type_.upper()[:4]]
        parameters, unknown = _fix_parameters(cls, dict(parameter_list))
        if unknown:
            raise ValueError(
                f"Beamline element {label} received unexpected parameter(s): "
                f"{unknown}"
            )
        return str(label), cls(**parameters)

    @lark.v_args(inline=True)
    def duplicate_item(
        self,
        count: lark.Token,
        label: lark.Token,
    ) -> DuplicatedLineItem:
        return DuplicatedLineItem(
            label=str(label),
            count=int(count),
        )

    @lark.v_args(inline=True)
    def positioned_item(
        self,
        label: lark.Token,
        position: lark.Token,
    ) -> PositionedLineItem:
        return PositionedLineItem(
            label=str(label),
            position=float(position),
        )

    @lark.v_args(inline=True)
    def line_item(
        self, item: Union[lark.Token, DuplicatedLineItem, PositionedLineItem]
    ) -> Union[str, DuplicatedLineItem, PositionedLineItem]:
        if isinstance(item, lark.Token):
            return str(item)
        return item

    def element_list(
        self, items: List[Union[lark.Token, DuplicatedLineItem, PositionedLineItem]]
    ) -> List[Union[lark.Token, DuplicatedLineItem, PositionedLineItem]]:
        return items

    def lattice(
        self, elements: List[Tuple[str, Union[BeamlineElement, Line]]]
    ) -> Lattice:
        return Lattice(
            dict(elements),
            filename=self._filename,
        )


class _MainInputTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`MainInput`.

    Attributes
    ----------
    _filename : str
        Filename source of the input.
    """

    _filename: Optional[pathlib.Path]
    # This maps, e.g., "setup" to the Setup dataclass
    type_map: Dict[str, Type[NameList]] = {
        cls._genesis_name_: cls for cls in NameList.__subclasses__()
    }

    def __init__(self, filename: AnyPath) -> None:
        super().__init__()
        self._filename = pathlib.Path(filename)

    @lark.v_args(inline=True)
    def parameter_set(
        self,
        parameter: lark.Token,
        value: lark.Token,
    ) -> Tuple[str, lark.Token]:
        return (str(parameter), value)

    def parameter_list(
        self, sets: List[Tuple[str, ValueType]]
    ) -> List[Tuple[str, ValueType]]:
        return list(sets)

    @lark.v_args(inline=True)
    def namelist(
        self,
        name: lark.Token,
        parameter_list: List[Tuple[str, lark.Token]],
        _: lark.Token,  # end
    ) -> NameList:
        cls = self.type_map[name]
        parameters, unknown = _fix_parameters(cls, dict(parameter_list))
        if unknown:
            raise ValueError(
                f"Namelist {name} received unexpected parameter(s): " f"{unknown}"
            )
        return cls(**parameters)

    @lark.v_args(inline=True)
    def main_input(self, *namelists: NameList) -> MainInput:
        return MainInput(namelists=list(namelists), filename=self._filename)
