from __future__ import annotations
import dataclasses
import lark
import pathlib
from decimal import Decimal

from typing import ClassVar, Dict, List, Optional, Tuple, Type, Union


LATTICE_GRAMMAR = pathlib.Path("version4") / "lattice.lark"
AnyPath = Union[pathlib.Path, str]
Float = Union[Decimal, float]


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
        filename,
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


ValueType = Union[int, Float, bool]


@dataclasses.dataclass
class BeamlineElement:
    _lattice_to_attr_: ClassVar[Dict[str, str]] = {
        "l": "length",
    }
    _attr_to_lattice_: ClassVar[Dict[str, str]] = dict(
        (v, k) for k, v in _lattice_to_attr_.items()
    )

    label: str
    unknown_parameters: Dict[str, ValueType] = dataclasses.field(default_factory=dict)

    @property
    def parameter_dict(self) -> Dict[str, ValueType]:
        skip = {"label", "unknown_parameters"}
        data = {}
        for attr in self.__annotations__:
            if attr.startswith("_") or attr in skip:
                continue
            value = getattr(self, attr)
            default = getattr(type(self), attr, None)
            if str(value) != str(default):
                param = self._attr_to_lattice_.get(attr, attr)
                data[param] = value
        data.update(self.unknown_parameters)
        return data

    def __str__(self) -> str:
        parameters = ", ".join(
            f"{name}={value}" for name, value in self.parameter_dict.items()
        )
        type_ = type(self).__name__.upper()
        return "".join(
            (
                self.label,
                f": {type_} = " "{",
                parameters,
                "};",
            )
        )


@dataclasses.dataclass
class Undulator(BeamlineElement):
    r"""
    Attributes
    ----------
    aw : float, default = 0.0
        The dimensionless rms undulator parameter. For planar undulator this
        value is smaller by a factor $1 / \sqrt{2}$ than its K-value, while for
        helical undulator rms and peak values are identical.
    lambdau : float, default = 0.0
        Undulator period length in meter. Default is 0 m.
    nwig : int, default = 0
        Number of periods.
    helical: bool, default = False
        Boolean flag whether the undulator is planar or helical. A planar
        undulator has helical=`false`. Note that setting it to `true`, does not
        change the roll-off parameters for focusing. To be consistent they have
        to be set directly.
    kx : float, default = 0.0
        Roll-off parameter of the quadratic term of the undulator field in x.
        It is normalized with respect to $k_u^2$.
    ky : float, default = 1.0
        Roll-off parameter of the quadratic term of the undulator field in y.
    ax : float, default = 0.0
        Offset of the undulator module in $x$ in meter.
    ay : float, default = 0.0
        Offset of the undulator module in $y$ in meter.
    gradx : float, default = 0
        Relative transverse gradient of undulator field in $x$ $\equiv (1/a_w)
        \partial a_w/\partial x$.
    grady : float, default = 0
        Relative transverse gradient of undulator field in $y$ $\equiv (1/a_w)
        \partial a_w/\partial y$.
    """

    aw: Float = 0.0
    lambdau: Float = 0.0
    nwig: int = 0
    helical: bool = False
    kx: Float = 0.0
    ky: Float = 1.0
    ax: Float = 0.0
    ay: Float = 0.0
    gradx: Float = 0
    grady: Float = 0


@dataclasses.dataclass
class Quadrupole(BeamlineElement):
    """
    Attributes
    ----------
    length : float, default = 0
        Length of the quadrupole in meter.
    k1 : float, default = 0
        Normalized focusing strength in 1/m^2.
    dx : float, default = 0
        Offset in $x$ in meter.
    dy : float, default = 0
        Offset in $y$ in meter.
    """

    #: Length of the quadrupole in meter.
    length: Float = 0
    #: Normalized focusing strength in 1/m^2.
    k1: Float = 0
    #: Offset in $x$ in meter.
    dx: Float = 0
    #: Offset in $y$ in meter.
    dy: Float = 0


@dataclasses.dataclass
class Drift(BeamlineElement):
    """
    Parameters
    ----------
    length : float, default = 0.0
        Length of the drift in meter.
    """

    #: Length of the drift in meter.
    length: Float = 0.0


@dataclasses.dataclass
class Corrector(BeamlineElement):
    """
    Parameters
    ----------
    length : float, default = 0
        Length of the corrector in meter.
    cx : float, default = 0
        Angle in $x$ in units of $\gamma \beta_x$.
    cy : float, default = 0
        Angle in $y$ in units of $\gamma \beta_y$.
    """

    #: Length of the corrector in meter.
    length: Float = 0
    #: angle in $x$ in units of $\gamma \beta_x$.
    cx: Float = 0
    #: angle in $y$ in units of $\gamma \beta_y$.
    cy: Float = 0


@dataclasses.dataclass
class Chicane(BeamlineElement):
    """
    length : float, default = 0
        Length of the chicane, which consists out of 4 dipoles without focusing. The first and last are placed at the beginning and end of the reserved space. The inner ones are defined by the drift length in between. Any remaining distance, namely the length subtracted by 4 times the dipole length and twice the drift length are placed between the second and third dipole.
    lb : float, default = 0
        Of an individual dipole in meter.
    ld : float, default = 0
        between the outer and inner dipoles, projected onto the undulator axis. The actual path length is longer by the factor $1/\cos\theta$, where $\theta$ is the bending angle of an individual dipole.
    delay : float, default = 0
        length difference between the straight path and the actual trajectory in meters. Genesis 1.3 calculates the bending angle internally starting from this value. $R_{56} = 2$`delay`.
    """

    #: Length of the chicane, which consists out of 4 dipoles without focusing. The first and last are placed at the beginning and end of the reserved space. The inner ones are defined by the drift length in between. Any remaining distance, namely the length subtracted by 4 times the dipole length and twice the drift length are placed between the second and third dipole.
    length: Float = 0
    #: of an individual dipole in meter.
    lb: Float = 0
    #: between the outer and inner dipoles, projected onto the undulator axis. The actual path length is longer by the factor $1/\cos\theta$, where $\theta$ is the bending angle of an individual dipole.
    ld: Float = 0
    #: length difference between the straight path and the actual trajectory in meters. Genesis 1.3 calculates the bending angle internally starting from this value. $R_{56} = 2$`delay`.
    delay: Float = 0


@dataclasses.dataclass
class PhaseShifter(BeamlineElement):
    #: Length of the phase shifter in meter.
    length: Float = 0
    #: in the ponderomotive phase of the electrons in units of rad. Note that Genesis 1.3 is doing an autophasing, so that the electrons at reference energy are not changing in ponderomotive phase in drifts.
    phi: Float = 0


@dataclasses.dataclass
class Marker(BeamlineElement):
    #: A non-zero value enforces the dump of the field distribution of this zero length element.
    dumpfield: int = 0
    #: non-zero value enforces the dump of the particle distribution.
    dumpbeam: int = 0
    #: non-zero value enforces the sorting of particles, if one-for-one simulations are enabled.
    sort: int = 0
    #: non-zero value stops the execution of the tracking module. Note that the output file still contains the full length with zeros as output for those integration steps which are no further calculated.
    stop: int = 0


@dataclasses.dataclass
class DuplicatedLineItem:
    #: Duplication count.
    count: int
    #: Element name.
    label: str

    def __str__(self) -> str:
        return f"{self.count}*{self.label}"


@dataclasses.dataclass
class PositionedLineItem:
    #: Position in meters.
    position: Float
    #: Element name.
    label: str

    def __str__(self) -> str:
        return f"{self.label}@{self.position}"


LineItem = Union[str, DuplicatedLineItem, PositionedLineItem]


@dataclasses.dataclass
class Line:
    label: str
    elements: List[LineItem] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        elements = ", ".join(str(element) for element in self.elements)
        return "".join(
            (
                self.label,
                ": LINE = {",
                elements,
                "};",
            )
        )


@dataclasses.dataclass
class Lattice:
    elements: List[Union[BeamlineElement, Line]] = dataclasses.field(
        default_factory=list
    )
    filename: Optional[pathlib.Path] = None

    def __str__(self) -> str:
        return "\n".join(str(element) for element in self.elements)

    @classmethod
    def from_file(cls, filename: AnyPath) -> Lattice:
        with open(filename) as fp:
            contents = fp.read()

        parser = new_lattice_parser()
        tree = parser.parse(contents)
        return _LatticeTransformer(filename).transform(tree)


def _fix_parameters(
    cls: Type[BeamlineElement],
    params: Dict[str, lark.Token],
) -> Tuple[Dict[str, ValueType], Dict[str, str]]:
    kwargs: Dict[str, ValueType] = {}
    extra: Dict[str, str] = {}
    for name, value in params.items():
        name = cls._lattice_to_attr_.get(name, name)
        dtype = cls.__annotations__.get(name, None)
        if dtype is None:
            extra[name] = str(value)
        elif dtype == "int":
            kwargs[name] = int(value)
        elif dtype == "Float":
            kwargs[name] = Decimal(value)
        elif dtype == "bool":
            kwargs[name] = value.lower() == "true"
        else:
            raise RuntimeError(f"Unexpected type annotation hit for {name}: {dtype}")
    return kwargs, extra


class _LatticeTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`Lattice`.

    Attributes
    ----------
    _filename : str
        Filename of grammar being transformed.
    """

    _filename: Optional[pathlib.Path]
    type_map: Dict[str, Type[BeamlineElement]] = {
        "UNDU": Undulator,
        "QUAD": Quadrupole,
        "DRIF": Drift,
        "CORR": Corrector,
        "CHIC": Chicane,
        "PHAS": PhaseShifter,
        "MARK": Marker,
        # "LINE": Line,
    }

    def __init__(self, filename: AnyPath) -> None:
        super().__init__()
        self._filename = pathlib.Path(filename)

    @lark.v_args(inline=True)
    def line(
        self,
        label: lark.Token,
        line: lark.Token,
        element_list: List[LineItem],
    ) -> Line:
        return Line(
            label=str(label),
            elements=element_list,
        )

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
    def beamline_element(
        self,
        label: lark.Token,
        type_: lark.Token,
        parameter_list: List[Tuple[str, lark.Token]],
    ) -> BeamlineElement:
        cls = self.type_map[type_.upper()[:4]]
        parameters, unknown = _fix_parameters(cls, dict(parameter_list))
        return cls(
            label=str(label),
            unknown_parameters=unknown,
            **parameters,
        )

    @lark.v_args(inline=True)
    def duplicate_item(
        self,
        count: lark.Token,
        label: lark.Token,
    ) -> DuplicatedLineItem:
        return DuplicatedLineItem(
            count=int(count),
            label=str(label),
        )

    @lark.v_args(inline=True)
    def positioned_item(
        self,
        label: lark.Token,
        position: lark.Token,
    ) -> PositionedLineItem:
        return PositionedLineItem(
            label=str(label),
            position=Decimal(position),
        )

    @lark.v_args(inline=True)
    def line_item(
        self, item: Union[lark.Token, DuplicatedLineItem, PositionedLineItem]
    ) -> BeamlineElement:
        if isinstance(item, lark.Token):
            return str(item)
        return item

    def element_list(
        self, items: List[Union[lark.Token, DuplicatedLineItem, PositionedLineItem]]
    ) -> List[Union[lark.Token, DuplicatedLineItem, PositionedLineItem]]:
        return items

    def lattice(self, elements: List[Union[BeamlineElement, Line]]) -> Lattice:
        return Lattice(
            elements,
            filename=self._filename,
        )
