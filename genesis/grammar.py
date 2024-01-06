from __future__ import annotations
import dataclasses
import lark
import pathlib
from decimal import Decimal

from typing import Dict, List, Optional, Tuple, Type, Union


LATTICE_GRAMMAR = "lattice.lark"
AnyPath = Union[pathlib.Path, str]
Float = Union[Decimal, float]


def new_parser(filename: str, **kwargs) -> lark.Lark:
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
    _renames_ = {
        "l": "length",
    }
    label: str
    unknown_parameters: dict[str, ValueType] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Undulator(BeamlineElement):
    #: The dimensionless rms undulator parameter. For planar undulator this
    #: value is smaller by a factor $1 / \sqrt{2}$ than its K-value, while for
    #: helical undulator rms and peak values are identical.
    aw: Float = 0.0
    #: Undulator period length in meter. Default is 0 m.
    lambdau: Float = 0.0
    #: Number of periods.
    nwig: int = 0
    #: Boolean flag whether the undulator is planar or helical. A planar
    #: undulator has helical=`false`. Note that setting it to `true`, does not
    #: change the roll-off parameters for focusing. To be consistent they have to
    #: be set directly.
    helical: bool = False
    #: Roll-off parameter of the quadratic term of the undulator field in x. It
    #: is normalized with respect to $k_u^2$.
    kx: Float = 0.0
    #: Roll-off parameter of the quadratic term of the undulator field in y.
    ky: Float = 1.0
    #: Offset of the undulator module in $x$ in meter.
    ax: Float = 0.0
    #: Offset of the undulator module in $y$ in meter.
    ay: Float = 0.0
    #: Relative transverse gradient of undulator field in $x$ $\equiv (1/a_w)
    #: \partial a_w/\partial x$.
    gradx: Float = 0
    #: Relative transverse gradient of undulator field in $y$ $\equiv (1/a_w)
    #: \partial a_w/\partial y$.
    grady: Float = 0


@dataclasses.dataclass
class Quadrupole(BeamlineElement):
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
    #: Length of the drift in meter.
    length: Float = 0.0


@dataclasses.dataclass
class Corrector(BeamlineElement):
    #: Length of the corrector in meter.
    length: Float = 0
    #: angle in $x$ in units of $\gamma \beta_x$.
    cx: Float = 0
    #: angle in $y$ in units of $\gamma \beta_y$.
    cy: Float = 0


@dataclasses.dataclass
class Chicane(BeamlineElement):
    #: Length of the chicane, which consists out of 4 dipoles without focusing.
    #: The first and last are placed at the beginning and end of the reserved
    #: space. The inner ones are defined by the drift length in between. Any
    #: remaining distance, namely the length subtracted by 4 times the dipole
    #: length and twice the drift length are placed between the second and third
    #: dipole.
    length: Float = 0
    #: of an individual dipole in meter.
    lb: Float = 0
    #: between the outer and inner dipoles, projected onto the undulator axis.
    #: The actual path length is longer by the factor $1/\cos\theta$, where
    #: $\theta$ is the bending angle of an individual dipole.
    ld: Float = 0
    #: length difference between the straight path and the actual trajectory in
    #: meters. Genesis 1.3 calculates the bending angle internally starting from
    #: this value. $R_{56} = 2$`delay`.
    delay: Float = 0


@dataclasses.dataclass
class PhaseShifter(BeamlineElement):
    #: Length of the phase shifter in meter.
    length: Float = 0
    #: in the ponderomotive phase of the electrons in units of rad. Note that
    #: Genesis 1.3 is doing an autophasing, so that the electrons at reference
    #: energy are not changing in ponderomotive phase in drifts.
    phi: Float = 0


@dataclasses.dataclass
class Marker(BeamlineElement):
    #: A non-zero value enforces the dump of the field distribution of this
    #: zero length element.
    dumpfield: int = 0
    #: non-zero value enforces the dump of the particle distribution.
    dumpbeam: int = 0
    #: non-zero value enforces the sorting of particles, if one-for-one
    #: simulations are enabled.
    sort: int = 0
    #: non-zero value stops the execution of the tracking module. Note that the
    #: output file still contains the full length with zeros as output for those
    #: integration steps which are no further calculated.
    stop: int = 0


@dataclasses.dataclass
class DuplicatedLineItem:
    #: Duplication count.
    count: int
    #: Element name.
    label: str


@dataclasses.dataclass
class PositionedLineItem:
    #: Position in meters.
    position: Float
    #: Element name.
    label: str


LineItem = Union[str, DuplicatedLineItem, PositionedLineItem]


@dataclasses.dataclass
class Line:
    label: str
    elements: List[LineItem] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Lattice:
    elements: List[Union[BeamlineElement, Line]] = dataclasses.field(
        default_factory=list
    )
    filename: Optional[pathlib.Path] = None

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
        name = cls._renames_.get(name, name)
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
