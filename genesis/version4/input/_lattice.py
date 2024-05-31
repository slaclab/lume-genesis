#!/usr/bin/env python
# vi: syntax=python sw=4 ts=4 sts=4
"""
This file is auto-generated by lume-genesis (`genesis.version4.input.manual`).

Do not hand-edit it.
"""

from __future__ import annotations

from typing import Literal, Union

import pydantic

from .. import types


class Undulator(types.BeamlineElement):
    r"""
    Lattice beamline element: an undulator.

    Undulator corresponds to Genesis 4 beamlineelement `undulator`.

    Attributes
    ----------
    aw : float, default=0.0
        The dimensionless rms undulator parameter. For planar undulator this value is
        smaller by a factor $1 / \sqrt{2}$ than its K-value, while for helical
        undulator rms and peak values are identical.
    lambdau : float, default=0.0
        Undulator period length in meter. Default is 0 m.
    nwig : int, default=0
        Number of periods.
    helical : bool, default=False
        Boolean flag whether the undulator is planar or helical. A planar undulator has
        helical=`false`. Note that setting it to `true`, does not change the roll-off
        parameters for focusing. To be consistent they have to be set directly.
    kx : float, default=0.0
        Roll-off parameter of the quadratic term of the undulator field in x. It is
        normalized with respect to $k_u^2$.
    ky : float, default=1.0
        Roll-off parameter of the quadratic term of the undulator field in y.
    ax : float, default=0.0
        Offset of the undulator module in $x$ in meter.
    ay : float, default=0.0
        Offset of the undulator module in $y$ in meter.
    gradx : float, default=0.0
        Relative transverse gradient of undulator field in $x$ $\equiv (1/a_w) \partial
        a_w/\partial x$.
    grady : float, default=0.0
        Relative transverse gradient of undulator field in $y$ $\equiv (1/a_w) \partial
        a_w/\partial y$.
    """

    type: Literal["undulator"] = "undulator"
    aw: float = pydantic.Field(
        default=0.0,
        description=(
            "The dimensionless rms undulator parameter. For planar undulator this value "
            r"is smaller by a factor $1 / \sqrt{2}$ than its K-value, while for helical "
            "undulator rms and peak values are identical."
        ),
    )
    lambdau: float = pydantic.Field(
        default=0.0,
        description="Undulator period length in meter. Default is 0 m.",
    )
    nwig: int = pydantic.Field(
        default=0,
        description="Number of periods.",
    )
    helical: bool = pydantic.Field(
        default=False,
        description=(
            "Boolean flag whether the undulator is planar or helical. A planar "
            "undulator has helical=`false`. Note that setting it to `true`, does not "
            "change the roll-off parameters for focusing. To be consistent they have to "
            "be set directly."
        ),
    )
    kx: float = pydantic.Field(
        default=0.0,
        description=(
            "Roll-off parameter of the quadratic term of the undulator field in x. It "
            "is normalized with respect to $k_u^2$."
        ),
    )
    ky: float = pydantic.Field(
        default=1.0,
        description="Roll-off parameter of the quadratic term of the undulator field in y.",
    )
    ax: float = pydantic.Field(
        default=0.0,
        description="Offset of the undulator module in $x$ in meter.",
    )
    ay: float = pydantic.Field(
        default=0.0,
        description="Offset of the undulator module in $y$ in meter.",
    )
    gradx: float = pydantic.Field(
        default=0.0,
        description=(
            r"Relative transverse gradient of undulator field in $x$ $\equiv (1/a_w) "
            r"\partial a_w/\partial x$."
        ),
    )
    grady: float = pydantic.Field(
        default=0.0,
        description=(
            r"Relative transverse gradient of undulator field in $y$ $\equiv (1/a_w) "
            r"\partial a_w/\partial y$."
        ),
    )
    label: str = ""


class Drift(types.BeamlineElement):
    r"""
    Lattice beamline element: drift.

    Drift corresponds to Genesis 4 beamlineelement `drift`.

    Attributes
    ----------
    L : float, default=0.0
        Length of the drift in meter.
    """

    type: Literal["drift"] = "drift"
    L: float = pydantic.Field(
        default=0.0,
        description="Length of the drift in meter.",
        validation_alias=pydantic.AliasChoices("L", "l"),
        serialization_alias="l",
    )
    label: str = ""


class Quadrupole(types.BeamlineElement):
    r"""
    Lattice beamline element: quadrupole.

    Quadrupole corresponds to Genesis 4 beamlineelement `quadrupole`.

    Attributes
    ----------
    L : float, default=0.0
        Length of the quadrupole in meter.
    k1 : float, default=0.0
        Normalized focusing strength in 1/m^2.
    x_offset : float, default=0.0
        Offset in $x$ in meter.
    y_offset : float, default=0.0
        Offset in $y$ in meter.
    """

    type: Literal["quadrupole"] = "quadrupole"
    L: float = pydantic.Field(
        default=0.0,
        description="Length of the quadrupole in meter.",
        validation_alias=pydantic.AliasChoices("L", "l"),
        serialization_alias="l",
    )
    k1: float = pydantic.Field(
        default=0.0,
        description="Normalized focusing strength in 1/m^2.",
    )
    x_offset: float = pydantic.Field(
        default=0.0,
        description="Offset in $x$ in meter.",
        validation_alias=pydantic.AliasChoices("x_offset", "dx"),
        serialization_alias="dx",
    )
    y_offset: float = pydantic.Field(
        default=0.0,
        description="Offset in $y$ in meter.",
        validation_alias=pydantic.AliasChoices("y_offset", "dy"),
        serialization_alias="dy",
    )
    label: str = ""


class Corrector(types.BeamlineElement):
    r"""
    Lattice beamline element: corrector.

    Corrector corresponds to Genesis 4 beamlineelement `corrector`.

    Attributes
    ----------
    L : float, default=0.0
        Length of the corrector in meter.
    cx : float, default=0.0
        Kick angle in $x$ in units of $\gamma \beta_x$.
    cy : float, default=0.0
        Kick angle in $y$ in units of $\gamma \beta_y$.
    """

    type: Literal["corrector"] = "corrector"
    L: float = pydantic.Field(
        default=0.0,
        description="Length of the corrector in meter.",
        validation_alias=pydantic.AliasChoices("L", "l"),
        serialization_alias="l",
    )
    cx: float = pydantic.Field(
        default=0.0,
        description=r"Kick angle in $x$ in units of $\gamma \beta_x$.",
    )
    cy: float = pydantic.Field(
        default=0.0,
        description=r"Kick angle in $y$ in units of $\gamma \beta_y$.",
    )
    label: str = ""


class Chicane(types.BeamlineElement):
    r"""
    Lattice beamline element: chicane.

    Chicane corresponds to Genesis 4 beamlineelement `chicane`.

    Attributes
    ----------
    L : float, default=0.0
        Length of the chicane, which consists out of 4 dipoles without focusing. The
        first and last are placed at the beginning and end of the reserved space. The
        inner ones are defined by the drift length in between. Any remaining distance,
        namely the length subtracted by 4 times the dipole length and twice the drift
        length are placed between the second and third dipole.
    lb : float, default=0.0
        Length of an individual dipole in meter.
    ld : float, default=0.0
        Drift between the outer and inner dipoles, projected onto the undulator axis.
        The actual path length is longer by the factor $1/\cos\theta$, where $\theta$
        is the bending angle of an individual dipole.
    delay : float, default=0.0
        Path length difference between the straight path and the actual trajectory in
        meters. Genesis 1.3 calculates the bending angle internally starting from this
        value. $R_{56} = 2$`delay`.
    """

    type: Literal["chicane"] = "chicane"
    L: float = pydantic.Field(
        default=0.0,
        description=(
            "Length of the chicane, which consists out of 4 dipoles without focusing. "
            "The first and last are placed at the beginning and end of the reserved "
            "space. The inner ones are defined by the drift length in between. Any "
            "remaining distance, namely the length subtracted by 4 times the dipole "
            "length and twice the drift length are placed between the second and third "
            "dipole."
        ),
        validation_alias=pydantic.AliasChoices("L", "l"),
        serialization_alias="l",
    )
    lb: float = pydantic.Field(
        default=0.0,
        description="Length of an individual dipole in meter.",
    )
    ld: float = pydantic.Field(
        default=0.0,
        description=(
            "Drift between the outer and inner dipoles, projected onto the undulator "
            r"axis. The actual path length is longer by the factor $1/\cos\theta$, where "
            r"$\theta$ is the bending angle of an individual dipole."
        ),
    )
    delay: float = pydantic.Field(
        default=0.0,
        description=(
            "Path length difference between the straight path and the actual trajectory "
            "in meters. Genesis 1.3 calculates the bending angle internally starting "
            "from this value. $R_{56} = 2$`delay`."
        ),
    )
    label: str = ""


class PhaseShifter(types.BeamlineElement):
    r"""
    Lattice beamline element: phase shifter.

    PhaseShifter corresponds to Genesis 4 beamlineelement `phaseshifter`.

    Attributes
    ----------
    L : float, default=0.0
        Length of the phase shifter in meter.
    phi : float, default=0.0
        Change in the ponderomotive phase of the electrons in units of rad. Note that
        Genesis 1.3 is doing an autophasing, so that the electrons at reference energy
        are not changing in ponderomotive phase in drifts.
    """

    type: Literal["phaseshifter"] = "phaseshifter"
    L: float = pydantic.Field(
        default=0.0,
        description="Length of the phase shifter in meter.",
        validation_alias=pydantic.AliasChoices("L", "l"),
        serialization_alias="l",
    )
    phi: float = pydantic.Field(
        default=0.0,
        description=(
            "Change in the ponderomotive phase of the electrons in units of rad. Note "
            "that Genesis 1.3 is doing an autophasing, so that the electrons at "
            "reference energy are not changing in ponderomotive phase in drifts."
        ),
    )
    label: str = ""


class Marker(types.BeamlineElement):
    r"""
    Lattice beamline element: marker.

    Marker corresponds to Genesis 4 beamlineelement `marker`.

    Attributes
    ----------
    dumpfield : int, default=0
        A non-zero value enforces the dump of the field distribution of this zero
        length element.
    dumpbeam : int, default=0
        A non-zero value enforces the dump of the particle distribution.
    sort : int, default=0
        A non-zero value enforces the sorting of particles, if one-for-one simulations
        are enabled.
    stop : int, default=0
        A non-zero value stops the execution of the tracking module. Note that the
        output file still contains the full length with zeros as output for those
        integration steps which are no further calculated.
    """

    type: Literal["marker"] = "marker"
    dumpfield: int = pydantic.Field(
        default=0,
        description=(
            "A non-zero value enforces the dump of the field distribution of this zero "
            "length element."
        ),
    )
    dumpbeam: int = pydantic.Field(
        default=0,
        description="A non-zero value enforces the dump of the particle distribution.",
    )
    sort: int = pydantic.Field(
        default=0,
        description=(
            "A non-zero value enforces the sorting of particles, if one-for-one "
            "simulations are enabled."
        ),
    )
    stop: int = pydantic.Field(
        default=0,
        description=(
            "A non-zero value stops the execution of the tracking module. Note that the "
            "output file still contains the full length with zeros as output for those "
            "integration steps which are no further calculated."
        ),
    )
    label: str = ""


AutogeneratedBeamlineElement = Union[
    Undulator,
    Drift,
    Quadrupole,
    Corrector,
    Chicane,
    PhaseShifter,
    Marker,
]
