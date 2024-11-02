from ..input._lattice import Quadrupole, Corrector, Drift, Marker, Undulator
from ..input._main import Setup, Field, Track, Beam


from pmd_beamphysics.units import mec2

from scipy.constants import c

from math import pi, sqrt
import numpy as np


def label_from_bmad_name(bmad_name: str) -> str:
    """Format a label by standardizing case, removing backslashes,
    and replacing disallowed characters."""
    name = bmad_name.upper()
    label = name.split("\\")[-1]  # Extracts text after backslash, if any
    return label.replace(".", "_").replace("#", "_")


def quadrupole_and_corrector_steps(quad: Quadrupole, cx=0, cy=0, num_steps=1):
    """
    This converts a Genesis4 Quadrupole into quad with a corrector


    Returns
    -------
    eles: List of Genesis4 elements

    """
    L1 = quad.L / num_steps
    label = quad.label
    L_corrector = 1e-9  # BUG: zero length does nothing, but this works.
    cx1 = cx / num_steps
    cy1 = cy / num_steps
    eles = []
    # Interleave kicks and quad steps
    for step in range(num_steps):
        # First step is half strength
        if step == 0:
            cx = cx1 / 2
            cy = cy1 / 2
        else:
            cx = cx1
            cy = cy1

        eles.append(
            Corrector(cx=cx, cy=cy, label=f"{label}_kick{step+1}", L=L_corrector)
        )
        eles.append(
            Quadrupole(
                L=L1,
                k1=quad.k1,
                x_offset=quad.x_offset,
                y_offset=quad.y_offset,
                label=f"{label}_step{step+1}",
            )
        )

    # Final half step
    eles.append(
        Corrector(cx=cx1 / 2, cy=cy1 / 2, label=f"{label}_kick{step+2}", L=L_corrector)
    )
    return eles


def genesis4_eles_from_tao_ele(tao, ele_id):
    """
    Create a Genesis4 elements from a pytao.Tao instance.

    Note that multip

    This is intended to be used with
        `tao_create_genesis4_lattice`

    Parameters
    ----------
    tao: Tao object
        Tao instance to create the Genesis4 lattice from.

    ele_id: int or str
        Name or index of the element in the Tao instance

    Returns
    -------
    eles: List of Genesis4 elements

    """
    info = tao.ele_head(ele_id)
    info.update(tao.ele_gen_attribs(ele_id))
    info.update(tao.ele_methods(ele_id))
    key = info["key"].lower()

    # Genesis4 calls this 'label':
    label = label_from_bmad_name(info["name"])

    x_offset = info.get("X_OFFSET", 0)
    y_offset = info.get("Y_OFFSET", 0)

    if key == "beginning_ele":
        eles = None
    elif key in ("drift", "pipe"):
        ele = Drift(L=info["L"], label=label)
        eles = [ele]
    elif key in ("marker", "monitor") and (info["L"] == 0):
        ele = Marker(label=label)
        eles = [ele]
        eles = None  # TEMP
    elif key == "quadrupole":
        cx = info["HKICK"]
        cy = info["VKICK"]

        ele = Quadrupole(
            L=info["L"],
            k1=info["K1"],
            label=label,
            x_offset=x_offset,
            y_offset=y_offset,
        )

        # Do nothing if there are no correctors
        if cx == 0 and cy == 0:
            eles = [ele]
        else:
            eles = quadrupole_and_corrector_steps(
                ele, cx=cx, cy=cy, num_steps=info["NUM_STEPS"]
            )

    elif key == "wiggler":
        # Check for offsets
        if x_offset != 0:
            raise NotImplementedError(f"x_offset not zero: {x_offset}")
        if y_offset != 0:
            raise NotImplementedError(f"y_offset not zero: {y_offset}")

        # aw calc
        B0 = B0 = info["B_MAX"]
        lambdau = info["L_PERIOD"]
        K = B0 * lambdau * c / (2 * pi * mec2)

        L = info["L"]
        nwig = int(info["N_PERIOD"])
        lambdau = info["L_PERIOD"]
        if not np.isclose(L, nwig * lambdau):
            raise ValueError(
                f"Inconsistent length for undulator {label}: {L} != {nwig}*{lambdau}"
            )

        if "helical" in info["field_calc"].lower():
            helical = True
            aw = K
        else:
            assert info["field_calc"].lower() == "planar_model"
            aw = K / sqrt(2)
            helical = False

        ele = Undulator(
            nwig=nwig,
            lambdau=lambdau,
            aw=aw,
            helical=helical,
            label=label,
        )
        eles = [ele]
    else:
        raise NotImplementedError(key)

    return eles


def genesis4_elements_and_line_from_tao(tao, match="*"):
    """
    Create a Genesis4 lattice from a pytao.Tao instance.


    Parameters
    ----------
    tao: Tao object
        Tao instance to create the Genesis4 lattice from.

    Returns
    -------
    elements: dict
    line_labels: list of str

    """
    elements = {}
    line_labels = []
    for ix_ele in tao.lat_list(match, "ele.ix_ele"):
        eles = genesis4_eles_from_tao_ele(tao, ix_ele)
        if eles is not None:
            for ele in eles:
                label = ele.label
                ele.label = ""
                if label in elements:
                    ele0 = elements[label]
                    if ele0 != ele:
                        raise ValueError(
                            f"Elements have the same name but different properties: {ele0}, {ele}"
                        )
                else:
                    elements[label] = ele
                line_labels.append(label)

    return elements, line_labels


def genesis4_namelists_from_tao(
    tao, ele_start: str = "beginning", branch: int = 0, universe: int = 1
):
    """
    Creates a Genesis4 input configuration from a Tao object, incorporating specified
    starting element, branch, and universe to extract relevant beamline and beam parameters.

    Parameters
    ----------
    tao : pytao.Tao
        A running Tao instance
    ele_start : str, optional
        The starting element within the Tao universe and branch, specified with the
        `@` and `>>` syntax (e.g., `'1@0>>element_name'`). Defaults to `'beginning'`.
    branch : int, optional
        The branch index within the specified Tao universe. Defaults to 0.
    universe : int, optional
        The universe index within the Tao object. Defaults to 1.

    Returns
    -------
    MainInput
        A MainInput object containing the Genesis4 input setup with namelists for
        the setup, field, beam, and tracking.

    Notes
    -----
    - This function gathers element attributes, Twiss parameters, and orbit data
      from the specified Tao element.
    - The function constructs the beamline name, computes the normalized gamma value
      from the total energy, and sets up the Genesis4 input namelists accordingly.

    Examples
    --------
    >>> tao = Tao(...)
    >>> genesis_input = genesis4_input_from_tao(tao, ele_start='beginning', branch=0, universe=1)

    """

    # Handle Tao universe @ branch >> sytax
    if ">>" not in ele_start:
        ele_start = f"{universe}@{branch}>>{ele_start}"

    attrs = tao.ele_gen_attribs(ele_start)
    twiss = tao.ele_twiss(ele_start)
    orbit = tao.ele_orbit(ele_start)

    beamline = tao.branch1(universe, branch)["name"]
    gamma0 = attrs["E_TOT"] / mec2

    setup = Setup(
        rootname=beamline,
        # lattice='Example1.lat',
        beamline=beamline,
        gamma0=gamma0,
        # nbins=8,
        # shotnoise=False,
    )

    # TODO: Generalize
    field = Field(power=5000.0, waist_size=3e-05, dgrid=0.0002, ngrid=255)

    beam = Beam(
        alphax=twiss["alpha_a"],
        alphay=twiss["alpha_b"],
        betax=twiss["beta_a"],
        betay=twiss["beta_b"],
        xcenter=orbit["x"],
        ycenter=orbit["y"],
        pxcenter=orbit["px"] * gamma0,
        pycenter=orbit["py"] * gamma0,
        gamma=(1 + orbit["pz"]) * gamma0,
    )
    namelists = [setup, field, beam, Track()]

    return namelists
