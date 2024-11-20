from ..input import Quadrupole, Corrector, Drift, Marker, Undulator
from ..input import Setup, Field, Track, Beam

from pmd_beamphysics.units import mec2

from scipy.constants import c

from math import pi, sqrt
import numpy as np

def ele_info(tao, ele_id):
    info = tao.ele_head(ele_id)
    info.update(tao.ele_gen_attribs(ele_id))
    info.update(tao.ele_methods(ele_id))
    info["key"] = info["key"].lower()
    return info


def label_from_bmad_name(bmad_name: str) -> str:
    """
    Formats a label by standardizing case, removing backslashes, and replacing disallowed characters.

    For superimposed elements with a composite name with '\', 
    the last name will be used.

    TODO: better lord-slave logic

    Parameters
    ----------
    bmad_name : str
        The Bmad name string to be formatted.

    Returns
    -------
    str
        A formatted label suitable for Genesis4.
    """
    name = bmad_name.upper()
    if "\\" in name and name.split("\\")[-1].isnumeric():
        raise NotImplementedError("Multipass elements not supported")
    
    label = name.split("\\")[-1]  # Extracts text after backslash, if any
    return label.replace(".", "_").replace("#", "_")


def quadrupole_and_corrector_steps(quad: Quadrupole, cx=0, cy=0, num_steps=1):
    """
    Converts a Genesis4 Quadrupole into a series of quadrupole and corrector steps.

    Parameters
    ----------
    quad : Quadrupole
        The Quadrupole element to be divided into steps.
    cx : float, optional
        The horizontal corrector strength. Defaults to 0.
    cy : float, optional
        The vertical corrector strength. Defaults to 0.
    num_steps : int, optional
        The number of steps to divide the quadrupole into. Defaults to 1.

    Returns
    -------
    list
        A list of Genesis4 elements, alternating between Corrector and Quadrupole steps.
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
    Creates Genesis4 elements from a specified element in a pytao.Tao instance.

    TODO: recogonize "chicane" bend patterns. 

    TODO: desplit elements

    Parameters
    ----------
    tao : Tao
        The Tao instance containing the element definitions.
    ele_id : int or str
        The identifier (name or index) of the element in the Tao instance.

    Returns
    -------
    list
        A list of Genesis4 elements created from the specified Tao element.

    Raises
    ------
    NotImplementedError
        If the element type or offset is not supported.
    ValueError
        If the undulator length is inconsistent with the number of periods and period length.
    """

    info = ele_info(tao, ele_id)
    key = info["key"].lower()
    
    # Make Genesis4 label
    name = info["name"] # Original Bmad name. We need to clean this.
    label = label_from_bmad_name(name)

    L = info.get("L", 0)
    x_offset = info.get("X_OFFSET", 0)
    y_offset = info.get("Y_OFFSET", 0)

    if key == "beginning_ele":
        eles = []

    elif key in ("hkicker", "vkicker", "kicker"):
        if L == 0:
            L = 1e-9 # Avoid bug with zero length corrector
            
        if key == "hkicker":
            cx = info["KICK"]
            cy = 0
        elif key == "vkicker":
            cx = 0
            cy = info["KICK"]
        else:
            cx = info["HKICK"]
            cy = info["VKICK"]
        ele = Corrector(cx=cx, cy=cy, label=label, L=L) 
        eles = [ele]
        
    elif key in ("drift", "ecollimator", "pipe"):
        ele = Drift(L=L, label=label)
        eles = [ele]

    elif key in ("lcavity", "rfcavity"):
        if info["GRADIENT"] == 0:
            eles = [Drift(L=L, label=label)]
        else:
            raise NotImplementedError(f"{key} '{name}' with nonzero gradient")            
    elif key in ("sbend", ):
        if info["G"] == 0:
            eles = [Drift(L=L, label=label)]
        else:
            raise NotImplementedError(f"{key} '{name}' with nonzero G")
        
    elif key in ("instrument", "marker", "monitor"):
        if L ==0:
            ele = Marker(label=label)
        else:
            ele = Drift(L=L, label=label)
        eles = [ele]
    elif key == "quadrupole":
        cx = info["HKICK"]
        cy = info["VKICK"]

        ele = Quadrupole(
            L=L,
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
        raise NotImplementedError(f"{label}: {key} with {L=}")

    return eles


def genesis4_elements_and_line_from_tao(tao, ele_start='beginning', ele_end='end', universe=1, branch=0):
    """
    Creates a Genesis4 lattice from a pytao.Tao instance.

    Parameters
    ----------
    tao : Tao
        The Tao instance to extract elements from.
    ele_start : str, optional
        Element to start. Defaults to "beginning".
    ele_end : str, optional
        Element to end. Defaults to "end".        
    branch : int, optional
        The branch index within the specified Tao universe. Defaults to 0.
    universe : int, optional
        The universe index within the Tao object. Defaults to 1.        

    Returns
    -------
    dict
        A dictionary mapping element labels to Genesis4 element objects.
    list of str
        A list of element labels in the order of the lattice line.

    Raises
    ------
    ValueError
        If elements with the same label have different properties.
    """

    elements = {}
    line_labels = []
    for ix_ele in tao.lat_list(f"{ele_start}:{ele_end}", "ele.ix_ele", ix_uni=universe, ix_branch=branch):
        eles = genesis4_eles_from_tao_ele(tao, ix_ele)

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
    tao, ele_start: str = "beginning", branch: int = 0, universe: int = 1,
    
):
    """
    Creates Genesis4 namelists from a Tao instance, using specified parameters to
    extract relevant beamline, beam, and field configurations.

    Parameters
    ----------
    tao : pytao.Tao
        A running Tao instance, providing access to element attributes, Twiss parameters,
        and orbit data.
    ele_start : str, optional
        The starting element within the specified Tao universe and branch, using the
        syntax `@` for universe and `>>` for branch (e.g., `'1@0>>element_name'`).
        Defaults to `'beginning'`.
    branch : int, optional
        The branch index within the specified Tao universe. Defaults to 0.
    universe : int, optional
        The universe index within the Tao object. Defaults to 1.

    Returns
    -------
    list
        A list of Genesis4 namelists for the setup, field, beam, and tracking configurations.

    Notes
    -----
    - This function collects attributes, Twiss parameters, and orbit data from the
      specified Tao element to construct Genesis4-compatible input data.
    - The generated `beamline` name is based on the Tao universe and branch configuration,
      with `gamma0` calculated from the total energy.
    - Additional parameters for field and beam properties are hard-coded but may be generalized.

    Examples
    --------
    >>> tao = pytao.Tao(...)
    >>> namelists = genesis4_namelists_from_tao(tao, ele_start='beginning', branch=0, universe=1)

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
