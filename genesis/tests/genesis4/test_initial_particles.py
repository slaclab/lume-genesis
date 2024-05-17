from typing import Optional
import numpy as np
import pytest

from pmd_beamphysics import ParticleGroup

from ...version4 import MainInput


def gaussian_data(
    n_particle: int = 100,
    charge: float = 1e-9,
    p0: float = 1e9,
    mean: Optional[np.ndarray] = None,
    sigma_mat: Optional[np.ndarray] = None,
):
    """
    Makes Gaussian particle data from a Bmad-style sigma matrix.

    Parameters
    ----------
    n_particle: int, default=100
        Number of particles.
    charge : float, default=1e-9
        Charge in C.
    p0 : float
        Reference momentum in eV/c
    mean : np.ndarray of shape (6,), optional
        Mean positions. Default = None gives zeros
    sigma_mat : np.ndarray of shape (6,6), optional
        Sigma matrix in Bmad units. If default, this is the identity * 1e-3

    Returns
    -------
    dict
        ParticleGroup-compatible data dictionary:
        >>> ParticleGroup(data=gaussian_data())
    """
    if mean is None:
        mean = np.zeros(6)

    if sigma_mat is None:
        cov = np.eye(6) * 1e-3
    else:
        cov = sigma_mat

    dat = np.random.multivariate_normal(mean, cov, size=n_particle)
    x = dat[:, 0]
    px = dat[:, 1]
    y = dat[:, 2]
    py = dat[:, 3]
    z = dat[:, 4]
    pz = dat[:, 5]

    data = {
        "x": x,
        "px": px * p0,
        "y": y,
        "py": py,
        "z": z,
        "pz": (1 + pz) * p0,
        "t": np.zeros(n_particle),
        "weight": charge / n_particle,
        "status": np.ones(n_particle),
        "species": "electron",
    }

    return data


def test_set_particles(main_input: MainInput) -> None:
    with pytest.raises(ValueError):
        main_input.initial_particles

    main_input.time.slen = 0.0
    group = ParticleGroup(data=gaussian_data())
    main_input.initial_particles = group
    assert main_input.initial_particles.particles == group
    assert main_input.time.slen != 0.0
