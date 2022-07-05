from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import genesis2_dpa_to_data


def final_particles(genesis_object):
    """
    Parameters
    ----------
    genesis_object: Genesis object with parsed dpa data.

    Returns
    -------
    particles: ParticleGroup object

    """
    g = genesis_object  # convenience
    param = g.output["param"]
    dpa = g.output["data"]["dpa"]
    current = g.output["data"]["current"]
    data = genesis2_dpa_to_data(
        dpa, xlamds=param["xlamds"], zsep=param["zsep"], current=current
    )

    return ParticleGroup(data=data)
