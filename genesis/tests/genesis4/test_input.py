from ...version4.input import Setup


def test_namelist_output():
    setup = Setup(
        rootname="Benchmark",
        lattice="Aramis.lat",
        beamline="ARAMIS",
        lambda0=1e-10,
        gamma0=11357.82,
        delz=0.045,
        shotnoise=0,
        beam_global_stat=True,
        field_global_stat=True,
    )
    assert (
        str(setup)
        == """
&setup
  rootname = Benchmark
  lattice = Aramis.lat
  beamline = ARAMIS
  gamma0 = 11357.82
  delz = 0.045
  shotnoise = 0
  beam_global_stat = true
  field_global_stat = true
&end
""".strip()
    )
