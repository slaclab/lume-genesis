import pathlib

from ...version4 import Genesis4
from ...version4.output import FieldFile
from ..conftest import genesis4_example2_path


def test_example2(tmp_path: pathlib.Path) -> None:
    G = Genesis4(genesis4_example2_path / "Example2.in")
    output = G.run(raise_on_error=True)
    key, *_ = output.load_fields()
    loaded_from_genesis = output.fields[key]

    field_filename = tmp_path / "dump.fld.h5"
    loaded_from_genesis.write_genesis4(field_filename)
    round_tripped = FieldFile.from_file(field_filename)

    assert round_tripped == loaded_from_genesis
