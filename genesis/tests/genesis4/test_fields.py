import numpy as np
import pathlib

import pytest

from ...version4 import Genesis4
from ...version4.input.core import NoSuchNamelistError
from ...version4.output import FieldFile
from ...version4.types import FieldFileParams
from ..conftest import genesis4_example2_path


def test_example2_field_reuse(tmp_path: pathlib.Path) -> None:
    G = Genesis4(genesis4_example2_path / "Example2.in")
    output = G.run(raise_on_error=True)
    key, *_ = output.load_fields()
    loaded_from_genesis = output.fields[key]

    field_filename = tmp_path / "dump.fld.h5"
    loaded_from_genesis.write_genesis4(field_filename)
    round_tripped = FieldFile.from_file(field_filename)

    assert round_tripped == loaded_from_genesis


def test_set_clear_initial_field() -> None:
    G = Genesis4(genesis4_example2_path / "Example2.in")

    with pytest.raises(NoSuchNamelistError):
        G.input.main.import_field

    # Check the top-level Genesis4 alias
    G.initial_field = FieldFile(dfl=np.zeros((4, 4, 2)), param=FieldFileParams())

    expected_filename = "initial_field.h5"
    assert G.input.main.import_field.file == expected_filename

    G.initial_field = None
    with pytest.raises(NoSuchNamelistError):
        G.input.main.import_field

    # Check the input property Genesis4.input helper
    G.input.initial_field = FieldFile(dfl=np.zeros((4, 4, 2)), param=FieldFileParams())

    expected_filename = "initial_field.h5"
    assert G.input.main.import_field.file == expected_filename

    G.input.initial_field = None
    with pytest.raises(NoSuchNamelistError):
        G.input.main.import_field


def test_initial_field_archive(tmp_path: pathlib.Path) -> None:
    G = Genesis4(
        genesis4_example2_path / "Example2.in",
        initial_field=FieldFile(
            dfl=np.zeros((4, 4, 2)),
            param=FieldFileParams(
                gridpoints=10,
                slicecount=11,
            ),
        ),
    )

    expected_filename = "initial_field.h5"
    assert G.input.main.import_field.file == expected_filename

    archive_file = tmp_path / "archive.h5"
    G.archive(archive_file)
    loaded = Genesis4.from_archive(archive_file)
    # assert G == loaded
    # Explicitly check our alias:
    assert G.initial_field == loaded.initial_field
    assert G.initial_field == loaded.input.initial_field


def test_set_initial_field(tmp_path: pathlib.Path) -> None:
    main_file = genesis4_example2_path / "Example2.in"
    lattice_file = genesis4_example2_path / "Example2.lat"
    G = Genesis4(main_file, lattice_file)
    for track in G.input.main.tracks:
        track.zstop = 0.1

    output = G.run(raise_on_error=True)
    field = output.load_field_by_key(list(output.field_files)[0])

    G = Genesis4(
        main_file,
        lattice_file,
        initial_field=field,
        workdir=tmp_path,
        use_temp_dir=False,
    )
    expected_filename = "initial_field.h5"
    assert G.input.main.import_field.file == expected_filename
    output = G.run(raise_on_error=True)

    field_file = tmp_path / expected_filename
    assert field_file.exists()

    loaded_field = FieldFile.from_file(field_file)
    loaded_field.label = "dump"
    assert loaded_field == field

    G.input.initial_field = None
    with pytest.raises(NoSuchNamelistError):
        G.input.main.import_field
