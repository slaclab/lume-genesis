import pathlib
from typing import Optional

from ...version4 import MainInput, Lattice, Genesis4, Genesis4Input


def run_with_instances(
    main_input: MainInput,
    lattice: Lattice,
    timeout: Optional[float] = None,
) -> Genesis4:
    input = Genesis4Input(
        main=main_input,
        lattice=lattice,
    )
    genesis = Genesis4(input, timeout=timeout)
    output = genesis.run(raise_on_error=True)
    assert output.run.success
    return genesis


def run_with_source(
    lattice: Lattice,
    main_input: MainInput,
    workdir: pathlib.Path,
    source_path: pathlib.Path,
):
    wrote_files = main_input.write_files(workdir, source_path=source_path)
    genesis = Genesis4(
        input=main_input.to_genesis(),
        lattice_source=lattice.to_genesis(),
        workdir=workdir,
        use_temp_dir=False,
    )
    for filename in wrote_files:
        print("Wrote:", filename)
    print("Files in workdir:")
    for filename in workdir.iterdir():
        print(filename)
        if filename.is_symlink():
            print(" ->", filename.readlink())
    output = genesis.run(raise_on_error=True)
    assert output.run.success
