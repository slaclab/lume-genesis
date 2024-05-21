import pathlib
from typing import Optional

import pydantic
import numpy as np

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
        lattice=lattice.to_genesis(),
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


def compare(obj, expected, history=()):
    print("Comparing:", history, type(obj).__name__)
    assert isinstance(obj, type(expected))
    # assert repr(obj) == repr(expected)
    if isinstance(obj, pydantic.BaseModel):
        for attr, fld in obj.model_fields.items():
            value = getattr(obj, attr)
            if isinstance(value, np.ndarray):
                assert fld.annotation is np.ndarray

            compare(
                getattr(obj, attr),
                getattr(expected, attr),
                history=history + (attr,),
            )
    elif isinstance(obj, dict):
        assert set(obj) == set(expected)
        for key in obj:
            compare(
                obj[key],
                expected[key],
                history=history + (key,),
            )
    elif isinstance(obj, (list, tuple)):
        assert len(obj) == len(expected)
        for idx, (value, value_expected) in enumerate(zip(obj, expected)):
            compare(
                value,
                value_expected,
                history=history + (idx,),
            )
    elif isinstance(obj, (np.ndarray, float)):
        assert np.allclose(obj, expected)
    else:
        assert obj == expected
