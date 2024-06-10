import logging
import pathlib

import matplotlib as mpl

mpl.use("Agg")

test_root = pathlib.Path(__file__).resolve().parent
repo_root = test_root.parent.parent
docs_root = repo_root / "docs"
docs_examples = docs_root / "examples"
genesis2_examples = docs_examples / "genesis2"
genesis4_examples = docs_examples / "genesis4"

genesis4_example1_path = genesis4_examples / "data" / "example1-steadystate"
genesis4_example2_path = genesis4_examples / "data" / "example2-dumps"

test_artifacts = test_root / "artifacts"
test_artifacts.mkdir(exist_ok=True)

logging.getLogger("matplotlib").setLevel("INFO")
