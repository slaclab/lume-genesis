import pathlib

import yaml

import genesis.version4 as g4

docs_root = pathlib.Path(__file__).resolve().parents[1] / "docs"
example = docs_root / "examples/genesis4/data/example1-steadystate"

G = g4.Genesis4(example / "Example1.in")
output = G.run()

hdf_summary = output.to_hdf_summary()

for info in hdf_summary.values():
    if info["units"]:
        info["units"] = str(info["units"])
    else:
        info.pop("units")
    info.pop("hdf_key")
    info.pop("python_attr")

with open("genesis4-output-summary.yaml", "w") as fp:
    yaml.dump(hdf_summary, fp)
