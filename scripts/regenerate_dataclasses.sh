#!!/bin/bash

set -xe

REPO_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../" &> /dev/null && pwd )

LATTICE_PY="${REPO_ROOT}/genesis/version4/input/generated_lattice.py"
MAIN_INPUT_PY="${REPO_ROOT}/genesis/version4/input/generated_main.py"

# git checkout -- "${LATTICE_PY}" "${MAIN_INPUT_PY}"
lattice=$(python -m genesis.version4.input.manual "${REPO_ROOT}/genesis/tests/genesis4/lattice.md")

main_input=$(python -m genesis.version4.input.manual "${REPO_ROOT}/genesis/tests/genesis4/main_input.md")
echo "$lattice" > "${LATTICE_PY}"
echo "$main_input" > "${MAIN_INPUT_PY}"
