#!!/bin/bash

set -xe

REPO_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../" &> /dev/null && pwd )

LATTICE_PY="${REPO_ROOT}/genesis/version4/input/_lattice.py"
MAIN_INPUT_PY="${REPO_ROOT}/genesis/version4/input/_main.py"

# git checkout -- "${LATTICE_PY}" "${MAIN_INPUT_PY}"
lattice=$(python genesis/version4/input/manual.py "${REPO_ROOT}/genesis/tests/genesis4/lattice.md")

main_input=$(python genesis/version4/input/manual.py "${REPO_ROOT}/genesis/tests/genesis4/main_input.md")
echo "$lattice" > "${LATTICE_PY}"
echo "$main_input" > "${MAIN_INPUT_PY}"

# Reformat the source code
ruff format "${LATTICE_PY}" "${MAIN_INPUT_PY}"
# And sort/fix imports
ruff check --fix --select I "${LATTICE_PY}" "${MAIN_INPUT_PY}"
