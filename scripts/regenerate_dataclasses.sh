#!!/bin/bash

set -xe

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../" &>/dev/null && pwd)

if [ -z "$GENESIS4"]; then
  set +x
  echo "Please set GENESIS4 to the root directory of Genesis4:"
  echo '$ git clone https://github.com/svenreiche/Genesis-1.3-Version4 genesis4_src'
  echo '$ export GENESIS4=$PWD/genesis4_src'
  exit 1
fi

MAIN_INPUT_MD="${REPO_ROOT}/genesis/tests/genesis4/main_input.md"
LATTICE_MD="${REPO_ROOT}/genesis/tests/genesis4/lattice.md"

cp -f "$GENESIS4"/manual/LATTICE.md "$LATTICE_MD"
cp -f "$GENESIS4"/manual/MAIN_INPUT.md "$MAIN_INPUT_MD"

LATTICE_PY="${REPO_ROOT}/genesis/version4/input/_lattice.py"
MAIN_INPUT_PY="${REPO_ROOT}/genesis/version4/input/_main.py"

# git checkout -- "${LATTICE_PY}" "${MAIN_INPUT_PY}"
lattice_src=$(python genesis/version4/input/manual.py "$LATTICE_MD")

main_input_src=$(python genesis/version4/input/manual.py "$MAIN_INPUT_MD")
echo "$lattice_src" >"${LATTICE_PY}"
echo "$main_input_src" >"${MAIN_INPUT_PY}"

# Reformat the source code
ruff format "${LATTICE_PY}" "${MAIN_INPUT_PY}"
# And sort/fix imports
ruff check --fix --select I "${LATTICE_PY}" "${MAIN_INPUT_PY}"
