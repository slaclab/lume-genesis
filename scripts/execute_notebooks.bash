#!/bin/bash
set -e

cd docs/examples/genesis4 || exit 1

NOTEBOOKS=$(git ls-files "*.ipynb")

SKIP_PATTERNS=("perlmutter" "fodo")

# Silence Jupyterlab warning
export PYDEVD_DISABLE_FILE_VALIDATION=1

for file in $NOTEBOOKS
do
    should_skip=false
    for SKIP in "${SKIP_PATTERNS[@]}"; do
        if [[ "$file" == *"$SKIP"* ]]; then
           should_skip=true
           break
       fi
    done

    if [ "$should_skip" = true ]; then
        echo "Skipping: $file"
    else
        echo "Processing: $file"
        pushd "$(dirname "$file")" || exit
        jupyter nbconvert --to notebook --execute "$(basename "$file")" --inplace
        popd || exit
    fi
done
