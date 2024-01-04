#!/bin/bash
set -e

NOTEBOOKS=$(find docs/examples/genesis4/ -type f -name "*.ipynb" -not -path '*/.*')

SKIP_PATTERNS=("perlmutter" "fodo")  

#echo $NOTEBOOKS

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
        echo "Skipping file $file"
    # Add your skip logic here
    else
        echo "Processing file $file"
        jupyter nbconvert --to notebook --execute $file --inplace
    fi
done
