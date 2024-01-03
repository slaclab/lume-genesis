#!/bin/bash

NOTEBOOKS=$(find docs/examples/genesis4/ -type f -name "*.ipynb" -not -path '*/.*')

SKIP="perlmutter" 

#echo $NOTEBOOKS

# Silence Jupyterlab warning
export PYDEVD_DISABLE_FILE_VALIDATION=1

for file in $NOTEBOOKS
do
    if [[ "$file" == *"$SKIP"* ]]; then 
        echo "Skipping $file"
        continue
    fi
  
    echo "Executing $file"
    jupyter nbconvert --to notebook --execute $file --inplace
done
