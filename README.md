# lume-genesis

Genesis tools for use in [LUME](http://lume.science/).

## Installing lume-genesis from conda-forge

### OpenMPI (recommended for parallel calculations)

```
conda install -c conda-forge lume-genesis genesis2=*=mpi_openmpi* genesis4=*=mpi_openmpi*
```

### MPICH (alternative for parallel calculations)

```
conda install -c conda-forge lume-genesis genesis2=*=mpi_mpich* genesis4=*=mpi_mpich*
```

### Non-MPI (non-parallel, single core calculations only)

```
conda install -c conda-forge lume-genesis
```

## List all `lume-genesis` versions available

```
conda search lume-genesis --channel conda-forge
```

## Development environment

A conda environment file is provided in this repository and may be used for a
development environment.

To create a new conda environment using this file, do the following:

```bash
git clone https://github.com/slaclab/lume-genesis
cd lume-genesis
conda env create -n lume-genesis-dev -f environment.yml
conda activate lume-genesis-dev
python -m pip install --no-deps -e .
```

Alternatively, with a virtualenv and pip:

```bash
git clone https://github.com/slaclab/lume-genesis
cd lume-genesis

python -m venv genesis-venv
source genesis-venv/bin/activate
python -m pip install -e .
```

## Related Publications

The lume-genesis package was used in the following publications:

_Very high brightness and power LCLS-II hard X-ray pulses_\
Aliaksei Halavanau, Franz-Josef Decker, Claudio Emma, Jackson Sheppard, and Claudio Pellegrini\
J. Synchrotron Rad. (2019). 26\
<https://doi.org/10.1107/S1600577519002492>
