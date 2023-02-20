# Installation


---
## Installing lume-genesis via conda-forge


Installing `lume-genesis` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `lume-genesis` can be installed with:

```
conda install lume-genesis
```

It is possible to list all of the versions of `lume-genesis` available on your platform with:

```
conda search lume-genesis --channel conda-forge
```

---
## Installing Genesis 1.3 version 2 Executables

See: [slaclab/Genesis-1.3-Version2 Installation](https://github.com/slaclab/Genesis-1.3-Version2#precompiled)

---
## Installing Genesis 1.3 version 4 Executables

See: [svenreiche/Genesis-1.3-Version4 dev installation](https://github.com/svenreiche/Genesis-1.3-Version4/blob/dev/manual/INSTALLATION.md)

Please use the `dev` branch to be compatible with LUME-Genesis.

Once built, set this environmental variable so that LUME-Genesis can find the executable:
```bash
export GENESIS4_BIN=/path/to/build/genesis4
```



### macOS 

Installation on macOS requires a suitable compiler and dependencies, which can be provided by [MacPorts](https://www.macports.org). With a working MacPorts, install the GCC12 compiler and dependencies:
```bash
sudo port install gcc12
sudo port select gcc mp-gcc12
sudo port install openmpi-gcc12
sudo port select mpi openmpi-gcc12-fortran
sudo port install hdf5 +openmpi 
sudo port install fftw-3
```

Get the latest Genesis4 code
```
git clone https://github.com/svenreiche/Genesis-1.3-Version4
cd Genesis-1.3-Version4/
git fetch 
git swtich dev
```

Then build Genesis:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```


## Genesis4 on Cori (NERSC)



```bash
git clone https://github.com/svenreiche/Genesis-1.3-Version4
cd Genesis-1.3-Version4/
git fetch 
git swtich dev
module load cray-hdf5-parallel
module load cray-fftw
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=CC
cmake --build build
```





---
## Old Genesis 1.3 v2.0 Installation
Go to <http://genesis.web.psi.ch/download.html> and download:
<http://genesis.web.psi.ch/download/source/genesis_source_2.0_120629.tar.gz>

Untar, and replace one of the source files with one provided in this repository (cloned at <ROOT>):
```
  tar -xzvf genesis_source_2.0_120629.tar
  cd  Genesis_Current
  cp <ROOT>/lume-genesis/extra/fix_genesis_input/input.f .
   
```
Edit Makefile to point to your compile, and type:
```
make
```
This should build the  `genesis` binary.


---
## Developers


Clone this repository:
```shell
git clone https://github.com/slaclab/lume-genesis.git
```

Create an environment `genesis-dev` with all the dependencies:
```shell
conda env create -f environment.yml
```


Install as editable:
```shell
conda activate genesis-dev
pip install --no-dependencies -e .
```




