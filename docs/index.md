# LUME-Genesis
Genesis tools for use in LUME


## Installation

Installing lume-genesis via conda-forge
=======================

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

Installing Genesis 1.3 version 2 Executables
============================================
See: [slaclab/Genesis-1.3-Version2 Installation](https://github.com/slaclab/Genesis-1.3-Version2#precompiled)


Installing Genesis 1.3 version 4 Executables
============================================
See: [svenreiche/Genesis-1.3-Version4 dev installation](https://github.com/svenreiche/Genesis-1.3-Version4/blob/dev/manual/INSTALLATION.md)

Please use the `dev` branch to be compatible with LUME-Genesis.

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



## Related Publications

The lume-genesis package was used in the following publications:

*Very high brightness and power LCLS-II hard X-ray pulses*\
Aliaksei Halavanau, Franz-Josef Decker, Claudio Emma, Jackson Sheppard, and Claudio Pellegrini\
J. Synchrotron Rad. (2019). 26\
https://doi.org/10.1107/S1600577519002492


