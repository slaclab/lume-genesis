# lume-genesis
Genesis tools for use in LUME

## Genesis 1.3 v2.0 Installation
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


