Installing LUME-Genesis via conda-forge
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



Developers
==========


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




