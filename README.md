[![Documentation Status](https://readthedocs.org/projects/dvidtools/badge/?version=latest)](http://dvidtools.readthedocs.io/en/latest/?badge=latest)[![Tests](https://github.com/flyconnectome/dvid_tools/actions/workflows/test-package.yml/badge.svg)](https://github.com/flyconnectome/dvid_tools/actions/workflows/test-package.yml)

# dvidtools
Python tools to fetch data from [DVID](https://github.com/janelia-flyem/dvid) servers.

Find the documentation [here](https://dvidtools.readthedocs.io).

Want to query a neuPrint server instead? Check out
[neuprint-python](https://github.com/connectome-neuprint/neuprint-python).

## What can `dvidtools` do for you?

- get/set user bookmarks
- get/set neuron annotations (names)
- download precomputed meshes, skeletons (SWCs) and ROIs
- get basic neuron info (# of voxels/synapses)
- get synapses
- get connectivity (adjacency matrix, connectivity table)
- retrieve labels (TODO, to split, etc)
- map positions to body IDs
- mesh or skeletonize sparsevols
- detect potential open ends (based on a script by [Stephen Plaza](https://github.com/stephenplaza))

## Install

Make sure you have [Python 3](https://www.python.org) (3.6 or later),
[pip](https://pip.pypa.io/en/stable/installing/) and
[git](https://git-scm.com) installed. Then run this in terminal:

```shell
pip install git+git://github.com/flyconnectome/dvid_tools@master
```

## Dependencies
- numpy
- pandas
- scikit-image
- tqdm
- scipy
- requests
- networkx

Above dependencies will be installed automatically. If you plan to use the tip
detector with classifier-derived confidence, you will also need
[sciki-learn](https://scikit-learn.org):

```shell
pip3 install scikit-learn
```

For from-scratch skeletonization you need to install `skeletor`:

```shell
pip3 install skeletor
```

## Examples
Please see the [documentation](https://dvidtools.readthedocs.io) for examples.
