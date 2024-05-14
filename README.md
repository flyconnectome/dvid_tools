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
- generate meshes or skeletons from scratch
- get basic neuron info (# of voxels/synapses)
- fetch synapses
- fetch connectivity (adjacency matrix, connectivity table)
- retrieve labels (TODO, to split, etc)
- map positions to body IDs
- detect potential open ends (based on a script by [Stephen Plaza](https://github.com/stephenplaza))

## Install

Make sure you have [Python 3](https://www.python.org) (3.8 or later),
[pip](https://pip.pypa.io/en/stable/installing/). Then run this:

```bash
pip3 install dvidtools
```

To install the dev version straight from Github:

```bash
pip3 install git+https://github.com/flyconnectome/dvid_tools@master
```

## Optional dependencies
Necessary dependencies will be installed automatically.

If you plan to use the tip detector with classifier-derived confidence, you
will also need [sciki-learn](https://scikit-learn.org):

```shell
pip3 install scikit-learn
```

For from-scratch skeletonization you need to install `skeletor`:

```shell
pip3 install skeletor
```

## Examples
Please see the [documentation](https://dvidtools.readthedocs.io) for examples.

## Testing

For testing you need to have two environment variables set: `DVID_TEST_SERVER`
and `DVID_TEST_NODE`. These should point to a DVID server/node that contain
the Janelia hemibrain dataset. Then run:

```bash
$ pytest -v
```
