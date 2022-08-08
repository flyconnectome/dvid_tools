import os
import dvid as dv
import pytest
import navis

import numpy as np
import pandas as pd
import trimesh as tm

from pathlib import Path

from .common import redact_server


dv.setup(server=os.environ['DVID_TEST_SERVER'],
         node=os.environ['DVID_TEST_NODE'],
         user='dvid_tools_CI'
         )

# These are DA1 lPNs
BODYID = 722817260
BODYIDS = [722817260, 754534424, 754538881]


@pytest.mark.parametrize('output', ['auto', 'navis', 'trimesh'])
@redact_server
def test_get_mesh(output):
    m = dv.get_meshes(BODYID, on_error='raise', output=output)

    assert isinstance(m, list if output == 'trimesh' else navis.NeuronList)
    assert isinstance(m[0], tm.Trimesh if output == 'trimesh' else navis.MeshNeuron)

    m = dv.get_meshes(BODYIDS, on_error='raise', output=output)

    assert isinstance(m, list if output == 'trimesh' else navis.NeuronList)
    assert isinstance(m[0], tm.Trimesh if output == 'trimesh' else navis.MeshNeuron)


@pytest.mark.parametrize('output', ['auto', 'navis', 'swc'])
@redact_server
def test_get_skeletons(output):
    sk = dv.get_skeletons(BODYID, on_error='raise', output=output)

    assert isinstance(sk, list if output == 'swc' else navis.NeuronList)
    assert isinstance(sk[0],
                      pd.DataFrame if output == 'swc' else navis.TreeNeuron)

    sk = dv.get_skeletons(BODYIDS, on_error='raise', output=output)

    assert isinstance(sk, list if output == 'swc' else navis.NeuronList)
    assert isinstance(sk[0], pd.DataFrame if output == 'swc' else navis.TreeNeuron)


@redact_server
def test_get_annotations():
    an = dv.get_annotation(BODYID)

    assert isinstance(an, dict)


@redact_server
def test_locs_to_ids():
    ids = dv.locs_to_ids([[16384, 15079, 10804]])

    assert isinstance(ids, np.ndarray)
    assert len(ids) == 1


@redact_server
def test_get_sizes():
    s = dv.get_sizes(BODYIDS)

    assert isinstance(s, np.ndarray)
    assert len(s) == len(BODYIDS)


@redact_server
def test_ids_exist():
    exists = dv.ids_exist(BODYIDS)

    assert isinstance(exists, np.ndarray)
    assert len(exists) == len(BODYIDS)
    assert all(exists)

    exists = dv.ids_exist([123])  # fake ID
    assert not any(exists)


# This seems to be broken ATM
#@redact_server
#def test_get_body_position():
#    locs = dv.get_body_position(BODYID)
#
#    assert isinstance(locs, np.ndarray)


@redact_server
def test_rois():
    rois = dv.get_available_rois()

    assert isinstance(rois, list)
    assert len(rois) > 0

    r = dv.get_roi(rois[0], form='MESH')


@redact_server
def test_get_sparsevol_size():
    s = dv.get_sparsevol_size(BODYID)

    assert isinstance(s, dict)



@pytest.mark.parametrize('scale', ['COARSE', 4])
@redact_server
def test_get_sparsevol(scale):
    vxl = dv.get_sparsevol(BODYID, scale=scale)

    assert isinstance(vxl, np.ndarray)
    assert vxl.ndim == 2
    assert vxl.shape[1] == 3



@pytest.mark.parametrize('scale', ['COARSE', 4])
@redact_server
def test_mesh_neuron(scale):
    mesh = dv.mesh_neuron(BODYID, scale=scale)

    assert isinstance(mesh, tm.Trimesh)


@redact_server
def test_get_segmentation_info():
    info = dv.get_segmentation_info()

    assert isinstance(info, dict)


@redact_server
def test_get_n_synapses():
    n_syn = dv.get_n_synapses(BODYID)

    assert isinstance(n_syn, dict)


@pytest.mark.parametrize('with_details', [True, False])
@redact_server
def test_get_synapses(with_details):
    syn = dv.get_synapses(BODYID, with_details=with_details)

    assert isinstance(syn, pd.DataFrame)


@redact_server
def test_get_connections():
    cn = dv.get_connections(BODYIDS[0], BODYIDS[1])

    assert isinstance(cn, pd.DataFrame)


@redact_server
def test_get_connectivity():
    cn = dv.get_connectivity(BODYID)

    assert isinstance(cn, pd.DataFrame)


@redact_server
def test_get_adjacency():
    adj = dv.get_adjacency(BODYIDS, BODYIDS)

    assert isinstance(adj, pd.DataFrame)


@redact_server
def test_get_last_mod():
    lm = dv.get_last_mod(BODYID)

    assert isinstance(lm, dict)


@redact_server
def test_get_skeleton_mutation():

    lm = dv.get_skeleton_mutation(BODYID)

    assert isinstance(lm, int)


@redact_server
def test_projects():
    proj = dv.list_projects()

    assert isinstance(proj, pd.DataFrame)

    mn = dv.get_master_node(proj.iloc[0].Root)

    assert isinstance(mn, str)
