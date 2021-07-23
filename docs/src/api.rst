.. _api:

API Reference
=============

Reading/Writing Data
++++++++++++++++++++

.. currentmodule:: dvid.fetch

.. autosummary::
    :toctree: _autosummary

    add_bookmarks
    edit_annotation
    ids_exist
    get_adjacency
    get_annotation
    get_assignment_status
    get_available_rois
    get_body_id
    get_body_position
    get_body_profile
    get_connections
    get_connectivity
    get_labels_in_area
    get_last_mod
    get_master_node
    get_n_synapses
    get_neuron
    get_roi
    get_segmentation_info
    get_sizes
    get_skeletons
    get_synapses
    get_user_bookmarks
    setup
    locs_to_ids

Tools
+++++

.. currentmodule:: dvid.tip

.. autosummary::
    :toctree: _autosummary

    detect_tips

Utility
+++++++

.. currentmodule:: dvid.utils

.. autosummary::
    :toctree: _autosummary

    check_skeleton
    gen_assignments
    heal_skeleton
    parse_swc_str
    reroot_skeleton
    save_swc
