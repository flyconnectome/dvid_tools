DVIDtools
=========

`dvidtools <https://github.com/flyconnectome/dvid_tools>`_ allows you to read
and write data from `DVID <https://github.com/janelia-flyem/dvid>`_ servers.

Install
-------

Make sure you have `Python 3 <https://www.python.org>`_,
`pip <https://pip.pypa.io/en/stable/installing/>`_ and
`git <https://git-scm.com>`_ installed. Then run this in terminal:

::

    pip install git+git://github.com/flyconnectome/dvid_tools@master

If you plan to use the :func:`tip detector <dvidtools.tip.detect_tips>` with
classifier-derived confidence, you will also need
`sciki-learn <https://scikit-learn.org>`_:

::

    pip install scikit-learn


What can ``dvidtools`` do for you?
----------------------------------

- :func:`get <dvidtools.fetch.get_user_bookmarks>`/ :func:`set <dvidtools.fetch.add_bookmarks>` user bookmarks
- :func:`get <dvidtools.fetch.get_annotation>`/:func:`set <dvidtools.fetch.edit_annotation>` neuron annotations (names)
- download :func:`meshes <dvidtools.fetch.get_neuron>`, :func:`skeletons <dvidtools.fetch.get_skeleton>` (SWCs) and :func:`ROIs <dvidtools.fetch.get_roi>`
- get basic neuron info (# of :func:`voxels <dvidtools.fetch.get_body_profile>`/:func:`info <dvidtools.fetch.get_n_synapses>`)
- get :func:`synapses <dvidtools.fetch.get_synapses>`
- get connectivity (:func:`adjacency matrix <dvidtools.fetch.get_body_profile>`, :func:`connectivity table <dvidtools.fetch.get_connectivity>`)
- retrieve :func:`info <dvidtools.fetch.get_labels_in_area>` (TODO, to split, etc)
- map :func:`position <dvidtools.fetch.get_body_position>` to body IDs
- detect potential open :func:`ends <dvidtools.tip.detect_tips>` (based on a script by `Stephen Plaza <https://github.com/stephenplaza>`_)

Check out the full :doc:`API </src/api>` for more.

Examples
--------

Setting up
::

    import dvid as dv

    # You can pass these parameters explicitly to each function
    # but defining them globally is more convenient
    server = 'http://127.0.0.1:8000'
    node = '54f7'
    user = 'schlegelp'

    dv.set_param(server, node, user)


Get user bookmarks and add annotations
::

    # Get bookmarks
    bm = dv.get_user_bookmarks()

    # Add column with neuron name (if available)
    bm['body name'] = bm['body ID'].map(lambda x: dv.get_annotation(x).get('name', None))


Fetch skeleton for a single neuron and save as SWC
::

    body_id = '1700937093'
    dv.get_skeleton(body_id, save_to=body_id + '.swc')


Get table of synapse locations
::

    body_id = '1700937093'
    syn = dv.get_synapses(body_id)


Get synaptic partners of a neuron
::

    body_id = '1700937093'
    partners = dv.get_connectivity(body_id)


Get connectivity in given ROI using `navis <https://navis.readthedocs.io>`_
::

    import navis

    # Get the LH ROI
    lh = navis.Volume(*dv.get_roi('LH'))

    # Fetch connectivity but use filter function
    lh_partners = dv.get_connectivity(body_id, pos_filter=lambda x: navis.in_volume(x, lh))


Detect potential open ends and write them to ``.json`` file that can be
imported into `neutu <https://github.com/janelia-flyem/NeuTu>`_.
::

    body_id = '1700937093'
    tips = dv.detect_tips(body_id, save_to='~/Documents/{}.json'.format(body_id))


You can do the same but weight potential open ends using a pre-trained
classifier that provides "confidence" values. These confidence range
from -1 to +1 and give some indication whether a tip needs human attention or
not. This requires `sciki-learn <https://scikit-learn.org>`_ to be installed.
In a terminal run::

    pip install scikit-learn

Once scikit-learn is installed, you can run the tip detector with
classifier confidences::

    tips = dv.detect_tips(body_id, use_clf=True,
                          save_to='~/Documents/{}.json'.format(body_id))
