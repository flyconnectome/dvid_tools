# dvid_tools
Python tools to query [DVID](https://github.com/janelia-flyem/dvid) server.

## Install

```Python
`pip3 install git+git://github.com/flyconnectome/dvid_tools@master`
```

## Dependencies
- numpy
- pandas
- scikit-image
- tqdm

## Examples

Setting up
```Python
import dvidtools as dt 

# You can pass these parameters explicitly to each function but defining
# them globally is more convenient
server = 'http://emdata3.int.janelia.org:8900'
node = '54f7'
user = 'schlegelp'

dt.set_param(server, node, user)
```

Get user bookmarks and add annotations
```Python
# Get bookmarks
bm = dt.get_user_bookmarks()

# Add column with neuron name (if available)
bm['body name'] = bm['body ID'].map(lambda x: dt.get_annotation(x).get('name', None))
```

Fetch and save SWC for a single neuron
```Python
body_id = '1700937093'
dt.get_skeleton(body_id, save_to=body_id + '.swc')
```

Get table of synapse locations
```Python
body_id = '1700937093'
syn = dt.get_synapses(body_id)
```

Get synaptic partners of a neuron
```Python
body_id = '1700937093'
partners = dt.get_connectivity(body_id)
```

Get connectivity in given ROI using [pymaid](https://pymaid.readthedocs.io)
```Python
import pymaid

# Get the LH ROI
lh = pymaid.Volume(*dt.get_roi('LH'))

# Fetch connectivity but use filter function
lh_partners = dt.get_connectivity(body_id, pos_filter=lambda x: pymaid.in_volume(x, lh))
```