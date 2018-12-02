# dvid_tools
Python tools to query DVID server.

## Dependencies
- numpy
- pandas
- scikit-image

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

# Write body ID in column
bm['BodyID'] = bm.Prop.map(lambda x : x['body ID'])

# Fetch annotations
bm['BodyName'] = bm.BodyID.map(lambda x: dt.get_annotation(x).get('name', None))
```

Fetch and save SWC for a single neuron
```Python
body_id = '1700937093'
dt.get_skeleton(body_id, save_to=body_id + '.swc')
```

