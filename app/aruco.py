import json

with open('aruco.json', 'r') as f:
    markers = json.load(f)

marker_lookup_table = {}

# converts integer representation of markers to binary grid for later use in image analysis
for i, orientations in enumerate(markers):
    for marker in orientations:
        bin_repr1 = format(marker[0], '08b')
        bin_repr2 = format(marker[1], '08b')
        code = (bin_repr1[:4], bin_repr1[4:], bin_repr2[:4], bin_repr2[4:])
        marker_lookup_table[code] = i