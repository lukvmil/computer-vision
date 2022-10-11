import json

with open('aruco.json', 'r') as f:
    markers = json.load(f)

# converts integer representation of markers to binary grid for later use in image analysis
for orientation in markers:
    m = orientation[0]
    bin_repr1 = format(m[0], '08b')
    bin_repr2 = format(m[1], '08b')
    code = [bin_repr1[:4], bin_repr1[4:], bin_repr2[:4], bin_repr2[4:]]