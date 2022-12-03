import numpy as np
from app.utilities import get_poly_angles

points = np.array([
 [[0, 0]],
 [[3, 3]],
 [[5, 3]],
 [[3, 2]]])

points.shape = (-1, 2)


print(get_poly_angles(points))
# funny shape because OpenCV. it's a Nx1 vector of 2-channel elements
# fix that up, remove the silly dimension

print(points)

# the vectors are differences of coordinates
# a points into the point, b out of the point
a = points - np.roll(points, 1, axis=0)
b = np.roll(a, -1, axis=0) # same but shifted

# we'll need to know the length of those vectors
alengths = np.linalg.norm(a, axis=1)
blengths = np.linalg.norm(b, axis=1)

# we need only the length of the cross product,
# and we work in 2D space anyway (not 3D),
# so the cross product can't result in a vector, just its z-component

# print(np.dot(a, b), np.cross(a, b))
dotproducts = [-np.dot(a[i], b[i]) for i in range(len(a))] / alengths / blengths
# cosine_angle = np.dot(a, b) / alengths / blengths
cos_angles = np.arccos(dotproducts)
cos_angles_degrees = np.degrees(cos_angles)

# import pdb; pdb.set_trace()

crossproducts = np.cross(a, b) / alengths / blengths

sin_angles = np.arcsin(crossproducts)
sin_angles_degrees = sin_angles / np.pi * 180



print("angles in degrees:")
print(sin_angles_degrees, sum(sin_angles_degrees))

print(cos_angles_degrees, sum(cos_angles_degrees))

# this is just for printing/displaying, not useful in code
print("point and angle:")
print(np.hstack([points, sin_angles_degrees.reshape((-1, 1))]))

final_angles = []

if len([x < 0 for x in sin_angles_degrees]) > 2:
    sin_angles_degrees = -sin_angles_degrees

for i in range(len(sin_angles)):
    sin_angle = round(sin_angles_degrees[i], 2)
    cos_angle = round(cos_angles_degrees[i], 2)

    if (sin_angle > 0) == (cos_angle > 0):
        final_angles.append(cos_angle)
    else:
        final_angles.append(360 - cos_angle)

print(final_angles)

