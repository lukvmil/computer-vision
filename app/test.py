import numpy as np

a = np.array([2,0])
b = np.array([4,1])
c = np.array([4,3])

ba = a - b
bc = c - b

cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle = np.arccos(cosine_angle)

# print(np.degrees(angle))

def get_poly_angles(poly):
    angles = []
    poly = [np.array(x) for x in poly]
    for i, b in enumerate(poly):
        if i == 0:
            a = poly[-1]
        else:
            a = poly[i-1]
        
        if i == len(poly) - 1:
            c = poly[0]
        else:
            c = poly[i+1]
        
        ba = a - b
        bc = c - b

        print(ba, bc)
        print(np.dot(ba, bc))
        print(np.linalg.norm(ba), np.linalg.norm(bc))
        

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        angles.append(np.degrees(angle))
    
    return angles

# print(get_poly_angles([
#     (0,0),
#     (5,0),
#     (5,3),
#     (3,1)
# ]))