import cv2
import numpy as np

def show_image(img, title="", size=(600, 800)):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, size[0], size[1])

def make_kernel(size):
    return np.ones((size,size), np.uint8)

# convert grayscale to bgr
def to3(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def get_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def get_poly_lengths(poly):
    lengths = []
    for i, curr_point in enumerate(poly):
        # if index is 0, compare to the last point
        if i == 0:
            prev_point = poly[-1]
        else:
            prev_point = poly[i-1]

        # euclidean distance
        dist = (curr_point[0][0] - prev_point[0][0]) ** 2 + (curr_point[0][1] - prev_point[0][1]) ** 2
        
        lengths.append(dist)
    return lengths

# checks if all side lengths are within 1 + dist % of the mean
def dist_from_mean_within(lengths, dist):
    mean = sum(lengths) / len(lengths)

    for l in lengths:
        if l > mean * dist:
            return False
    
    return True

def weighted_average(dict):
    val_sum = 0
    for v in dict.values():
        val_sum += v
    
    average = 0

    for k,v in dict.items():
        average += k * (v / val_sum)

    return average


def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img