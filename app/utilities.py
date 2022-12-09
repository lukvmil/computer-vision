import cv2
import numpy as np
from app import config

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
        # dist = (curr_point[0][0] - prev_point[0][0]) ** 2 + (curr_point[0][1] - prev_point[0][1]) ** 2
        dist = get_dist(curr_point, prev_point)

        lengths.append(dist)
    return lengths

def get_poly_angles(points):
    a = points - np.roll(points, 1, axis=0)
    b = np.roll(a, -1, axis=0)

    alengths = np.linalg.norm(a, axis=1)
    blengths = np.linalg.norm(b, axis=1)

    dotproducts = [-np.dot(a[i], b[i]) for i in range(len(a))] / alengths / blengths
    cross = lambda x,y:np.cross(x,y)
    crossproducts = cross(a, b) / alengths / blengths

    cos_angles = np.arccos(dotproducts)
    cos_angles_degrees = np.degrees(cos_angles)

    sin_angles = np.arcsin(crossproducts)
    sin_angles_degrees = sin_angles / np.pi * 180

    final_angles = []

    if len([x < 0 for x in sin_angles_degrees]) > 2:
        sin_angles_degrees = -sin_angles_degrees

    for i in range(len(sin_angles)):
        sin_angle = sin_angles_degrees[i]
        cos_angle = cos_angles_degrees[i]

        if (sin_angle > 0) == (cos_angle > 0):
            final_angles.append(cos_angle)
        else:
            final_angles.append(360 - cos_angle)


    # print(sin_angles_degrees)
    # print(cos_angles_degrees)

    return sin_angles_degrees
        

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


def draw_grid(img, grid_shape, color=(255, 0, 0), thickness=1, offset=config.sub_pixel_offset):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = int(round(h / rows)), int(round(w / cols))

    # draw vertical lines
    for x in range(dx, w, dx):
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in range(dy, h, dy):
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    for x in range(0, h, dy):
        for y in range(0, w, dx):
            cv2.rectangle(img, (x+offset,y+offset), (x+dx-offset, y+dy-offset), color=(0,255,0), thickness=thickness)

    return img


def average_point(points):
    sum_x, sum_y = 0, 0

    for p in points:
        sum_x += p[0]
        sum_y += p[1]
    
    return int(sum_x / len(points)), int(sum_y / len(points))
    

def process_histogram(image):
    # histogram calculated for each match
    hist = cv2.calcHist([image], [0], None, [256], (0, 256), accumulate=False)

    # all code below is histogram processing
    # calculates the two distinct values for that sample
    # (in ideal case this is the black and white of the code)

    # converting to python list
    hist_dict = {}
    hist_list = [int(x[0]) for x in hist.tolist()]

    # avg val / 2
    clamp_value = (sum(hist_list) / len(hist_list)) / 2

    for i, val in enumerate(hist_list):
        # clamps values below 100 to 0, better show gap between histogram peaks
        hist_dict[i] = val if val > clamp_value else 0

    segments = []
    curr_segment = {}

    for key, val in hist_dict.items():
        if val != 0:
            curr_segment[key] = val
        else:
            if len(curr_segment):
                segments.append(curr_segment)
                curr_segment = {}

    # sorted by segment length in descending order
    segments.sort(key=lambda x: len(x), reverse=True)
    
    # plt.figure()
    # plt.plot(hist, color="red")
    # plt.title("Value")
    # plt.show()

    return segments


def diff_of_blurs(img, k1, k2):
        img_blur_low = cv2.GaussianBlur(img, (k1, k1), 0)
        img_blur_high = cv2.GaussianBlur(img, (k2, k2), 0)
        img_blur_diff = cv2.absdiff(img_blur_high, img_blur_low)

        mean_value = img_blur_diff.sum() / (len(img_blur_diff) * len(img_blur_diff[0]))

        return img_blur_diff, mean_value