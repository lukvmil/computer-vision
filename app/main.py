import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from app.utilities import *
from app.aruco import *


directory = "aruco/custom_aruco"
images = [directory + '\\' + os.fsdecode(f) for f in os.listdir(os.fsencode(directory))]
marker_lookup_table = load_markers('aruco.json')
# print(marker_lookup_table)

# for key in marker_lookup_table.keys():
#     for i in key: print(i)
#     print()
#     if marker_lookup_table[key] != 0: break


sleep = 1000000
standard_height = 800
marker_size = 4
marker_size += 2 # accounting for boundary pixels

USE_VIDEO = True

if USE_VIDEO:
    vid = cv2.VideoCapture(2)
    sleep = 1

# clahe
# kernelizied correlation filters
# trackerkcf

img_id = 18
while True:
    if USE_VIDEO:
        ret, img = vid.read()
    else:
        img = cv2.imread(images[img_id])

    # automatically resizes to standard height while preserving aspect ratio
    height, width, channels = img.shape
    aspect_ratio = width / height
    
    normalized_height = standard_height
    normalized_width = int(standard_height * aspect_ratio)

    img = cv2.resize(img, (normalized_width, normalized_height))

    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # the high and low pass are diffed to remove noise / softer edges form shadows

    img_blur_diff_lg, avg_blur_diff_lg = diff_of_blurs(img_bw, 11, 51)
    img_blur_diff_md, avg_blur_diff_md = diff_of_blurs(img_bw, 7, 31)
    img_blur_diff_sm, avg_blur_diff_sm = diff_of_blurs(img_bw, 5, 21)

    img_bw_diff = img_blur_diff_sm

    show_image(img_blur_diff_lg, "11-51")
    show_image(img_blur_diff_md, "7-31")
    show_image(img_blur_diff_sm, "5-21")

    # use three different pairs and pick the best = highest mean pixel value

    # clahe applied for alternate pipeline, this was less effective than other approach
    # clahe = cv2.createCLAHE(clipLimit = 0, tileGridSize=(4,4))
    # img_clahe = clahe.apply(img_bw)

    # img_clahe_blurred = cv2.GaussianBlur(img_clahe, (51, 51), 0)
    # img_clahe_diff = cv2.absdiff(img_clahe, img_clahe_blurred)

    # img_clahe_thresh = cv2.threshold(img_clahe_diff, 170, 255, cv2.THRESH_BINARY_INV)[1]

    # morphological operations, thresholding
    img_bw_close = cv2.morphologyEx(img_bw_diff, cv2.MORPH_CLOSE, make_kernel(3), iterations=1)
    img_thresh = cv2.threshold(img_bw_diff, 10, 255, cv2.THRESH_BINARY)[1]
    img_thresh_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, make_kernel(3), iterations=1)
    img_thresh_dilate = cv2.dilate(img_thresh_close, make_kernel(3), iterations=3)

    img_thresh_dilate = img_thresh_close

    # contours detected from dilated edge
    contours, _ = cv2.findContours(img_thresh_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    img_thresh3 = to3(img_thresh_dilate)

    contour_list = [[x, cv2.contourArea(x)] for x in contours]
    contour_list.sort(key=lambda x: x[1], reverse=True)

    areas = [x[1] for x in contour_list]

    # contour_list = []

    for contour, area in contour_list[:10]: 
        # if area < 200_000: continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.05 # 0.009

        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

        # converting to a python list
        points = [x[0] for x in approx.tolist()]

        # only looking for four sided polygons
        if len(points) != 4: 
            cv2.drawContours(img_thresh3, [approx], 0, (0, 255, 255), 5)
            continue

        side_lengths = get_poly_lengths(points)
        
        angles = get_poly_angles(points)
        abs_angles = [abs(x) for x in angles]
        min_angle = min(abs_angles)

        if min_angle < 20:
            cv2.drawContours(img_thresh3, [approx], 0, (255, 0, 0), 5)
            continue

        # checks if lengths are within a % of the mean, approximately square
        if not dist_from_mean_within(side_lengths, 3): 
            cv2.drawContours(img_thresh3, [approx], 0, (0, 0, 255), 5)
            print('EXCLUDED')
            continue

        avg_point = average_point(points)
        normalized_points = [ [x[0] - avg_point[0], x[1] - avg_point[1]] for x in points ]

        # sorting in a clockwise order using reverse tangents
        normalized_points.sort(key=lambda x: math.atan2(x[1], x[0]), reverse=False)

        size = int(max(side_lengths))
        src = np.float32(approx)
        dst = np.float32([[0,0], [size-1, 0], [size-1, size-1], [0, size-1]])

        H_mat = cv2.getPerspectiveTransform(src, dst)

        normalized_marker_size = 256

        corrected_marker = cv2.warpPerspective(img_bw, H_mat, (size, size), flags=cv2.INTER_NEAREST)
        normalized_marker = cv2.resize(corrected_marker, (normalized_marker_size, normalized_marker_size))

        normalized_marker3 = to3(normalized_marker)

        draw_grid(normalized_marker3, (6, 6))

        segments = process_histogram(normalized_marker)

        # if (len(segments) < 2):
        #     # print("less than 2 values detected, discarding")
        #     cv2.drawContours(img_thresh3, [approx], 0, (255, 0, 0), 5)
        #     continue

        # min_segment_width = 5

        # if (len(segments[0]) < min_segment_width) or (len(segments[1]) < min_segment_width):
        #     # print("value peak width less than 5, discarding")
        #     cv2.drawContours(img_thresh3, [approx], 0, (255, 0, 0), 5)
        #     continue

        # checks that there are two distinct value peaks with a width > 5 distinct values
        
        # value1 = weighted_average(segments[0])
        # value2 = weighted_average(segments[1])

        # if value1 > value2:
        #     value_high = value1
        #     value_low = value2
        # else:
        #     value_high = value2
        #     value_low = value1

        # pixel_threshold = (value_high + value_low) / 2

        pixel_threshold = normalized_marker.sum() / (len(normalized_marker) * len(normalized_marker[0]))
        pixel_size = normalized_marker_size / marker_size

        marker_code = read_marker(normalized_marker, pixel_threshold, pixel_size, marker_size)

        print(marker_code)
        marker_id = marker_lookup_table.get(tuple(marker_code))
        print('MATCH FOUND:', marker_id)
        show_image(normalized_marker3, 'square')

        if marker_id is not None:
            show_image(normalized_marker3, 'marker ' + str(marker_id))
            print('ANGLES', angles)

            cv2.drawContours(img_thresh3, [approx], 0, (0, 255, 0), 5)
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)
        else:
            cv2.drawContours(img_thresh3, [approx], 0, (0, 0, 255), 5)
            
    # show_image(np.hstack((img_clahe, img_clahe_blurred, img_clahe_diff)), "clahe", (600*3, 800))
    show_image(np.hstack((img, to3(img_bw_diff), img_thresh3)), "pipeline", (600*3, 800))
    # show_image(np.hstack((to3(img_thresh), to3(img_thresh_close), to3(img_thresh_dilate), img_thresh3)), "diff", (600*3, 800))

    # show_image(img_clahe, 'clahe')

    # show_image(img, 'img')
    # show_image(img_bw, 'img_bw')
    # show_image(img_bw_diff, "combined_diff")
    # show_image(img_bw_close, "closed")
    # show_image(img_thresh, "thresh")
    # show_image(img_thresh_close, "thresh_closed")
    # show_image(img_thresh_dilate, "thresh_dilate")
    # show_image(img_thresh3, "contours")

    
    action = cv2.waitKey(sleep)
    if action & 0xFF == 27:
        break
    elif action == ord(' '):
        if not USE_VIDEO:
            img_id += 1
            if img_id >= len(images):
                break
        continue
    elif action == ord('r'):
        sleep = 1
    elif action == ord('s'):
        sleep = 1000000

    
if USE_VIDEO:
    vid.release()

cv2.destroyAllWindows()
