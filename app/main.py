import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from app.utilities import *
from app.aruco import *
from app import config


images = [config.directory + '\\' + os.fsdecode(f) for f in os.listdir(os.fsencode(config.directory))]
marker_lookup_table = load_markers('aruco.json')

sleep = 1000000
standard_height = config.standard_height
marker_size = config.marker_size
marker_size += 2 # accounting for boundary pixels
stroke = 2

if config.USE_VIDEO:
    vid = cv2.VideoCapture(config.video_feed)
    sleep = 1

detection_count = 0
frames = 0

img_id = 0
while True:
    if config.USE_VIDEO:
        ret, img = vid.read()
    else:
        if config.DEBUG: print(images[img_id])
        img = cv2.imread(images[img_id])

    marker_detected = False

    # automatically resizes to standard height while preserving aspect ratio
    height, width, channels = img.shape
    aspect_ratio = width / height
    
    normalized_height = standard_height
    normalized_width = int(standard_height * aspect_ratio)

    img = cv2.resize(img, (normalized_width, normalized_height))

    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # the high and low pass are diffed to remove noise / softer edges form shadows
    # img_blur_diff_lg, avg_blur_diff_lg = diff_of_blurs(img_bw, 11, 51)
    img_blur_diff_md, avg_blur_diff_md = diff_of_blurs(img_bw, 7, 31)
    # img_blur_diff_sm, avg_blur_diff_sm = diff_of_blurs(img_bw, 5, 21)

    img_bw_diff = img_blur_diff_md

    # morphological operations, thresholding
    img_bw_close = cv2.morphologyEx(img_bw_diff, cv2.MORPH_CLOSE, make_kernel(3), iterations=1)
    img_thresh_low = cv2.threshold(img_bw_close, config.binary_threshold_low, 255, cv2.THRESH_BINARY)[1]
    img_thresh_high = cv2.threshold(img_bw_close, config.binary_threshold_high, 255, cv2.THRESH_BINARY)[1]
    
    img_thresh_low_close = cv2.morphologyEx(img_thresh_low, cv2.MORPH_CLOSE, make_kernel(3), iterations=1)
    img_thresh_low_dilate = cv2.dilate(img_thresh_low_close, make_kernel(3), iterations=1)

    # img_thresh_high_close = cv2.morphologyEx(img_thresh_low, cv2.MORPH_CLOSE, make_kernel(3), iterations=1)
    # img_thresh_high_dilate = cv2.dilate(img_thresh_high_close, make_kernel(3), iterations=2)

    # contours detected from dilated edge
    contours_low, _ = cv2.findContours(img_thresh_low_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_high, _ = cv2.findContours(img_thresh_high_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # import pdb; pdb.set_trace()

    img_thresh3 = to3(img_thresh_low_dilate)

    # contours sorted from greatest to smallest area
    contour_list_low = [[x, cv2.contourArea(x)] for x in contours_low]
    # contour_list_high = [[x, cv2.contourArea(x)] for x in contours_high]
    contour_list = contour_list_low # + contour_list_high
    contour_list.sort(key=lambda x: x[1], reverse=True)

    areas = [x[1] for x in contour_list]

    found_points = []

    # iterates through n largest contours
    for contour, area in contour_list[:config.search_n_largest_contours]:
        perimeter = cv2.arcLength(contour, True)
        epsilon = config.poly_approx_epsilon
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

        # converting to a python list
        points = [x[0] for x in approx.tolist()]

        # min area contours discarded
        if area < config.min_area:
            if config.DEBUG:
                try:
                    cv2.drawContours(img_thresh3, [approx], 0, (255, 255, 0), stroke)
                    cv2.drawContours(img, [approx], 0, (255, 255, 0), stroke)
                except NameError:
                    import pdb; pdb.set_trace()
            continue

        # only looking for four sided polygons
        if len(points) != 4: 
            if config.DEBUG:
                cv2.drawContours(img_thresh3, [approx], 0, (255, 0, 0), stroke)
                cv2.drawContours(img, [approx], 0, (255, 0, 0), stroke)
            continue

        side_lengths = get_poly_lengths(points)
        angles = get_poly_angles(points)
        abs_angles = [abs(x) for x in angles]
        min_angle = min(abs_angles)

        # min angle contours discarded
        if min_angle < config.min_angle:
            if config.DEBUG:
                cv2.drawContours(img_thresh3, [approx], 0, (0, 255, 255), stroke)
                cv2.drawContours(img, [approx], 0, (255, 0, 0), stroke)
            continue

        # checks if lengths are within a % of the mean, approximately square
        # if not dist_from_mean_within(side_lengths, 3): 
        #     cv2.drawContours(img_thresh3, [approx], 0, (0, 0, 255), stroke)
        #     print('EXCLUDED')
        #     continue

        avg_point = average_point(points)

        # ignores duplicate markers when contours overlap
        duplicate_marker = False
        for pt in found_points:
            if get_dist(pt, avg_point) < config.duplicate_marker_min_distance:
                duplicate_marker = True
                break
        
        if duplicate_marker: continue

        # normalizes points based on average
        normalized_points = [ [x[0] - avg_point[0], x[1] - avg_point[1]] for x in points ]

        # sorting in a clockwise order using reverse tangents
        normalized_points.sort(key=lambda x: math.atan2(x[1], x[0]), reverse=False)

        # longest side used for square dimensions of propsective transform
        size = int(max(side_lengths))
        src = np.float32(approx)
        dst = np.float32([[0,0], [size-1, 0], [size-1, size-1], [0, size-1]])

        H_mat = cv2.getPerspectiveTransform(src, dst)

        normalized_marker_size = config.normalized_marker_size

        # normalized to 
        corrected_marker = cv2.warpPerspective(img_bw, H_mat, (size, size), flags=cv2.INTER_NEAREST)
        normalized_marker = cv2.resize(corrected_marker, (normalized_marker_size, normalized_marker_size))

        # segments = process_histogram(normalized_marker)

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

        # threshold calculated as average pixel value of a marker
        pixel_threshold = normalized_marker.sum() / (len(normalized_marker) * len(normalized_marker[0]))
        pixel_size = normalized_marker_size / marker_size

        # computer marker code from normalized marker image
        marker_code = read_marker(normalized_marker, pixel_threshold, pixel_size, marker_size, config.sub_pixel_offset)
        # marker id found from lookup table
        marker_id = marker_lookup_table.get(tuple(marker_code))

        normalized_marker3 = to3(normalized_marker)
        draw_grid(normalized_marker3, (marker_size, marker_size))

        # print(marker_code)
        # print('MATCH FOUND:', marker_id)
        # show_image(normalized_marker3, 'square ' + str(marker_code))

        if marker_id is not None:
            if config.DEBUG:
                show_image(normalized_marker3, 'marker ' + str(marker_id))
                # cv2.drawContours(img_thresh3, [approx], 0, (0, 255, 0), stroke)

            found_points.append(avg_point)

            marker_detected = True

            cv2.drawContours(img, [approx], 0, (0, 255, 0), stroke)
            cv2.putText(img, str(marker_id), avg_point, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
        else:
            if config.DEBUG:
                cv2.drawContours(img_thresh3, [approx], 0, (0, 0, 255), stroke)
                cv2.drawContours(img, [approx], 0, (0, 0, 255), stroke)
            
    if config.DEBUG:
        show_image(np.hstack((img, to3(img_bw_diff), img_thresh3)), "pipeline", (600*3, 800))
    else:
        show_image(img, "main")

    if marker_detected:
        detection_count += 1

    frames += 1
    
    action = cv2.waitKey(sleep)
    if action & 0xFF == 27:
        break
    elif action == ord('r'):
        sleep = 1
    elif action == ord('s'):
        sleep = 1000000
    
    if not config.USE_VIDEO:
        img_id += 1
        if img_id >= len(images):
            break

    
if config.USE_VIDEO:
    vid.release()

cv2.destroyAllWindows()

print(f"{detection_count}/{frames if config.USE_VIDEO else img_id} detected markers")