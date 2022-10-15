import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
# from skimage import measure

def show_image(img, title="", size=(600, 800)):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, size[0], size[1])

def make_kernel(size):
    return np.ones((size,size), np.uint8)

# convert grayscale to bgr
def to3(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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


directory = "example_markers"
images = [directory + '\\' + os.fsdecode(f) for f in os.listdir(os.fsencode(directory))]
sleep = 1000000

# clahe
# kernelizied correlation filters
# trackerkcf

for i in images:
    img = cv2.imread(i)

    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculates two gaussian blurs with a small and large kernel size
    # each one is diffed with the original image for edge detection
    img_bw_blurred_low = cv2.GaussianBlur(img_bw, (11, 11), 0)
    img_bw_diff_low = cv2.absdiff(img_bw, img_bw_blurred_low)

    img_bw_blurred_high = cv2.GaussianBlur(img_bw, (51, 51), 0)
    img_bw_diff_high = cv2.absdiff(img_bw, img_bw_blurred_high)

    # the high and low pass are diffed to remove noise / softer edges form shadows
    img_bw_diff = cv2.absdiff(img_bw_diff_high, img_bw_diff_low)

    # clahe applied for alternate pipeline, this was less effective than other approach
    # clahe = cv2.createCLAHE(clipLimit = 10)
    # img_clahe = clahe.apply(img_bw)

    # img_clahe_blurred = cv2.GaussianBlur(img_clahe, (51, 51), 0)
    # img_clahe_diff = cv2.absdiff(img_clahe, img_clahe_blurred)

    # img_clahe_thresh = cv2.threshold(img_clahe_diff, 170, 255, cv2.THRESH_BINARY_INV)[1]

    # morphological operations, thresholindg
    img_bw_close = cv2.morphologyEx(img_bw_diff, cv2.MORPH_CLOSE, make_kernel(5), iterations=1)
    img_thresh = cv2.threshold(img_bw_diff, 20, 255, cv2.THRESH_BINARY)[1]
    img_thresh_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, make_kernel(5), iterations=1)
    img_thresh_dilate = cv2.dilate(img_thresh_close, make_kernel(5), iterations=3)

    # contours detected from dilated edge
    contours, _ = cv2.findContours(img_thresh_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_thresh3 = to3(img_thresh_dilate)

    for c in contours:
        area = cv2.contourArea(c)

        # considers "significant" contours
        if area > 200_000:
            approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
            
            # only looking for four sided polygons
            if len(approx) == 4:
                print("target found with correct area and sides")
                lengths = get_poly_lengths(approx)
                # checks if lengths are within a % of the mean, approximately square
                if dist_from_mean_within(lengths, 3):
                    print("side length ratio correct")
                    
                    # contour converted to mask
                    mask = np.zeros(img_bw.shape, np.uint8)
                    cv2.drawContours(mask, [approx], 0, (255), -1)
                    # used to mask and isolate from grayscale image
                    subset = cv2.bitwise_and(img_bw, img_bw, mask=mask)

                    # histogram calculated for each match
                    hist = cv2.calcHist([img_bw], [0], mask, [256], (0, 256), accumulate=False)

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

                    cv2.drawContours(img_thresh3, [approx], 0, (0, 255, 0), 5)

                    # plt.figure()
                    # plt.plot(hist, color="red")
                    # plt.title("Value")
                    # plt.show()

                    if (len(segments) < 2):
                        print("less than 2 values detected, discarding")
                        continue

                    min_segment_width = 5

                    if (len(segments[0]) < min_segment_width) or (len(segments[1]) < min_segment_width):
                        print("value peak width less than 5, discarding")
                        continue

                    # checks that there are two distinct value peaks with a width > 5 distinct values
                    
                    value1 = weighted_average(segments[0])
                    value2 = weighted_average(segments[1])

                    if value1 > value2:
                        value_high = value1
                        value_low = value2
                    else:
                        value_high = value2
                        value_low = value1

                    # print(area)
                    # show_image(subset, 'mask')

                    cv2.drawContours(img_thresh3, [approx], 0, (0, 0, 255), 5)
                
    # show_image(np.hstack((img_clahe, img_clahe_blurred, img_clahe_diff)), "clahe", (600*3, 800))
    show_image(np.hstack((to3(img_bw), to3(img_bw_diff), img_thresh3)), "pipeline", (600*3, 800))

    # show_image(img, 'img')
    # show_image(img_bw, 'img_bw')
    # show_image(img_bw_blurred_high, "blurred_high")
    # show_image(img_bw_diff_low, "diff_low")
    # show_image(img_bw_diff_high, "diff_high")
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
        continue
    elif action == ord('r'):
        sleep = 1
    elif action == ord('s'):
        sleep = 1000000


cv2.destroyAllWindows()
