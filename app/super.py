import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure

def show_image(img, title="", size=(600, 800)):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, size[0], size[1])

def make_kernel(size):
    return np.ones((size,size), np.uint8)

directory = "aruco\\shadows_only"
images = [directory + '\\' + os.fsdecode(f) for f in os.listdir(os.fsencode(directory))]
sleep = 1000000

# clahe
# kernelizied correlation filters
# trackerkcf

for i in images:
    img = cv2.imread(i)
    # show_image(sample)

    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sample_grey = cv2.bitwise_not(sample_grey)
    # show_image(img_bw, "bw")

    clahe = cv2.createCLAHE(clipLimit = 10)
    img_clahe = clahe.apply(img_bw)

    # show_image(img_clahe, "clahe")

    img_bw_blurred = cv2.GaussianBlur(img_bw, (51, 51), 0)
    img_bw_diff = 255 - cv2.absdiff(img_bw, img_bw_blurred)

    img_clahe_blurred = cv2.GaussianBlur(img_clahe, (51, 51), 0)
    # show_image(img_blurred, "blurred")

    img_clahe_diff = 255 - cv2.absdiff(img_clahe, img_clahe_blurred)
    # show_image(img_diff, "diff")

    # thresh = cv2.adaptiveThreshold(img_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    img_clahe_thresh = cv2.threshold(img_clahe_diff, 170, 255, cv2.THRESH_BINARY_INV)[1]

    # thresh_blurred = cv2.GaussianBlur(img_clahe_thresh, (11, 11), 0)

    # thresh_diff = 255 - cv2.absdiff(img_clahe_thresh, thresh_blurred)

    

    img_thresh = cv2.threshold(img_bw_diff, 210, 255, cv2.THRESH_BINARY_INV)[1]
    # show_image(img_thresh, "thresh")

    # img_thresh = cv2.bitwise_not(img_thresh)
    # show_image(img_thresh_inv, "inv")

    # img_thresh_erode = cv2.erode(img_thresh, make_kernel(1), iterations=5)

    img_thresh_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, make_kernel(5), iterations=1)
    # show_image(img_thresh_close, "close")

    img_canny = cv2.Canny(img_bw, 100, 200)

    lines = cv2.HoughLinesP(img_thresh, 1, np.pi/180, 300, minLineLength=30, maxLineGap=10)
    
    empty = cv2.cvtColor(img_thresh_close, cv2.COLOR_GRAY2RGB)

    empty = np.zeros(img_bw.shape, np.uint8)

    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(empty, (x1, y1), (x2, y2), (255), 5)

    # img_closed = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, make_kernel(3))



    # final2 = np.hstack((img_diff, img_closed, img_eroded))


    # show_image(np.hstack((img_clahe, img_clahe_blurred, img_clahe_diff)), "clahe", (600*3, 800))
    show_image(np.hstack((img_bw, img_bw_blurred, img_bw_diff)), "normal", (600*3, 800))
    show_image(np.hstack((img_thresh, img_thresh_close, empty)), "thresh", (600*3, 800))
    # show_image(np.hstack((cv2.cvtColor(img_bw,cv2.COLOR_GRAY2RGB), cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB), thresh_color)), "thresh", (600*3, 800))
    # show_image(empty, "lines")
    # show_image(final2, "test2", (600*3, 800))

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
