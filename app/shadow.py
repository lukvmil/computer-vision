import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure

def show_image(img, title=""):
    cv2.imshow(title, cv2.resize(img, (600, 800)))

directory = "aruco\\shadows_only"
images = [directory + '\\' + os.fsdecode(f) for f in os.listdir(os.fsencode(directory))]
sleep = 1000000

# clahe
# kernelizied correlation filters
# trackerkcf

for i in images:
    sample = cv2.imread(i)
    # show_image(sample)

    sample_grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    # sample_grey = cv2.bitwise_not(sample_grey)
    show_image(sample_grey, "grey")

    blurred = cv2.GaussianBlur(sample_grey, (25, 25), 0)
    show_image(blurred, "blurred")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    show_image(thresh, "thresh")

    sample_eroded = cv2.erode(thresh, np.ones((3,3),np.uint8), iterations=1)
    # show_image(sample_eroded, "erode")
    

    sample_dilated = cv2.morphologyEx(sample_eroded, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8), iterations=5)
    # show_image(sample_dilated, "close")

    sample_blur = cv2.medianBlur(sample_grey, 51)
    show_image(sample_blur, "blur")

    backdrop = np.zeros(sample_grey.shape, np.uint8)
    backdrop[:] = (75)

    sample_diff = 255 - cv2.absdiff(sample_grey, blurred)
    show_image(sample_diff)

    # sample_diff = 255 - cv2.absdiff(sample_grey, sample_blur)
    # show_image(sample_diff)
    
    # norm_img = sample_diff.copy()
    # cv2.normalize(sample_diff, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # show_image(norm_img, "norm")


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
