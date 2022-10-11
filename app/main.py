import cv2
import numpy as np
import os

def show_image(img, title=""):
    cv2.imshow(title, cv2.resize(img, (600, 800)))

directory = "aruco\\custom_aruco"
images = [directory + '\\' + os.fsdecode(f) for f in os.listdir(os.fsencode(directory))]
sleep = 1000000

for i in images:
    sample = cv2.imread(i)
    # show_image(sample)

    sample_grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    # show_image(sample_grey)

    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    print(sample_hsv)
    sample_h, sample_s, sample_v = cv2.split(sample_hsv)
    sat_mask = cv2.inRange(
        sample_hsv,
        np.array([0,0,0]),
        np.array([255,100,255])
    )
    # show_image(sat_mask, "mask")
    show_image(sample_v)


    # blurred = cv2.GaussianBlur(sample_grey, (25, 25), 0)
    # show_image(blurred, "blurred")


    sample_inv_grey = cv2.bitwise_not(sample_grey)
    # show_image(sample_inv_grey, "inv")

    masked = cv2.bitwise_and(sample, sample, mask=sat_mask)
    # show_image(masked, "masked")

    ret1, binary_image = cv2.threshold(src=sample_grey, thresh=200, maxval=255, type=cv2.THRESH_OTSU)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((13,13),np.uint8),iterations=1)
    show_image(binary_image, "otsu")
    
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
