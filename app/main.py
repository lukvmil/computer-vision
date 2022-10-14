import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure

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
    sample_h, sample_s, sample_v = cv2.split(sample_hsv)
    sat_mask = cv2.inRange(
        sample_hsv,
        np.array([0,0,0]),
        np.array([255,100,255])
    )
    # show_image(sat_mask, "mask")


    # blurred = cv2.GaussianBlur(sample_grey, (25, 25), 0)
    # show_image(blurred, "blurred")


    sample_inv_grey = cv2.bitwise_not(sample_grey)
    # show_image(sample_inv_grey, "inv")

    masked = cv2.bitwise_and(sample, sample, mask=sat_mask)
    # show_image(masked, "masked")

    ret1, binary_image = cv2.threshold(src=sample_grey, thresh=200, maxval=255, type=cv2.THRESH_OTSU)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((13,13),np.uint8),iterations=1)
    binary_image_inv =  cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        area = cv2.contourArea(c)


        if area > 400:
            approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)

            if len(approx) == 4:
                mask = np.zeros(sample_grey.shape, np.uint8)
                cv2.drawContours(mask, [approx], 0, (255), -1)
                subset = cv2.bitwise_and(sample, sample, mask=mask)
                show_image(subset, f"mask-{count}")

                total_val = np.sum(subset)
                print(total_val / area, total_val, area)

                cv2.drawContours(sample, [approx], 0, (0, 0, 255), 5)
                # show_image(mask, f"mask-{count}")
        count += 1


    show_image(sample)

    show_image(binary_image, "binary")
    
    labels = measure.label(binary_image_inv)
    properties = measure.regionprops(labels)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))

    # for p in properties:
    #     ax.plot(np.round(p.centroid[1]), np.round(p.centroid[0]), '.g', markersize=10)
    #     a, b, c, d = p.bbox
    #     feature_sample = sample[a:c,b:d]
        # show_image(feature_sample, str(p))
    # plt.show()

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
