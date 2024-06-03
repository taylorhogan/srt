# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time

import cv2 as cv
from matplotlib import pyplot as plt


def file_age(filepath):
    dt = time.time() - os.path.getmtime(filepath)
    seconds = dt
    minutes = int(seconds) / 60  # 120 minutes
    hours = minutes / 60  # 2 hours

    return "File is " + str(int(minutes)) + " minutes old"


def analyse_safety(image_path):
    img_rgb = cv.imread(image_path)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    img2 = img.copy()
    img_rgb_copy = img_rgb.copy()

    template = cv.imread('base_images/marker.jpg', cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #          'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    methods = ['cv.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        r = int(w / 2)

        c = (top_left[0] + r, top_left[1] + r)

        cv.circle(img, c, r * 5, 255, 2)

        plt.imshow(img, cmap='gray')
        plt.title('Detected Point' + str(c)), plt.xticks([]), plt.yticks([])
        plt.savefig('./scratch/position.png')
        plt.show()

    cv.rectangle(img_rgb_copy, (0, 0, 2000, 400), (0, 0, 0), -1)

    age = file_age(image_path)
    org = (10, 200)
    cv.putText(img_rgb_copy, age, org, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=10, color=(255, 255, 255))

    org = (10, 300)
    cv.putText(img_rgb_copy, "Roof is closed", org, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=10,
               color=(255, 255, 255))

    org = (10, 400)
    cv.putText(img_rgb_copy, "Scope is Parked", org, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=10,
               color=(255, 255, 255))

    cv.imshow("foo", img_rgb_copy)
    cv.waitKey(0)
    cv.destroyAllWindows()


analyse_safety('base_images/inside.jpg')
