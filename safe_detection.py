
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('base_images/inside.jpg')
img = cv.imread('base_images/inside.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = cv.imread('base_images/marker.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
#methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
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


    r = int(w/2)
    c = (top_left[0] + r, top_left[1] + r)




    cv.circle(img, c, r * 5, 255, 2)
    plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point' + str(c)), plt.xticks([]), plt.yticks([])
    plt.savefig('position.png')
    plt.show()


