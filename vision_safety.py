# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time
import math
import cv2 as cv
import baseconfig as config
import matplotlib.pyplot as plt


def take_snapshot():
    print ("taking picture")
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    if ret:
        img_src = frame
        cv.imwrite("base_images/scope_view.jpg", img_src)
    else:
        print("no Image")


def find_template(image, template_image_path):
    template = cv.imread(template_image_path)
    assert template is not None, "file could not be read, check with os.path.exists()"
    d, w, h = template.shape[::-1]


    method = 'cv.TM_SQDIFF_NORMED'

    local_img = image.copy()
    method = eval(method)

    # Apply template Matching
    res = cv.matchTemplate(local_img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    r = int(max(w, h) / 2)

    c = (int(top_left[0] + w/2), int(top_left[1]+h/2))

    return c, r



def analyse_safety(image_path):
    print("a")
    cfg = config.FlowConfig().config
    print("b")

    img_rgb = cv.imread(image_path)
    print("read snapshot")

    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_grayscale = cv.imread(image_path)
    height, width = img_grayscale.shape[:2]

    print(width, height)
    img_grayscale_copy = img_grayscale.copy()
    img_rgb_copy = img_rgb.copy()
    c_roof, r_roof = find_template(img_rgb, cfg['camera safety']['roof template'])
    c_scope, r_scope = find_template(img_rgb, cfg['camera safety']['parked template'])
    roof_closed_error = math.dist (c_roof, cfg["camera safety"]["closed pos"])
    parked_error = math.dist (c_scope, cfg["camera safety"]["parked pos"])
    is_closed = abs(roof_closed_error) < 2
    is_parked = abs (parked_error) < 2
    is_open = not is_closed


    cv.circle(img_grayscale_copy, c_roof, r_roof, (255, 0, 0), 2)
    cv.circle(img_grayscale_copy, c_scope, r_scope, (0, 0, 255), 2)
    plt.imshow(img_grayscale_copy)
    plt.title('roof (x,y) parked (x,y)' + str(c_roof) + " " + str(c_scope)), plt.xticks([]), plt.yticks([])

    plt.savefig('./scratch/position.png')

    return is_closed, is_open, is_parked

