# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time
import math
import cv2 as cv
import inside_camera_server
import config
import matplotlib.pyplot as plt

cfg = config.data()

def find_template(image, template_image_path):
    template = cv.imread(template_image_path,  cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    #d, w, h = template.shape[::-1]
    w, h = template.shape[::-1]

    method = 'cv.TM_SQDIFF_NORMED'

    methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
               'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']

    local_img = image.copy()
    method = eval(method)

    # Apply template Matching
    res = cv.matchTemplate(local_img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    print ("fitness: " + str(max_val))



    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    r = int(min(w, h) / 2)

    c = (int(top_left[0] + w / 2), int(top_left[1] + h / 2))

    return c, r, max_val


def fitness(image, delta):

    c_scope, r_scope, accuracy = find_template(image, cfg['camera safety']['parked template'])

    parked_error = math.dist(c_scope, cfg["camera safety"]["parked pos"])



    parked = abs(parked_error) < delta

    return  parked,  c_scope,  r_scope, accuracy


def analyse_safety(image_path):

    img_rgb = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    print("read snapshot")

    assert img_rgb is not None, "file could not be read, check with os.path.exists()"

    height, width = img_rgb.shape[:2]
    delta = min(width, height) * 0.1

    print(width, height)

    img_rgb_copy = img_rgb.copy()

    alpha = 0.5  # Contrast control
    beta = 0  # Brightness control
    best = 0
    best_alpha = 0
    best_beta = 0

    best_accuracy = 0
    for attempt in range(0, 10):

        # call convertScaleAbs function
        adjusted = cv.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
        parked, c_scope, r_scope, accuracy  = fitness(adjusted, delta)
        print (parked, c_scope, r_scope, accuracy)

        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_alpha = alpha
            best_beta = beta




        alpha = alpha + 0.2

    alpha = best_alpha
    beta = best_beta
    adjusted = cv.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
    parked, c_scope, r_scope, accuracy = fitness(adjusted, delta)
    print("solution found at " + str(alpha))
    cv.circle(img_rgb_copy, c_scope, r_scope, (0, 0, 255), 2)
    plt.imshow(img_rgb_copy,cmap = 'gray')


    plot_path = 'base_images/position.png'
    if os.path.exists(plot_path):
        os.remove(plot_path)

    plt.savefig(plot_path)

    mod_date = time.ctime(os.path.getmtime(image_path))
    return parked,  mod_date


if __name__ == '__main__':
    cfg = config.data()
    inside_camera_server.take_snapshot()
    is_parked,  mod_date = analyse_safety(cfg["camera safety"]["scope_view"])
    #is_closed, is_parked, is_open, mod_date = analyse_safety("./base_images/inside.jpg")
    reply = "Scope Parked:" + str(is_parked) + "\n"
    reply += "Copied Date:" + mod_date + "\n"
    print(reply)
