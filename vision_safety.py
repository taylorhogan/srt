# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time
import math
import sys
import cv2 as cv
import baseconfig as config
import inside_camera_server
import matplotlib.pyplot as plt

cfg = config.FlowConfig().config

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

    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    r = int(min(w, h) / 2)

    c = (int(top_left[0] + w / 2), int(top_left[1] + h / 2))

    return c, r


def fitness(image, delta):

    c_roof, r_roof = find_template(image, cfg['camera safety']['roof template'])
    c_scope, r_scope = find_template(image, cfg['camera safety']['parked template'])

    roof_closed_error = math.dist(c_roof, cfg["camera safety"]["closed pos"])
    roof_open_error = math.dist(c_roof, cfg["camera safety"]["open pos"])
    parked_error = math.dist(c_scope, cfg["camera safety"]["parked pos"])


    # print("roof " + str(c_roof))
    # print("scope " + str(c_scope))
    # print("closed error " + str(roof_closed_error))
    # print("open error " + str(roof_open_error))
    # print("parked error " + str(parked_error))


    closed = abs(roof_closed_error) < delta
    parked = abs(parked_error) < delta
    open = abs(roof_open_error) < delta

    return closed, parked, open, c_roof, c_scope, r_roof, r_scope


def analyse_safety(image_path):
    print("a")

    print("b")

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

    for attempt in range(0, 10):

        # call convertScaleAbs function
        adjusted = cv.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
        closed, parked, open, c_roof, c_scope, r_roof, r_scope = fitness(adjusted, delta)

        found = 0
        if closed:
            found = found + 1
        if open:
            found = found + 1
        if parked:
            found = found + 1
        print ("Attempt:" + str(attempt) + " Found:" + str(found))
        if found > best:
            best = found
            best_alpha = alpha
            best_beta = beta
        else:
            alpha = alpha + 0.2

    alpha = best_alpha
    beta = best_beta
    adjusted = cv.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
    closed, parked, open, c_roof, c_scope, r_roof, r_scope = fitness(adjusted, delta)
    print("solution found at " + str(alpha))
    cv.circle(img_rgb_copy, c_roof, r_roof, (255, 0, 0), 2)
    cv.circle(img_rgb_copy, c_scope, r_scope, (0, 0, 255), 2)
    plt.imshow(img_rgb_copy,cmap = 'gray')

    plt.title('roof (x,y) parked (x,y)' + str(c_roof) + " " + str(c_scope)), plt.xticks([]), plt.yticks([])

    plot_path = './base_images/position.png'
    if os.path.exists(plot_path):
        os.remove(plot_path)

    plt.savefig(plot_path)

    mod_date = time.ctime(os.path.getmtime(image_path))
    return closed, parked, open, mod_date


if __name__ == '__main__':
    if len(sys.argv) == 1:
        status = inside_camera_server.take_snapshot()
        #status = inside_camera_server.take_snapshot("./base_images/inside.jpg")
    else:
        status = inside_camera_server.take_snapshot("./base_images/inside.jpg")

    if status:

        is_closed, is_parked, is_open, mod_date = analyse_safety(cfg["camera safety"]["scope_view"])
        reply = "Roof Closed: " + str(is_closed) + "\n"
        reply += "Roof Open: " + str(is_open) + "\n"
        reply += "Scope Parked:" + str(is_parked) + "\n"
        reply += "Copied Date:" + mod_date + "\n"
        print(reply)
