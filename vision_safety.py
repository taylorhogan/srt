# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time
import math
import sys
import cv2 as cv
import baseconfig as config
import inside_camera_server
import matplotlib.pyplot as plt



def find_template(image, template_image_path):
    template = cv.imread(template_image_path)
    assert template is not None, "file could not be read, check with os.path.exists()"
    d, w, h = template.shape[::-1]
    print (template_image_path + " is " + str(w)  + " wide and " + str(h) + " high " + str(d) + " depth ")

    method = 'cv.TM_CCOEFF_NORMED'

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

    c = (int(top_left[0] + w/2), int(top_left[1]+h/2))

    return c, r



def analyse_safety(image_path):
    print("a")
    cfg = config.FlowConfig().config
    print("b")

    img_rgb = cv.imread(image_path)
    print("read snapshot")

    alpha = 1.5  # Contrast control
    beta = 1 # Brightness control

    # call convertScaleAbs function
    adjusted = cv.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
    cv.imshow('Original', img_rgb)
    cv.imshow('Increased Contrast', adjusted)

    cv.waitKey(0)
    cv.destroyAllWindows()




    assert img_rgb is not None, "file could not be read, check with os.path.exists()"

    height, width = img_rgb.shape[:2]

    print(width, height)

    img_rgb_copy = img_rgb.copy()
    c_roof, r_roof = find_template(img_rgb, cfg['camera safety']['roof template'])
    c_scope, r_scope = find_template(img_rgb, cfg['camera safety']['parked template'])
    roof_closed_error = math.dist (c_roof, cfg["camera safety"]["closed pos"])
    roof_open_error = math.dist (c_roof, cfg["camera safety"]["open pos"])
    parked_error = math.dist (c_scope, cfg["camera safety"]["parked pos"])
    print ("roof " + str(c_roof))
    print ("scope " + str(c_scope))
    print ("closed error " + str (roof_closed_error))
    print ("open error " + str (roof_open_error))
    print ("parked error " + str(parked_error))

    delta = min(width, height) * 0.1
    is_closed = abs(roof_closed_error) < delta
    is_parked = abs (parked_error) < delta
    is_open =  abs (roof_open_error) < delta


    cv.circle(img_rgb_copy, c_roof, r_roof, (255, 0, 0), 2)
    cv.circle(img_rgb_copy, c_scope, r_scope, (0, 0, 255), 2)
    plt.imshow(img_rgb_copy)
    plt.title('roof (x,y) parked (x,y)' + str(c_roof) + " " + str(c_scope)), plt.xticks([]), plt.yticks([])

    plot_path = './base_images/position.png'
    if os.path.exists(plot_path):
        os.remove (plot_path)

    plt.savefig(plot_path)

    mod_date = time.ctime(os.path.getmtime(image_path))
    return is_closed, is_open, is_parked, mod_date


if __name__ == '__main__':
    if len(sys.argv) == 1:
        status = inside_camera_server.take_snapshot()
    else:
        status = inside_camera_server.take_snapshot("./base_images/inside.jpg")

    if status:
        cfg = config.FlowConfig().config
        is_closed, is_open, is_parked, mod_date = analyse_safety(cfg["camera safety"]["scope_view"])
        reply = "Roof Closed: " + str(is_closed) + "\n"
        reply += "Roof Open: " + str(is_open) + "\n"
        reply += "Scope Parked:" + str(is_parked) + "\n"
        reply += "Copied Date:" + mod_date + "\n"
        print (reply)


