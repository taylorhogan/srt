# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import sys
import time

import cv2 as cv

import baseconfig as cfg





def find_template(image, template_image):
    template = cv.imread('base_images/marker.jpg', cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]

    method = 'cv.TM_SQDIFF_NORMED'

    local_img = image.copy()
    method = eval(method)

    # Apply template Matching
    res = cv.matchTemplate(local_img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    r = int(w / 2)

    c = (top_left[0] + r, top_left[1] + r)

    return c, r


def analyse_safety(image_path, out_path):
    config = cfg.FlowConfig().config

    img_rgb = cv.imread(image_path)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_grayscale = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    height, width = img_grayscale.shape[:2]

    print(width, height)
    img_grayscale_copy = img_grayscale.copy()
    img_rgb_copy = img_rgb.copy()

    # c_open, r_open = find_template(config['camera safety']['image_path'], config['camera_safety']['open_template'])
    # c_closed, r_closed = find_template(config['camera safety']['image_path'], config['camera_safety']['closed_template'])
    # c_parked, r_parked = find_template(config['camera safety']['image_path'], config['camera_safety']['parked_template'])
    # c_goal = config['camera safety']['image_path']
    #
    # cv.circle(img_grayscale_copy, c, r * 5, 255, 2)
    # plt.imshow(img_grayscale_copy, cmap='gray')
    # plt.title('Detected Point' + str(c)), plt.xticks([]), plt.yticks([])
    # plt.savefig('./scratch/position.png')
    # plt.show()

    cv.rectangle(img_rgb_copy, (0, height - 400, width, height), (0, 0, 0), -1)
    font_height = 100
    mtime = "last modified: {}".format(time.ctime(os.path.getmtime(sys.argv[1])))
    org = (10, height - 2 * font_height)
    cv.putText(img_rgb_copy, mtime, org, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=10, color=(255, 255, 255))

    org = (10, height - font_height)
    cv.putText(img_rgb_copy, "Roof is closed", org, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=10,
               color=(255, 255, 255))

    org = (10, height)
    cv.putText(img_rgb_copy, "Scope is Parked", org, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=10,
               color=(255, 255, 255))

    # cv.imshow("foo", img_rgb_copy)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    cv.imwrite(out_path, img_rgb_copy)

    return False, True, True


def main(path_in, path_out):
    open, closed, parked = analyse_safety(path_in, path_out)
    print("Open:" + str(open))
    print("Closed:" + str(closed))
    print("Park:" + str(parked))

def take_snapshot (path_out):
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    img_src = frame
    cv.imwrite("scope_view.jpg", img_src)

if __name__ == '__main__':
    print(sys.argv)
    take_snapshot("")
    #main(sys.argv[1], sys.argv[2])
