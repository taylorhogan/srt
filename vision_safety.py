# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time
import math
import cv2 as cv
import inside_camera_server
import config


cfg = config.data()


def find_template_rectangle (image, template_image_path):
    # Load the main image and the template
    template = cv.imread(template_image_path, cv.IMREAD_COLOR)  # Path to the template
    main_image = image

    # Convert the images to grayscale for processing
    main_image_gray = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    # Get dimensions of the template
    template_height, template_width = template_gray.shape[:2]

    # Apply template matching (choose a method, e.g., TM_CCOEFF_NORMED)
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(main_image_gray, template_gray, method)

    # Find the minimum and maximum values with their locations
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Decide the top-left corner of the best match based on the method
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        best_match_top_left = min_loc
    else:
        best_match_top_left = max_loc

    # Calculate the bottom-right corner using the top-left corner and template size
    best_match_bottom_right = (best_match_top_left[0] + template_width,
                               best_match_top_left[1] + template_height)


    center = ((best_match_top_left[0] + template_width) / 2, (best_match_top_left[1] + template_height) / 2)
    return best_match_top_left, best_match_bottom_right, center

def test_find_template(image, template_image_path):

    main_image = image

    best_match_top_left, best_match_bottom_right, center = find_template_rectangle(image, template_image_path)

    print (best_match_top_left, best_match_bottom_right, center)
    # Draw a rectangle around the matched region on the original image
    cv.rectangle(main_image, best_match_top_left, best_match_bottom_right, (0, 255, 0), 2)


    # Display the results
    cv.imshow("Matched Image", main_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_template(image, template_image_path):
    best_match_top_left, best_match_bottom_right, center = find_template_rectangle(image, template_image_path)









def visual_status():

    print ("take snapshot")
    inside_camera_server.take_snapshot()

    print("read snapshot")
    image_path = cfg["camera safety"]["scope_view"]
    image_rgb = cv.imread(image_path, cv.IMREAD_COLOR)
    mod_date = time.ctime(os.path.getmtime(image_path))

    print ("analysing image")
    parked_best_match_top_left, parked_best_match_bottom_right, parked_center = find_template_rectangle(image_rgb, cfg['camera safety']['parked template'])
    closed_best_match_top_left, closed_best_match_bottom_right, closed_center = find_template_rectangle(image_rgb, cfg['camera safety']['closed template'])

    cv.rectangle(image_rgb, parked_best_match_top_left, parked_best_match_bottom_right, (0, 0, 255), 2)
    cv.imwrite(cfg["camera safety"]["scope_view"], image_rgb)
    cv.rectangle(image_rgb, closed_best_match_top_left, closed_best_match_bottom_right, (0, 255, 0), 2)
    cv.imwrite(cfg["camera safety"]["scope_view"], image_rgb)


    parked_error = math.dist(parked_center, cfg["camera safety"]["parked pos"])
    print(parked_center)
    print(cfg["camera safety"]["parked pos"])
    print (parked_error)
    parked = abs(parked_error) < 20

    closed_error = math.dist(closed_center, cfg["camera safety"]["parked pos"])
    print(closed_center)
    print(cfg["camera safety"]["closed pos"])
    print(closed_error)
    accuracy = cfg["camera safety"]["accuracy"]
    print(accuracy)
    closed = abs(closed_error) < accuracy

    mod_date = time.ctime(os.path.getmtime(cfg["camera safety"]["scope_view"]))
    return parked,  closed, mod_date



if __name__ == '__main__':
    cfg = config.data()
    just_finding_template = True
    if just_finding_template:

        inside_camera_server.take_snapshot()
        image =  img_rgb = cv.imread(cfg["camera safety"]["scope_view"], cv.IMREAD_COLOR)
        test_find_template(image, cfg['camera safety']['open template'])
    else:
        parked, closed, mod_date = visual_status()
        print (parked, closed, mod_date)
