# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os
import time
import math
import cv2 as cv
import inside_camera_server
import config
import matplotlib.pyplot as plt

cfg = config.data()

def test_find_template(image, template_image_path):



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

    print (best_match_top_left, best_match_bottom_right)
    # Draw a rectangle around the matched region on the original image
    cv.rectangle(main_image, best_match_top_left, best_match_bottom_right, (0, 255, 0), 2)


    # Display the results
    cv.imshow("Matched Image", main_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_template(image, template_image_path):
    # Load the main image and the template
    template = cv.imread(template_image_path, cv.IMREAD_COLOR)  # Path to the template
    main_image = image

    # Convert the images to grayscale for processing
    main_image_gray = cv2cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
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

    c_scope = (best_match_top_left[0] + template_width // 2, best_match_top_left[1] + template_height // 2)







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
        #print (parked, c_scope, r_scope, accuracy)

        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_alpha = alpha
            best_beta = beta




        alpha = alpha + 0.2

    alpha = best_alpha
    beta = best_beta
    adjusted = cv.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
    #parked, c_scope, r_scope, accuracy = fitness(adjusted, delta)
    print("solution found at " + str(alpha))
    cv.circle(img_rgb_copy, c_scope, r_scope, (0, 0, 255), 2)
    #plt.imshow(img_rgb_copy,cmap = 'gray')


    plot_path = 'base_images/position.png'
    if os.path.exists(plot_path):
        os.remove(plot_path)

    plt.savefig(plot_path)

    mod_date = time.ctime(os.path.getmtime(image_path))
    return parked,  mod_date


if __name__ == '__main__':
    cfg = config.data()
    inside_camera_server.take_snapshot()
    image =  img_rgb = cv.imread(cfg["camera safety"]["scope_view"], cv.IMREAD_COLOR)
    test_find_template(image, cfg['camera safety']['parked template'])
    #is_parked,  mod_date = analyse_safety(cfg["camera safety"]["scope_view"])
    #is_closed, is_parked, is_open, mod_date = analyse_safety("./base_images/inside.jpg")
    #reply = "Scope Parked:" + str(is_parked) + "\n"
    #reply += "Copied Date:" + mod_date + "\n"
    #print(reply)
