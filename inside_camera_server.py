
import cv2 as cv
import config
import shutil



def take_snapshot(test_path=None):
    cfg = config.data()
    if test_path is not None:
        to_path = cfg["camera safety"]["scope_view"]
        shutil.copyfile(test_path, to_path)
        return True

    print("taking picture")

    no_image = cfg["camera safety"]["no_image"]
    to_path = cfg["camera safety"]["scope_view"]
    shutil.copyfile(no_image, to_path)

    vid = cv.VideoCapture(0)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = vid.read()
    if ret:
        img_src = frame
        cv.imwrite(to_path, img_src)
        return True
    else:
        print("no Image")
        return False



if __name__ == '__main__':
    cfg = config.data()

