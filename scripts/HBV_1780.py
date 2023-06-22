'''
The driver functions to read images from the camera.
'''

import cv2
from HBV_1780_constants import *

def hbv_1780_start(device_index = 2):
    """Initialize the camera.

    Args:
        device_index (int, optional): the camera device index in the operating system. Defaults to 2.

    Returns:
        cv2.VideoCapture: the video capture object to the camera, you can use the standard OpenCV API to read the frames.
    """
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HBV_1780_SET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HBV_1780_SET_HEIGHT)

    # -- read a frame for testing --
    ret, test_frame = cap.read()
    
    if not ret:
        print("camera device invalid or not functioning properly, please check")
        return None

    return cap

def hbv_1780_get_left_and_right(image):
    """Get the left and right images from the raw image returned from the camera.

    Args:
        image (numpy.ndarray): the raw image read from the camera.

    Returns:
        numpy.ndarray, numpy.ndarray: the left and right images, respectively.
    """
    left = image[:, 0:HBV_1780_SET_WIDTH_HALF, :]
    right = image[:, HBV_1780_SET_WIDTH_HALF:, :]
    return left, right

# -- for debug, you can run this script on its own to check if everything is fine --
if __name__ == "__main__":
    camera = hbv_1780_start()

    if camera != None:
        cv2.namedWindow("left")
        cv2.namedWindow("right")

        while True:
            ret, frame = camera.read()
            if not ret:
                print("failed to grab frame")
                break

            frame_l, frame_r = hbv_1780_get_left_and_right(frame)

            cv2.imshow("left", frame_l)
            cv2.imshow("right", frame_r)

            k = cv2.waitKey(1)
            # -- finish if the 'q' key is pressed --
            if k == ord('q'):
                break

        camera.release()

        cv2.destroyAllWindows()
