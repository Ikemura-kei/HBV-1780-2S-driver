import cv2
from HBV_1780_constants import *

def hbv_1780_start(device_index = 2):
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HBV_1780_SET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HBV_1780_SET_HEIGHT)
    # cap.set(cv2.CAP_PROP_FPS, 2)

    ret, test_frame = cap.read()
    
    if not ret:
        print("camera device invalid or not functioning properly, please check")
        return None

    return cap

def hbv_1780_get_left_and_right(image):
    left = image[:, 0:HBV_1780_SET_WIDTH_HALF-1, :]
    right = image[:, HBV_1780_SET_WIDTH_HALF:-1, :]
    return left, right

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
            if k == ord('q'):
                print("Escape hit, closing...")
                break

        camera.release()

        cv2.destroyAllWindows()
