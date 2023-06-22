'''
This file allows taking pictures using the stereo camera (e.g. for calibration) either continuously or by key presses.
'''
from argparse import ArgumentParser

from HBV_1780 import *
from HBV_1780_constants import *

import time
import datetime
import cv2
import os

# -- constants --
MODE_BY_KEY = 0 # MODE 1
MODE_CONTINUOUS = 1 # MODE 2
MODE_CONTINUOUS_DURATION_SECOND = 1.5 # we define in MODE 2, we take a picture once per some seconds

def main(right_path, left_path, mode):
    # -- check if the mode specified is valid --
    if mode != MODE_BY_KEY and mode != MODE_CONTINUOUS:
        print("--> Invalid mode")
        return
    
    # -- we ask user to indicate the start timing of picture taking when in MODE 2 --
    if mode == MODE_CONTINUOUS:
        print("--> Please press any key to start taking pictures, the process will start 2 seconds after you press any key.")
        k = input()
        time.sleep(2)
    else:
        print("--> Press 'c' to take a picture.")
    print("--> Once the process started, press 'q' to stop the process.")
    
    # -- initialize camera --
    camera = hbv_1780_start()
    if camera is None:
        print("--> No camera detected!")
        return 0

    # -- initialize visualizer --
    cv2.namedWindow('left')
    cv2.namedWindow('right')
    print("--> Left image size:", hbv_1780_get_left_and_right(camera.read()[1])[0].shape)
    print("--> Right image size:", hbv_1780_get_left_and_right(camera.read()[1])[1].shape)
    
    last_taken = time.time() # used to judge if there has been sufficient time passed since last picture taking
    pic_cnt = 0 # keeps track of the number of pictures taken
    while True:
        err, image = camera.read()

        if err is False:
            print('--> Camera error! Returning.')
            break

        left, right = hbv_1780_get_left_and_right(image)

        cv2.imshow('left', left)
        cv2.imshow('right', right)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        duration = time.time() - last_taken
        
        # -- take picture --:
        #   * when in MODE 1, by pressing the 'c' key
        #   * when in MODE 2, once per a time specified by MODE_CONTINUOUS_DURATION_SECOND
        is_shooting = key == ord('c') if mode==MODE_BY_KEY else duration>=MODE_CONTINUOUS_DURATION_SECOND
        
        if is_shooting:
            last_taken = time.time()
            pic_cnt += 1

            print("--> Took a picture, current #pictures:", pic_cnt)

            cv2.imwrite(os.path.join(left_path, str(pic_cnt) + '.jpg'), left)
            cv2.imwrite(os.path.join(right_path, str(pic_cnt) + '.jpg'), right)

if __name__ == "__main__":
    print("--> Start HBV-1780 shooting.")

    parser = ArgumentParser(description="HBV_1780 picture taking program.")

    parser.add_argument('--pfolder', type=str, required=True, help='the parent folder storing the folder containing images taken.')
    parser.add_argument('--mode', type=int, required=True, help='the picture taking mode, 0 for by-key mode, 1 for continous mode.')
    parser.add_argument('--name', type=str, default="", dest="folder_name", help="the name of the folder containing the images taken, will be created under the parent folder specified.")

    args = parser.parse_args()

    # -- verify if the passed parent_folder argument is indeed a valid directory --
    if not os.path.isdir(args.pfolder):
        print("--> [%s] is not an existing directory! Please pass a valid parent folder." % (args.pfolder))
        exit()
    # -- give a default name (the current date and time) to the folder storing pictures taken --
    if args.folder_name=="":
        date_time = datetime.datetime.now()
        date_string = date_time.strftime("%m-%d-%Y-%H:%M:%S")
        args.folder_name=date_string

    path = os.path.join(args.pfolder, args.folder_name)
    right_folder = os.path.join(path, 'right')
    left_folder = os.path.join(path, 'left')
    print('--> Creating folders \n\t[%s] \n\tand \n\t[%s]' % (left_folder, right_folder))

    os.makedirs(right_folder, exist_ok=True)
    os.makedirs(left_folder, exist_ok=True)

    main(left_folder, right_folder, args.mode)

    