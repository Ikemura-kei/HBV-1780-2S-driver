from argparse import ArgumentParser
from HBV_1780 import *
from HBV_1780_constants import *
import time
import datetime
import cv2
import os

# -- constants --
TAKE_PIC_COOLDOWN = 2
MODE_BY_KEY = 0
MODE_CONTINUOUS = 1
BY_KEY_DURATION = 1.5

def main(right_path, left_path, mode):
    if mode != MODE_BY_KEY and mode != MODE_CONTINUOUS:
        print("--> Invalid mode")
        return
    
    if mode == MODE_CONTINUOUS:
        print("--> Please press any key to start taking pictures, the process will start 2 seconds after you press any key")
        k = input()
        time.sleep(2)
    
    camera = hbv_1780_start()

    if camera is None:
        print("no camera detected!")
        return 0

    cv2.namedWindow('left')
    cv2.namedWindow('right')

    print("left image size:", hbv_1780_get_left_and_right(camera.read()[1])[0].shape)
    print("right image size:", hbv_1780_get_left_and_right(camera.read()[1])[1].shape)
    
    cooldown_counter = 0
    last_taken = time.time()
    pic_cnt = 0
    while True:
        cooldown_counter += 1
        err, image = camera.read()

        if err is False:
            print('camera error')
            break

        left, right = hbv_1780_get_left_and_right(image)

        cv2.imshow('left', left)
        cv2.imshow('right', right)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        
        duration = time.time() - last_taken
        
        is_shooting = key == ord('c') if mode==MODE_BY_KEY else duration>=BY_KEY_DURATION
        
        if is_shooting and cooldown_counter > TAKE_PIC_COOLDOWN:
            last_taken = time.time()
            cooldown_counter = 0
            pic_cnt += 1

            print("took picture", pic_cnt)

            cv2.imwrite(os.path.join(left_path, str(pic_cnt) + '.jpg'), left)
            cv2.imwrite(os.path.join(right_path, str(pic_cnt) + '.jpg'), right)


if __name__ == "__main__":
    print("start HBV_1780 shooting")

    parser = ArgumentParser(description="HBV_1780 picture taking program")

    parser.add_argument('--pfolder', type=str, required=True, dest='parent_folder')
    parser.add_argument('--mode', type=int, required=True)

    date_time = datetime.datetime.now()
    d = date_time.strftime("%m-%d-%Y-%H:%M:%S")
    parser.add_argument('--name', type=str, default=d, dest="folder_name")

    args = parser.parse_args()

    # verify if the passed parent_folder argument is indeed a valid directory
    if not os.path.isdir(args.parent_folder):
        print("--> [%s] is not a directory! Please pass a valid parent folder." % (args.parent_folder))
        exit()

    path = os.path.join(args.parent_folder, args.folder_name)
    right_folder = os.path.join(path, 'right')
    left_folder = os.path.join(path, 'left')
    print('--> Creating folders \n[%s] \nand \n[%s]\n' % (left_folder, right_folder))

    os.mkdir(path)
    os.mkdir(right_folder)
    os.mkdir(left_folder)

    main(left_folder, right_folder, args.mode)

    