'''
This file allows performing intrinsic calibration for the left and right camera, as well as stereo calibration.
'''

from argparse import ArgumentParser
from ast import arg
import cv2
import os
import numpy as np

# -- constants --
LEFT = 0
RIGHT = 1

def load_image_paths(parent_folder_of_calib_images):
    """Get the paths to all calibration images, including the left and right camera images.

    Args:
        parent_folder_of_calib_images (str): the path to the folder containing the calibration images, there should be two subfolders named "left" and "right" containing the respective calibration images.
    
    Returns:
        list of str, list of str: the lists of paths to the left and right calibration images, respectively.
    """
    left_paths, right_paths = [], []
    l_path, r_path = os.path.join(parent_folder_of_calib_images, 'left'), os.path.join(parent_folder_of_calib_images, 'right')

    for idx, f in enumerate(os.listdir(l_path)):
        l_file = os.path.join(l_path, f)
        r_file = os.path.join(r_path, f)

        # -- check file existance on the right folder --
        if not os.path.exists(r_file):
            print("--> The image '%s' exists in the left folder but not in the right folder!" % (f))

        left_paths.append(l_file)
        right_paths.append(r_file)

    return left_paths, right_paths

def get_obj_pnts(row, col):
    """Get the object points of the corners of the calibration pattern (i.e. points in the world coordinate) for calibration.

    Args:
        row (int): the number of rows on the chessboard pattern
        col (int): the number of columns on the chessboard pattern

    Returns:
        numpy.ndarray: the object points, of shape (row*col, 3).
    """
    obj_pnts = np.zeros((row * col, 3), np.float32)

    x, y = 0, 0
    for i in range(col):
        x = i
        for j in range(row):
            y = j
            pnt = np.array([x, y, 0])
            obj_pnts[x * row + y] = pnt

    return obj_pnts

def get_img_pnts(img, row, col, winSize):
    """Get the image points from a calibration image.

    Args:
        img (numpy.ndarray): the input calibration image
        row (int): the number of rows on the chessboard pattern
        col (int): the number of columns on the chessboard pattern
        winSize (tuple of two int): the window size used for cv2.cornerSubPix() function
        
    Returns:
        refine_corner (numpy.ndarray): the refined corner points
        image_with_corners (numpy.ndarray): the caliration image with the detected corners drawn
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # we use grayscale image for finding corners

    # -- step 1, apply chessboard corner detection --
    ret, corners = cv2.findChessboardCorners(gray_img, (row, col), None)
    if ret is False:
        return None, img

    # -- step 2, apply refinement to the output from step 1 --
    # NOTE: (criteria type, max num iterations, epsilon)
    subpix_refine_term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001)
    refine_corner = cv2.cornerSubPix(gray_img, corners, winSize, (-1, -1), subpix_refine_term_criteria) 

    # -- step 3, show the chessboard corners to user for sanity check --
    image_with_corners = np.copy(img)
    cv2.drawChessboardCorners(image_with_corners, (row, col), refine_corner, ret)
    cv2.putText(image_with_corners, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.52, (33,100,255), 1)

    return refine_corner, image_with_corners

def do_stereo_calibration(row, col, img_folder, tile_size, verbose):
    """Perform stereo calibration.

    Args:
        row (int): the number of rows on the chessboard
        col (int): the number of columns on the chessboard
        img_folder (str): the path to the parent folder containing calibration images (specifically, the 'left' and 'right' subfolder)
        tile_size (int): the size of the tile on the chessboard pattern, specified in mm
        verbose (bool): True to show detected corners, False otherwise

    Returns:
        calib_dict: the dictionary containing calibration results of the following:
                    * 'cmtx' (list of 2 numpy.ndarray): the 3x3 intrinsic matrices of the left and right camera, respectively
                    * 'dist' (list of 2 numpy.ndarray): the 5-element distortion coefficients of the left and right camera, respectively
                    * 'img_p' (list of 2 numpy.ndarray): the image points to each calibration images on the left and right camera, respectively
                    * 'R' (list of 2 numpy.ndarray): the 3x3 rotation matrices from the left camera and right camera, respectively, to the left camera. (so the first is identity matrix)
                    * 'T' (list of 2 numpy.ndarray): the 3-element translation vectors from the left camera and right camera, respectively, to the left camera. (so the first is [0, 0, 0])
    """
    # -- get paths to all calibration images --
    l_path, r_path = load_image_paths(img_folder)
    print("--> Totally %d images loaded for each camera" % (len(l_path)))

    calib_dict = {"cmtx": [], "dist": [], "img_p": [], "R": [], "T": []} # this is the returned calibration results
    
    # -- get basic properties of the calibration images --
    sample = cv2.imread(l_path[0])
    width = sample.shape[1]
    height = sample.shape[0]

    # -- calibrate the intrinsic of the left camera --
    print("--> Calibrating left camera")
    cmtx1, dist1, img_p1, obj_p = single_camera_calibration(row, col, l_path, tile_size, verbose)
    calib_dict["cmtx"].append(cmtx1)
    calib_dict["dist"].append(dist1)
    calib_dict["img_p"].append(img_p1)

    # -- calibrate the intrinsic of the right camera --
    print("--> Calibrating right camera")
    cmtx2, dist2, img_p2, _ = single_camera_calibration(row, col, r_path, tile_size, verbose)
    calib_dict["cmtx"].append(cmtx2)
    calib_dict["dist"].append(dist2)
    calib_dict["img_p"].append(img_p2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    rmse, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(obj_p, calib_dict["img_p"][LEFT], calib_dict["img_p"][RIGHT], calib_dict["cmtx"][LEFT], calib_dict["dist"][LEFT],
                                                                 calib_dict["cmtx"][RIGHT], calib_dict["dist"][RIGHT], (width, height), criteria = criteria, flags = stereocalibration_flags)

    # -- log the stereo calibration results --
    print('--> Stereo calibration results:')
    print('--> \tReprojection RMSE: ', rmse)
    print('--> \tRotaion matrix:\n', R)
    print('--> \tTranslation:\n', T)

    calib_dict["R"].append(np.eye(3, dtype=np.float32))
    calib_dict["T"].append(np.array([[0], [0], [0]]))
    calib_dict["R"].append(R)
    calib_dict["T"].append(T)

    return calib_dict

def single_camera_calibration(row, col, img_folder, tile_size, verbose):
    """Calibrate a single camera's intrinsic.

    Args:
        row (int): the number of rows on the chessboard
        col (int): the number of columns on the chessboard
        img_folder (str): the path to the parent folder containing calibration images (specifically, the 'left' and 'right' subfolder)
        tile_size (int): the size of the tile on the chessboard pattern, specified in mm
        verbose (bool): True to show detected corners, False otherwise

    Returns:
        cmtx (numpy.ndarray): the 3x3 intrinsic matrix
        dist (numpy.ndarray): the 5-element distortion coefficient
        img_p (numpy.ndarray): the detected chessboard corners
        obj_p (numpy.ndarray): the object points (in the world coordinates) of the chessboard corners
    """
    img_size = cv2.imread(img_folder[0]).shape
    print("--> Calibration image size:", img_size)

    img_p, obj_p = [], []
    for idx, l_p in enumerate(img_folder):
        obj_pnts = get_obj_pnts(row, col) * tile_size
        l_im = cv2.imread(l_p)
        img_pnts, img_with_img_pnts = get_img_pnts(l_im, row, col, (5, 5))
        
        if verbose:
            cv2.imshow('corners', img_with_img_pnts)
            
            # -- press 's' to discard a sample --
            if cv2.waitKey(0) == ord('s'):
                print('--> Skipped image sample')
                continue
            
        if img_pnts is None:
            print("--> Find corner failed for this image")
            continue

        img_p.append(img_pnts)
        obj_p.append(obj_pnts)

    cv2.destroyAllWindows()
    print("--> Obj points shape:", obj_p[0].shape, "img points shape:", img_p[0].shape)
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, (img_size[1], img_size[0]), None, None)

    print("\n#####################################################################\nRMSE re-projection error: %.8f\n" % (ret))
    print("intrinsic camera matrix:\n", cmtx)
    print('#####################################################################\n')

    return cmtx, dist, img_p, obj_p

if __name__ == "__main__":
    parser = ArgumentParser(description="stereo calibration")

    parser.add_argument('--img_folder', type=str, required=True, help="the parent folder to the calibration images, this folder should contain two subfolders named 'left' and 'right' containing the respective calibration images.")
    parser.add_argument('-r', type=int, required=True, dest='row', help="the number of rows on the chessboard pattern.")
    parser.add_argument('-c', type=int, required=True, dest='col', help="the number of columns on the chessboard pattern.")
    parser.add_argument('--tile_size', type=float, required=True, help="the size of each tile on the chessboard patter, specified in mm.")
    parser.add_argument('--save_dir', type=str, required=True, help="the directory saving the calibration results.")
    parser.add_argument('--npy_file', type=str, required=False, default="", help="the name of the calibration result file.")
    parser.add_argument('--camera_name', type=str, required=True, help="the name of the camera, used for naming the calibration result files if no specific name is indicated to name the resulting file.")
    parser.add_argument("--inspect", action="store_true", default=False, help="True to inspect the corner detecton picture by picture, False otherwise, that is, skip the inspection process")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print("creating folder", args.save_dir)
        os.makedirs(args.save_dir)

    # -- to avoid name clash --
    i = 0
    while os.path.exists(os.path.join(args.save_dir, args.camera_name + '_' + str(i) + '.npy')):
        i += 1
    
    # -- give a proper name to the calibration result file --
    if args.npy_file == "":
        args.npy_file = args.camera_name + '_' + str(i) + '.npy'
    elif ".npy" not in args.npy_file:
        args.npy_file += ".npy"
    args.npy_file = os.path.join(args.save_dir, args.npy_file)
    print("--> Will save calibration results to:", args.npy_file)
        
    # -- perform stereo calibration --
    result_dict = do_stereo_calibration(args.row, args.col, args.img_folder, args.tile_size, verbose=args.inspect)
    result_dict.pop("img_p") # we don't want to save the image points as a part of calibration results

    # -- save calibration results as a numpy file --
    print("--> Writing calibration results ...")
    np.save(args.npy_file, result_dict)

    # -- load the written calibration parameters for check --
    back = np.load(args.npy_file, allow_pickle=True)
    for k in back.item().keys():
        print(k)
        print(back.item().get(k))
    