
import cv2
import sys
from multiprocessing import Pool
import mediapipe as mp
import numpy as np
import contourHelper as cH
import poseModule as pM
from itertools import repeat

if __name__ == '__main__':
    # convert image to contour
    character_contour = sys.argv[1]

    # get mediapipe ready
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    # get video clip ready
    video_path = sys.argv[2]

    # pM.getPose(video_path, True) is list to images that we convert to list of contours
    # for each contour we use cv2.matchShapes which compares two contours the lower the
    # value the closer the contours are.
    with Pool(4) as p:
        list_of_contours = p.map(cH.get_contour_from_image, pM.getPose(video_path, True))
        list_of_matched_shape = p.starmap(cv2.matchShapes,
                                          zip(list_of_contours, repeat(character_contour, len(list_of_contours)),
                                              repeat(1, len(list_of_contours)), repeat(0.0, len(list_of_contours))))

    np_arr = np.argsort(np.array(list_of_matched_shape))[:1]
    print(np_arr)


