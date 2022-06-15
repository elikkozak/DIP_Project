import cv2
import numpy as np


def get_contour_from_path(path):
    image = cv2.imread(path, 0)

    # convert to Binary image
    ret, thresh = cv2.threshold(image, 200, 255, 1)
    # get Contours
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # get largest Contour
    contours_sizes = [(cv2.contourArea(cnt), cnt) for cnt in contours]

    biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]
    return biggest_contour


def get_contour_from_image(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to Binary image
    ret, thresh = cv2.threshold(image, 200, 255, 0)
    # get Contours
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # get largest Contour
    contours_sizes = [(cv2.contourArea(cnt), cnt) for cnt in contours]
    if len(contours_sizes) == 0:
        return None
    biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]
    return biggest_contour


def convert_contour_to_img(contour, img):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255),
                     -1)
    return mask
