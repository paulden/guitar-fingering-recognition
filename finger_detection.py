import cv2
import numpy as np
from matplotlib import pyplot as plt
from image import Image
from cv2 import ximgproc
from rotate_crop import *


def skin_detection(img):
    for index_line, line in enumerate(img):
        for index_pixel, pixel in enumerate(line):
            if pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20 and max(pixel) - min(pixel) > 15 \
                    and abs(pixel[2] - pixel[1]) > 15 and pixel[2] > pixel[0] and pixel[2] > pixel[1]\
                    and index_pixel > len(line)/2:
                # img[index_line][index_pixel] = (255, 255, 255)
                pass
            else:
                img[index_line][index_pixel] = (0, 0, 0)
    return img


def hand_detection(neck):
    neck.set_image(skin_detection(neck.image))
    neck.set_image(cv2.medianBlur(neck.image, 5))
    neck.set_gray(cv2.cvtColor(neck.image, cv2.COLOR_BGR2GRAY))
    canny_edges = neck.edges_canny(min_val=100, max_val=150, aperture=3)

    height = len(neck.image)
    width = len(neck.image[0])
    contour_image = np.zeros((height, width, 3), np.uint8)
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(neck.gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(neck.image, contours, -1, (0, 255, 0), 3)

    '''circles = cv2.HoughCircles(contour_image, cv2.HOUGH_GRADIENT, 1, 5,
                               param1=100, param2=20, minRadius=10, maxRadius=100)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(neck.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(neck.image, (i[0], i[1]), 2, (0, 0, 255), 3)'''

    return neck.image

if __name__ == "__main__":
    chord_image = Image(path="./pictures/chordE.jpg")
    rc_image = crop_neck_picture(rotate_neck_picture(chord_image))
    hand = hand_detection(rc_image)
    plt.imshow(hand)
    plt.show()
