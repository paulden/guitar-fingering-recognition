import cv2
import numpy as np
from matplotlib import pyplot as plt
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


def refine_hand_region(neck, skin):

    # 1. We want to check where string lines are detected. If none are found, it may mean the hand hides them
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_strings = np.zeros((height, width, 3), np.uint8)

    edges = neck.edges_sobely()
    edges = threshold(edges, 127)

    lines = neck.lines_hough_transform(edges, 50, 20)  # TODO: Calibrate params automatically
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_strings, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_str = Image(img=neck_with_strings)
    neck_str_gray = neck_str.gray

    # 2. We divide neck image in square of 50*50px. If we detect string line going through, we rule it out
    square_size = 40
    x_nb = width // square_size
    y_nb = height // square_size

    for i in range(y_nb):
        for j in range(x_nb):
            lines_in_square_left = 0
            lines_in_square_right = 0
            skin_in_square_below = 0
            for k in range(i*square_size, min((i+1)*square_size, height)):
                if neck_str_gray[k][j * square_size] > 0:
                    lines_in_square_left += 1
                if neck_str_gray[k][min((j + 1) * square_size, width-1)] > 0:
                    lines_in_square_right += 1
            for l in range(j*square_size, min((j+1)*square_size, width)):
                if skin[min((i+1)*square_size, height-1)][l].any() > 0:
                    skin_in_square_below += 1
            if lines_in_square_left > 1 and lines_in_square_right > 1:
                for k in range(i * square_size, min((i + 1) * square_size, height)):
                    for l in range(j * square_size, min((j + 1) * square_size, width)):
                        skin[k][l] = (0, 0, 0)

    return skin


def hand_detection(neck):
    neck.set_image(skin_detection(neck.image))
    neck.set_image(cv2.medianBlur(neck.image, 5))
    neck.set_gray(cv2.cvtColor(neck.image, cv2.COLOR_BGR2GRAY))
    canny_edges = neck.edges_canny(min_val=70, max_val=100, aperture=3)

    height = len(neck.image)
    width = len(neck.image[0])
    contour_image = np.zeros((height, width, 3), np.uint8)
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(neck.gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    '''for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, False)
        if perimeter > 100:
            cv2.drawContours(contour_image, cnt, -1, (255, 255, 255), 3)'''

    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)

    '''circles = cv2.HoughCircles(contour_image, cv2.HOUGH_GRADIENT, 1, 5,
                               param1=100, param2=20, minRadius=20, maxRadius=90)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(neck.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(neck.image, (i[0], i[1]), 2, (0, 0, 255), 3)'''

    # return cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    return neck.image


if __name__ == "__main__":
    chord_image = Image(path="./pictures/chordAm.png")
    rc_image = crop_neck_picture(rotate_neck_picture(chord_image))
    new = refine_hand_region(rc_image, skin_detection(rc_image.image))
    # hand = hand_detection(rc_image)
    plt.imshow(new)
    plt.show()
