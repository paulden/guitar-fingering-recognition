from image import Image
from functions import *
from statistics import stdev, mean, median
from rotate_crop import *
from matplotlib import pyplot as plt
import cv2
import numpy as np


def string_detection(neck):
    """

    :param neck: An Image object of the picture cropped around the horizontal neck
    :return:
    """
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

    return Image(img=neck_with_strings)


def fret_detection(neck):
    """

    :param neck: An Image object of the picture cropped around the horizontal neck
    :return:
    """
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_strings = np.zeros((height, width, 3), np.uint8)

    edges = neck.edges_sobelx()
    edges = threshold(edges, 127)

    lines = neck.lines_hough_transform(edges, 20, 5)  # TODO: Calibrate params automatically
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_strings, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return Image(img=neck_with_strings)


if __name__ == "__main__":
    chord_image = Image(path="./pictures/chordBm.jpg")
    rotated_image = rotate_neck_picture(chord_image)
    for i in range(10):
        rotated_image = rotate_neck_picture(rotated_image)
    cropped_image = crop_neck_picture(rotated_image)
    neck_string = string_detection(cropped_image)
    neck_fret = fret_detection(cropped_image)
    neck_grid = Image(img=(neck_string.image + neck_fret.image))
    neck_grid.print_plt(is_gray=False)
