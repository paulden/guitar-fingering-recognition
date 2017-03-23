from image import Image
from strings import Strings
from functions import *
from statistics import median, StatisticsError
from random import random
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

    lines = neck.lines_hough_transform(edges, 50, 50)  # TODO: Calibrate params automatically
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_strings, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_str = Image(img=neck_with_strings)
    neck_str_gray = neck_str.gray

    slices = {}
    nb_slices = int(width / 50)
    for i in range(nb_slices):
        slices[(i+1)*nb_slices] = []

    for index_line, line in enumerate(neck_str_gray):
        for index_pixel, pixel in enumerate(line):
            if pixel == 255 and index_pixel in slices:
                slices[index_pixel].append(index_line)

    slices_differences = {}
    for k in slices.keys():
        temp = []
        n = 0
        slices[k] = list(sorted(slices[k]))
        for p in range(len(slices[k])-1):
            temp.append(slices[k][p+1]-slices[k][p])
            if slices[k][p+1]-slices[k][p] > 1:
                n += 1
        slices_differences[k] = temp

    points = []
    for j in slices_differences.keys():
        gaps = []
        for l in range(len(slices_differences[j])):
            if slices_differences[j][l] > 1:
                gaps.append(slices_differences[j][l])
        try:
            median_gap = median(gaps)
            for gap in gaps:
                if abs(gap-median_gap) > 3:  # TODO: Relax condition on difference if no convenient gaps are found
                    gaps.remove(gap)
            if len(gaps) == 5:
                    for p in range(len(slices[j])-1):
                        current_gap = slices[j][p+1]-slices[j][p]
                        if abs(current_gap - median_gap) <= 3:
                            points.append((j, int(slices[j][p] + current_gap/2)))
        except StatisticsError:
            pass

    strings = Strings(['E', 'a', 'd', 'g', 'b', 'e'])

    for i in range(5):  # TODO: Manage errors if we don't have 5 gaps
        a = (points[i+5][1]-points[i][1])/(points[i+5][0]-points[i][0])
        b = points[i][1] - a*points[i][0]
        current_block = strings.tuning[i]
        strings.blocks[current_block] = [(0, int(b)), (width-1, int(a*(width-1)+b))]
        red = random()*255
        blue = random() * 255
        green = random() * 255
        cv2.line(neck.image, (0, int(b)), (width-1, int(a*(width-1)+b)), (red, green, blue), 2)

    strings.blocks[strings.tuning[5]] = [(0, height), (width, height)]

    # return Image(img=neck.image)
    return strings


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
    print(neck_string)
    neck_fret = fret_detection(cropped_image)
    # neck_grid = Image(img=(neck_string.image + neck_fret.image))
    # neck_grid.print_plt(is_gray=False)
    # neck_fret.print_plt(is_gray=False)
