from image import Image
from functions import *
from statistics import stdev, mean, median
from math import inf
from matplotlib import pyplot as plt
import cv2
import numpy as np


def rotate_neck_picture(image):
    """
    Rotating the picture so that the neck of the guitar is horizontal. We use Hough transform to detect lines
    and calculating the slopes of all lines, we rotate it according to the median slope.
    Hopefully, most lines detected will be strings or neck lines so the median slope is the slope of the neck
    An image with lots of noise and different lines will result in poor results.
    :param image: an Image object
    :return rotated_neck_picture: an Image object rotated according to the angle of the median slope detected in param image
    """
    image_to_rotate = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    lines = image.lines_hough_transform(edges, 50, 50)  # TODO: Calibrate params automatically
    size = len(lines)
    slopes = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append(abs((y2 - y1) / (x2 - x1)))

    mean_slope = mean(slopes)
    median_slope = median(slopes)
    std_slope = stdev(slopes)

    angle = median_slope*45

    return Image(img=rotate(image_to_rotate, -angle))


def crop_neck_picture(image):
    """
    Cropping the picture so we only work on the region of interest (i.e. the neck)
    We're looking for a very dense region where we detect horizontal line
    Currently, we identify it by looking at parts where there are more than two lines at the same y (height)
    :param image: an Image object of the neck (rotated horizontally if necessary)
    :return cropped_neck_picture: an Image object cropped around the neck
    """
    image_to_crop = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    lines = image.lines_hough_transform(edges, 50, 50)  # TODO: Calibrate params automatically
    size = len(lines)
    y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(image_to_crop, (x1, y1), (x2, y2), (0, 255, 0), 2)
            y.append(y1)
            y.append(y2)

    y_sort = list(sorted(y))
    y_differences = [0]

    first_y = 0
    last_y = inf

    for i in range(len(y_sort)-1):
        y_differences.append(y_sort[i+1]-y_sort[i])
    for i in range(len(y_differences)-1):
        if y_differences[i] == 0:
            last_y = y_sort[i]
            if i != 0 and first_y == 0:
                first_y = y_sort[i]

    return Image(img=image_to_crop[first_y:last_y])


if __name__ == "__main__":
    chord_image = Image(path="./pictures/chordG.jpg")
    rotated_image = rotate_neck_picture(chord_image)
    for i in range(10):
        rotated_image = rotate_neck_picture(rotated_image)
    cropped_image = crop_neck_picture(rotated_image)
    cropped_image.print_plt()
