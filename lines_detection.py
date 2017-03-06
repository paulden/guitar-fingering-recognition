from image import Image
from functions import *
from statistics import stdev, mean, median
from math import inf
from matplotlib import pyplot as plt
import cv2
import numpy as np

chord_bm = Image("./pictures/chordG.jpg")
img = chord_bm.image
gray = chord_bm.gray


# Rotating picture

edges = chord_bm.edges_sobely()
edges = threshold(edges, 127)

lines = chord_bm.lines_hough_transform(edges, 50, 50)
size = len(lines)
slopes = []

for x in range(size):
    for x1, y1, x2, y2 in lines[x]:
        slopes.append(abs((y2 - y1) / (x2 - x1)))

mean_slope = mean(slopes)
median_slope = median(slopes)
std_slope = stdev(slopes)

angle = mean_slope*45
img = rotate(img, -angle)
chord_bm.set_image(img)
chord_bm.set_gray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


# Cropping picture

min_x = inf
min_y = inf
max_x = 0
max_y = 0

edges = chord_bm.edges_sobely()
edges = threshold(edges, 127)

lines = chord_bm.lines_hough_transform(edges, 50, 50)
size = len(lines)
slopes = []

for x in range(size):
    for x1, y1, x2, y2 in lines[x]:
        if abs((y2-y1)/(x2-x1)) < std_slope/4:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if x1 < min_x:
                min_x = x1
            if x2 < min_x:
                min_x = x2
            if y1 < min_y:
                min_y = y1
            if y2 < min_y:
                min_y = y2
            if x1 > max_x:
                max_x = x1
            if x2 > max_x:
                max_x = x2
            if y1 > max_y:
                max_y = y1
            if y2 > max_y:
                max_y = y2


# print(min_x, max_x, min_y, max_y)
plt.imshow(img[min_y:max_y, min_x:max_x])
plt.show()
