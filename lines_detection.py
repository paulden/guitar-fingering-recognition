from image import Image
from functions import *
from statistics import stdev, mean, median
from matplotlib import pyplot as plt
import cv2
import numpy as np

chord_bm = Image("./pictures/chordBm.jpg")
img = chord_bm.image
gray = chord_bm.gray


# Guitar strings

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

for x in range(size):
    for x1, y1, x2, y2 in lines[x]:
        if abs(abs((y2-y1)/(x2-x1)) - mean_slope) < std_slope/2:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


plt.imshow(img)
plt.show()
