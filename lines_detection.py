from image import Image
from functions import *
from statistics import stdev, mean
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
print(lines)
coefficients = []

for x in range(len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        coefficients.append(abs((x1-y1)/(x2-y2)))

print(coefficients)
coefficients = sorted(set(coefficients))
print(coefficients)

for x in range(len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        if abs((x1-y1)/(x2-y2)) == 12.75:
            print(lines)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


# plt.imshow(img)
# plt.show()
