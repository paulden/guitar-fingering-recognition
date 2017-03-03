from image import Image
from functions import *
from matplotlib import pyplot as plt
import cv2
import numpy as np

chord_bm = Image("./pictures/chordBm.jpg")
img = chord_bm.image
gray = chord_bm.gray


# Guitar strings

edges = chord_bm.edges_sobely()
edges = threshold(edges, 127)

minLineLength = 50
maxLineGap = 50
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength, maxLineGap)

for x in range(len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(img)
plt.show()
