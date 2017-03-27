import numpy as np
from image import Image
import cv2
from matplotlib import pyplot as plt
from rotate_crop import *
from finger_detection import skin_detection


def kmeans_segmentation(img):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    plt.imshow(res2)
    plt.show()

if __name__ == "__main__":
    chord_image = Image(path="./pictures/chordAm.png")
    rc_image = crop_neck_picture(rotate_neck_picture(chord_image))
    skin = skin_detection(rc_image.image)
    kmeans_segmentation(skin)