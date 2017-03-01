import numpy as np
from matplotlib import pyplot as plt
import cv2


def Canny(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def Laplacian(img):
    edges = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    return edges


def Sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    return sobelx, sobely

if __name__ == "__main__":
    img = cv2.imread("./pictures/chordBm.jpg", cv2.IMREAD_GRAYSCALE)
    edgesx, edgesy = Sobel(img)
    plt.imshow(edgesx, cmap='gray')
    plt.show()
    plt.imshow(edgesy, cmap='gray')
    plt.show()