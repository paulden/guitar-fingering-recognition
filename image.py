import numpy as np
from matplotlib import pyplot as plt
import cv2


class Image:
    def __init__(self, path):
        self.image = cv2.imread(path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __str__(self):
        return self.image

    def print_cv2(self, is_gray=True):
        if is_gray:
            cv2.imshow('image', self.gray)
        else:
            cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print_plt(self, is_gray=True):
        if is_gray:
            plt.imshow(self.gray, cmap='gray')
        else:
            plt.imshow(self.image)
        plt.show()

    def edges_canny(self, min_val=100, max_val=200, aperture=3):
        return cv2.Canny(self.gray, min_val, max_val, aperture)

    def edges_laplacian(self, ksize=3):
        return cv2.Laplacian(self.gray, cv2.CV_8U, ksize)

    def edges_sobelx(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 1, 0, ksize)

    def edges_sobely(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 0, 1, ksize)

