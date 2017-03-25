import os
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from grid_detection import string_detection, fret_detection
import cv2


def string_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_string = string_detection(cropped_image)[1]
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(chord_image.image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(neck_string.image)

    plt.show()


def fret_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_fret = fret_detection(cropped_image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(chord_image.image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(neck_fret.image)

    plt.show()


def grid_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_strings = string_detection(cropped_image)[0]
        neck_fret = fret_detection(cropped_image)
        for string, pts in neck_strings.separating_lines.items():
            cv2.line(neck_fret.image, pts[0], pts[1], (255, 0, 127), 2)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(chord_image.image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(neck_fret.image)

    plt.show()


if __name__ == "__main__":
    # string_detection_tests()
    # fret_detection_tests()
    grid_detection_tests()
