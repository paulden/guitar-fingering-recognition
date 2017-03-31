import os
import time
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from grid_detection import string_detection, fret_detection
import cv2


def string_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_string = string_detection(cropped_image)[1]
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_string.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()


def fret_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_fret = fret_detection(cropped_image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_fret.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()


def grid_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_strings = string_detection(cropped_image)[0]
        neck_fret = fret_detection(cropped_image)
        for string, pts in neck_strings.separating_lines.items():
            cv2.line(neck_fret.image, pts[0], pts[1], (127, 0, 255), 2)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_fret.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()


if __name__ == "__main__":
    # string_detection_tests()
    # fret_detection_tests()
    grid_detection_tests()
