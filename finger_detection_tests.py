import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from finger_detection import hand_detection
import cv2


def hand_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)

        hand = hand_detection(cropped_image)

        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB))

    plt.show()

if __name__ == "__main__":
    hand_detection_tests()
