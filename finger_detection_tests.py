import os
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from finger_detection import hand_detection, locate_hand_region, skin_detection
import cv2
import time


def hand_detection_tests(b):
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)

        skin = skin_detection(cropped_image.image)
        refined_hand_region = locate_hand_region(skin)
        hand = hand_detection(refined_hand_region)

        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(hand[b], cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()

if __name__ == "__main__":
    print("What would you like to get? \n\t1 - Contours \n\t2 - Circular Hough transform")
    choice = input("[1/2] > ")
    if choice == "1":
        print("Detecting contours...")
        hand_detection_tests(0)
    elif choice == "2":
        print("Detecting circular Hough transform results...")
        hand_detection_tests(1)
    else:
        print("Command not defined - Aborted.")
