import os
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from finger_detection import hand_detection, locate_hand_region, skin_detection
import cv2


def hand_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("Fichier trouvé : " + filename + " - Traitement en cours...")
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
        plt.imshow(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB))
        print("Terminé !")

    plt.show()

if __name__ == "__main__":
    hand_detection_tests()
