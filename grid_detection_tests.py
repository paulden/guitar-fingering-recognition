import os
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture
from grid_detection import string_detection, fret_detection


def string_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        chord_image = Image(path='./pictures/'+filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        try:
            neck_string = string_detection(cropped_image)
        except IndexError:
            pass
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(chord_image.image)
        plt.subplot(int("42" + str(i)))
        i += 1
        try:
            plt.imshow(neck_string.image)
        except UnboundLocalError:
            pass

    plt.show()


if __name__ == "__main__":
    string_detection_tests()