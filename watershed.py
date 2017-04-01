import numpy as np
from matplotlib import pyplot as plt
from rotate_crop import *
from image import Image


def watershed_segmentation(img):
    """
    Source : https://learndeltax.blogspot.fr/2016/02/segmentation-using-cannywatershed-in.html
    As of today, watershed segmentation is not used. I was just trying out new ways to detect hand.
    :param img: an image as defined in OpenCV
    :return: simply plotting result for illustration purpose
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = cv2.erode(thresh, None, iterations=1)
    bgt = cv2.dilate(thresh, None, iterations=1)

    ret, bg = cv2.threshold(bgt, 1, 128, 1)

    marker = cv2.add(fg, bg)
    canny = cv2.Canny(marker, 110, 150)

    new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    marker32 = np.int32(marker)
    cv2.watershed(img, marker32)
    m = cv2.convertScaleAbs(marker32)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    res = cv2.bitwise_and(img, img, mask=thresh)
    res3 = cv2.bitwise_and(img, img, mask=thresh_inv)
    res4 = cv2.addWeighted(res, 1, res3, 1, 0)
    final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)

    perimeter_array = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, False)
        perimeter_array.append(perimeter)

    sorted_data = sorted(zip(perimeter_array, contours), key=lambda x: x[0], reverse=True)

    for j in range(1):  # change here from 1 to len(sorted_data) in range to get more or less contours detected
        final = cv2.drawContours(res4, sorted_data[j][1], -1, (0, 255, 0), 3)

    plt.imshow(final)
    plt.show()


if __name__ == "__main__":
    chord_image = Image(path="./pictures/chordAm.png")
    rc = crop_neck_picture(rotate_neck_picture(chord_image)).image
    watershed_segmentation(cv2.cvtColor(rc, cv2.COLOR_BGR2RGB))
