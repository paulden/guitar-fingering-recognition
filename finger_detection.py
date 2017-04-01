from collections import defaultdict
import numpy as np
from rotate_crop import *


def skin_detection(img):
    """
    Naively detecting skin in image. Non-skin will be black (0, 0, 0)
    :param img: an image as defined in OpenCV
    :return: an image as defined in OpenCV
    """
    for index_line, line in enumerate(img):
        for index_pixel, pixel in enumerate(line):
            if pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20 and max(pixel) - min(pixel) > 15 \
                    and abs(pixel[2] - pixel[1]) > 15 and pixel[2] > pixel[0] and pixel[2] > pixel[1] \
                    and index_pixel > len(line) / 2:
                # img[index_line][index_pixel] = (255, 255, 255)
                pass
            else:
                img[index_line][index_pixel] = (0, 0, 0)
    return img


def locate_hand_region(img):
    """
    Refining hand region after skin detection by returning the region with the highest density
    of non-black pixel when looking at regions split vertically
    :param img: an image as defined in OpenCV, after skin detection
    :return: an image as defined in OpenCV
    """
    height = len(img)
    width = len(img[0])
    hand_region = np.zeros((height, width, 3), np.uint8)

    x_dict = defaultdict(int)
    # x_values = []
    for line in img:
        for j, pixel in enumerate(line):
            if pixel.all() > 0:
                # x_values.append(j)
                x_dict[j] += 1

    # plt.hist(x_values, bins=200)
    # plt.show()

    max_density = max(x_dict.values())
    max_x_density = 0
    for x, density in x_dict.items():
        if density == max_density:
            max_x_density = x
            break
    min_x = min(x_dict.keys())
    max_x = max(x_dict.keys())

    m = 0
    last_density = x_dict[max_density]
    while 1:
        if max_x_density - m == min_x:
            break
        m += 1
        current_density = x_dict[max_x_density - m]
        if current_density < 0.1 * max_density:
            break
        elif current_density < 0.5 * last_density:
            break
        last_density = current_density

    n = 0
    last_density = x_dict[max_density]
    while 1:
        if max_x_density + n == max_x:
            break
        n += 1
        current_density = x_dict[max_x_density + n]
        if current_density < 0.1 * max_density:
            break
        elif current_density < 0.5 * last_density:
            break
        last_density = current_density

    tolerance = 20
    min_limit = max_x_density - m - tolerance
    max_limit = max_x_density + n + tolerance

    for i, line in enumerate(img):
        for j, pixel in enumerate(line):
            if min_limit < j < max_limit:
                hand_region[i][j] = img[i][j]

    return hand_region


def hand_detection(skin):
    """
    Detecting contours in hand using Canny detection
    Also trying to find fingertips (currently not working...)
    :param img: an image as defined in OpenCV, after skin detection and refining
    :return: an image as defined in OpenCV, you may choose whether you want the contours or the results
    of the circular Hough transform (which is not working as of 02/04/2017...)
    """
    neck = Image(img=skin)
    neck.set_image(locate_hand_region(skin_detection(neck.image)))
    neck.set_image(cv2.medianBlur(neck.image, 5))
    neck.set_gray(cv2.cvtColor(neck.image, cv2.COLOR_BGR2GRAY))
    canny_edges = neck.edges_canny(min_val=70, max_val=100, aperture=3)

    # height = len(neck.image)
    # width = len(neck.image[0])
    # contour_image = np.zeros((height, width, 3), np.uint8)
    # contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    #
    # ret, thresh = cv2.threshold(neck.gray, 127, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     perimeter = cv2.arcLength(cnt, False)
    #     if perimeter > 100:
    #         cv2.drawContours(contour_image, cnt, -1, (255, 255, 255), 3)
    #
    # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)

    circles = cv2.HoughCircles(canny_edges, cv2.HOUGH_GRADIENT, 1, 5,
                               param1=100, param2=20, minRadius=20, maxRadius=90)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(neck.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(neck.image, (i[0], i[1]), 2, (0, 0, 255), 3)

    return cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR), neck.image


def refine_hand_region(neck, skin):
    """
    Deprecated - Useless
    Refining hand region by dividing image into squares and keeping squares where no string is detected
    :param neck: An Image object of the picture cropped around the horizontal neck
    :param skin: an image as defined in OpenCV, after skin detection
    :return: an image as defined in OpenCV refining around the hand
    """
    # 1. We want to check where string lines are detected. If none are found, it may mean the hand hides them
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_strings = np.zeros((height, width, 3), np.uint8)

    edges = neck.edges_sobely()
    edges = threshold(edges, 127)

    lines = neck.lines_hough_transform(edges, 50, 20)  # TODO: Calibrate params automatically
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_strings, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_str = Image(img=neck_with_strings)
    neck_str_gray = neck_str.gray

    # 2. We divide neck image in square of 40*40px. If we detect string line going through, we rule it out
    square_size = 40
    x_nb = width // square_size
    y_nb = height // square_size

    for i in range(y_nb):
        for j in range(x_nb):
            lines_in_square_left = 0
            lines_in_square_right = 0
            skin_in_square_below = 0
            for k in range(i * square_size, min((i + 1) * square_size, height)):
                if neck_str_gray[k][j * square_size] > 0:
                    lines_in_square_left += 1
                if neck_str_gray[k][min((j + 1) * square_size, width - 1)] > 0:
                    lines_in_square_right += 1
            for l in range(j * square_size, min((j + 1) * square_size, width)):
                if skin[min((i + 1) * square_size, height - 1)][l].any() > 0:
                    skin_in_square_below += 1
            if lines_in_square_left > 1 and lines_in_square_right > 1:
                for k in range(i * square_size, min((i + 1) * square_size, height)):
                    for l in range(j * square_size, min((j + 1) * square_size, width)):
                        skin[k][l] = (0, 0, 0)

    return skin


if __name__ == "__main__":
    print("Run finger_detection_tests.py to have a look at results!")
