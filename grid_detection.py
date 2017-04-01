from strings import Strings
from rotate_crop import *
import cv2
import numpy as np
from collections import defaultdict


def string_detection(neck):
    """
    Detecting and separating strings into separate blocks by choosing numerous vertical slices in image
    We then look for a periodical pattern on each slice (ie strings detected), store points in between strings
    and connect them using a regression fitting function to separe each string
    :param neck: An Image object of the picture cropped around the horizontal neck
    :return strings, Image_string: either a string object which is a dict associating each string to a line
    (ie a tuple of points) delimiting the bottom of the string block // or directly an Image object with those
    lines displayed (illustration purpose)
    """
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_strings = np.zeros((height, width, 3), np.uint8)

    # 1. Detect strings with Hough transform and form an Image based on these
    edges = neck.edges_sobely()
    edges = threshold(edges, 127)

    lines = neck.lines_hough_transform(edges, 50, 20)  # TODO: Calibrate params automatically if possible
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_strings, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_str = Image(img=neck_with_strings)
    neck_str_gray = neck_str.gray

    # 2. Slice image vertically at different points and calculate gaps between strings at these slices
    slices = {}
    nb_slices = int(width / 50)
    for i in range(nb_slices):
        slices[(i + 1) * nb_slices] = []  # slices dict is {x_pixel_of_slice : [y_pixels_where_line_detected]}

    for index_line, line in enumerate(neck_str_gray):
        for index_pixel, pixel in enumerate(line):
            if pixel == 255 and index_pixel in slices:
                slices[index_pixel].append(index_line)

    slices_differences = {}  # slices_differences dict is {x_pixel_of_slice : [gaps_between_detected_lines]}
    for k in slices.keys():
        temp = []
        n = 0
        slices[k] = list(sorted(slices[k]))
        for p in range(len(slices[k]) - 1):
            temp.append(slices[k][p + 1] - slices[k][p])
            if slices[k][p + 1] - slices[k][p] > 1:
                n += 1
        slices_differences[k] = temp

    points = []
    points_dict = {}
    for j in slices_differences.keys():
        gaps = [g for g in slices_differences[j] if g > 1]
        points_dict[j] = []

        if len(gaps) > 3:
            median_gap = median(gaps)
            for index, diff in enumerate(slices_differences[j]):
                if abs(diff - median_gap) < 4:
                    points_dict[j].append((j, slices[j][index] + int(median_gap / 2)))
                elif abs(diff / 2 - median_gap) < 4:
                    points_dict[j].append((j, slices[j][index] + int(median_gap / 2)))
                    points_dict[j].append((j, slices[j][index] + int(3 * median_gap / 2)))

        points.extend(points_dict[j])

    '''for p in points:
        print(p)
        cv2.circle(neck.image, p, 3, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(neck.image, cv2.COLOR_BGR2RGB))
    plt.show()'''

    points_divided = [[] for i in range(5)]
    for s in points_dict.keys():
        for i in range(5):
            try:
                # cv2.circle(neck.image, points_dict[s][i], 3, (255, 0, 0), -1)
                points_divided[i].append(points_dict[s][i])
            except IndexError:
                pass

    # 3. Use fitLine function to form lines separating each string

    tuning = ["E", "A", "D", "G", "B", "E6"]
    strings = Strings(tuning)

    for i in range(5):
        cnt = np.array(points_divided[i])
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L12, 0, 0.01, 0.01)  # best distType found was DIST_L12

        left_extreme = int((-x * vy / vx) + y)
        right_extreme = int(((width - x) * vy / vx) + y)

        strings.separating_lines[tuning[i]] = [(width - 1, right_extreme), (0, left_extreme)]

        cv2.line(neck.image, (width - 1, right_extreme), (0, left_extreme), (0, 0, 255), 2)

    return strings, Image(img=neck.image)


def fret_detection(neck):
    """
    Detecting frets by detecting vertical components that are potential frets (provided they are lines detected
    by Hough transform and respect a logarithmic ratio)
    :param neck: An Image object of the picture cropped around the horizontal neck
    :return: an Image object of the picture with the frets detected (illustration purpose at the moment)
    """
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_frets = np.zeros((height, width, 3), np.uint8)

    # 1. Detect frets with Hough transform and form an Image based on these
    edges = neck.edges_sobelx()
    edges = threshold(edges, 127)
    # edges = cv2.medianBlur(edges, 3)

    lines = neck.lines_hough_transform(edges, 20, 5)  # TODO: Calibrate params automatically if possible
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_frets, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_fr = Image(img=neck_with_frets)
    neck_fret_gray = neck_fr.gray

    # 2. Slice image horizontally at different points and calculate gaps between frets at these slices
    slices = {}
    nb_slices = int(height / 15)
    for i in range(nb_slices):
        slices[(i + 1) * nb_slices] = []  # slices dict is {y_pixel_of_slice : [x_pixels_where_line_detected]}

    for index_line, line in enumerate(neck_fret_gray):
        for index_pixel, pixel in enumerate(line):
            if pixel == 255 and index_line in slices:
                slices[index_line].append(index_pixel)

    slices_differences = {}  # slices_differences dict is {y_pixel_of_slice : [gaps_between_detected_lines]}
    for k in slices.keys():
        temp = []
        n = 0
        slices[k] = list(sorted(slices[k]))
        for p in range(len(slices[k]) - 1):
            temp.append(slices[k][p + 1] - slices[k][p])
            if slices[k][p + 1] - slices[k][p] > 1:
                n += 1
        slices_differences[k] = temp

    x_values = defaultdict(int)
    for j in slices_differences.keys():
        for index, gap in enumerate(slices_differences[j]):
            if gap > 1:
                x_values[slices[j][index]] += 1

    potential_frets = []
    x_values = dict(x_values)
    for x, nb in x_values.items():
        if nb > 1:
            potential_frets.append(x)

    potential_frets = list(sorted(potential_frets))
    potential_frets = remove_duplicates(potential_frets)

    # 3. Sort potential frets by looking for a ratio and building missings frets
    potential_ratio = []
    for i in range(len(potential_frets) - 1):
        potential_ratio.append(round(potential_frets[i + 1] / potential_frets[i], 3))

    ratio = potential_ratio[-1]
    last_x = potential_frets[-1]
    while 1:
        last_x *= ratio
        if last_x >= width:
            break
        else:
            potential_frets.append(int(last_x))

    for x in potential_frets:
        cv2.line(neck.image, (x, 0), (x, height), (127, 0, 255), 3)

    return Image(img=neck.image)


if __name__ == "__main__":
    print("Run grid_detection_tests.py to have a look at results!")
