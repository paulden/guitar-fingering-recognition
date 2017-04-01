import cv2


def threshold(img, s):
    I = img
    I[I <= s] = 0
    I[I > s] = 255
    return I


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def remove_duplicates(l):
    gaps = []
    new_l = []
    for i in range(len(l) - 1):
        gaps.append(l[i + 1] - l[i])
    for index, g in enumerate(gaps):
        if g > 15:
            new_l.append(l[index])
    return new_l
