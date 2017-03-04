def threshold(img, s):
    I = img
    I[I <= s] = 0
    I[I > s] = 255
    return I
