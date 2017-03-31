import os
import time
import cv2
from matplotlib import pyplot as plt
from image import Image
from rotate_crop import rotate_neck_picture, crop_neck_picture, resize_image

i = 1
plt.figure(1)
for filename in os.listdir('./pictures/'):
    print("File found: " + filename + " - Processing...")
    start_time = time.time()
    chord_image = Image(path='./pictures/' + filename)
    resized = resize_image(chord_image.image)
    new = Image(img=resized)
    rotated_image = rotate_neck_picture(new)
    cropped_image = crop_neck_picture(rotated_image)
    plt.subplot(int("42" + str(i)))
    i += 1
    plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
    plt.subplot(int("42" + str(i)))
    i += 1
    plt.imshow(cv2.cvtColor(cropped_image.image, cv2.COLOR_BGR2RGB))
    print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

plt.show()
