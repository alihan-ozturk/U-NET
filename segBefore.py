import PIL.Image
import numpy as np
from cv2 import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

path = r"C:\Users\RoboGor\Desktop\seg\inverseImg"

imgShape = np.array([1088, 1920])
cropArea = np.array([1000, 1000])
w, h = imgShape - cropArea

size = (40, 40)
area = size[0] * size[1]

kernel = np.ones(size)

for img_name in os.listdir(path):
    abs_path = os.path.join(path, img_name)
    img = np.array(Image.open(abs_path).convert("L"))
    w_, h_ = np.random.randint(0, w), np.random.randint(0, h)
    img = img[w_:1000 + w_, h_:1000 + h_]
    new = cv2.erode(img, kernel)
    d = dict()
    num_labels, labels = cv2.connectedComponents(new, connectivity=8)
    for j in range(1, num_labels):
        temp = labels == j
        temSum = temp.sum()
        if temSum < area:
            labels[temp] = 0
        else:
            d[j] = temSum
    print(num_labels)
    plt.imshow(labels, cmap="gray")
    plt.show()
    input("bas")
