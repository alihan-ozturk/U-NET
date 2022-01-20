import numpy as np
from PIL import Image
import os

path = r"C:\Users\RoboGor\Desktop\seg\img"

denominator = len(os.listdir(path)) * 255

rgbM = np.array([.0, .0, .0])
rgbStd = np.array([.0, .0, .0])

for img_name in os.listdir(path):
    abs_path = os.path.join(path, img_name)
    img = np.array(Image.open(abs_path))
    rgbM += np.array([img[:, :, i].mean() for i in range(3)])
    rgbStd += np.array([img[:, :, i].std() for i in range(3)])

print(rgbM / denominator)
print(rgbStd / denominator)
