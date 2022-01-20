import numpy as np
from cv2 import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

IMAGE_DIR = r"C:\Users\RoboGor\Desktop\seg\img"
MASK_DIR = r"C:\Users\RoboGor\Desktop\seg\inverseImg"

augmentations = A.Compose([
    A.RandomCrop(width=1000, height=1000),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Resize(400, 400),
    ToTensorV2()
])


class aerialDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_names = os.listdir(img_path)
        self.mask_names = os.listdir(mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_names[idx])
        mask_path = os.path.join(self.mask_path, self.mask_names[idx])
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            transform = self.transform(image=image, mask=mask)
            image = transform["image"]
            mask = transform["mask"]

        image = image / 255.0

        return image, mask


BATCH_SIZE = 1

data = aerialDataset(IMAGE_DIR, MASK_DIR, transform=augmentations)
robogor_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def func(img, d):
    w, h = img.shape
    midPoint = ((w - 1) / 2, (h - 1) / 2)
    iterW = 4
    iterH = 2

    nloop = int((max(w, h) - 2) / 2)
    nIncrease = int((min(w, h) - 2) / 2)

    upCells = []
    downCells = []
    rightCells = []
    leftCells = []

    stop = 0

    up = int(w / 2 - 2)
    down = int(w / 2 + 1)
    left = int(h / 2 - 2)
    right = int(h / 2 + 1)

    for k in range(1, nloop + 1):
        for i in range(iterW):
            cellUp = (up, left + i)
            cellDown = (down, left + i)
            cell1 = img[cellUp]
            cell2 = img[cellDown]
            if cell1 != 0:
                upCells.append(func2(midPoint, cellUp, d, cell1))
                stop = 1
            if cell2 != 0:
                downCells.append(func2(midPoint, cellDown, d, cell2))
                stop = 1

        for j in range(iterH):
            cellLeft = (up + j + 1, left)
            cellRight = (up + j + 1, right)
            cell1 = img[cellLeft]
            cell2 = img[cellRight]
            if cell1 != 0:
                leftCells.append(func2(midPoint, cellLeft, d, cell1))
                stop = 1
            if cell2 != 0:
                rightCells.append(func2(midPoint, cellRight, d, cell2))
                stop = 1

        if stop:
            return np.sum(upCells), np.sum(rightCells), np.sum(downCells), np.sum(leftCells)
        else:
            if k >= nIncrease:
                if w < h:
                    iterW = 0
                    left -= 1
                    right += 1
                else:
                    iterH = 0
                    up -= 1
                    down += 1
            else:
                iterW += 2
                iterH += 2
                up -= 1
                down += 1
                left -= 1
                right += 1
    return 0, 0, 0, 0


def func2(homePoint, targetPoint, d, label):
    x, y = homePoint
    x_, y_ = targetPoint
    dist = (x - x_) ** 2 + (y - y_) ** 2
    return d[label] / dist


size = (40, 40)
area = size[0] * size[1]
actionSpaces = {0: "up", 1: "right", 2: "down", 3: "left", 4: "land"}
kernel = np.ones(size)

for i, (image, mask) in enumerate(robogor_loader):
    d = dict()
    mask = mask[0].cpu().numpy()
    new = cv2.erode(mask, kernel)
    num_labels, labels = cv2.connectedComponents(new)
    counter = 0
    for label in range(1, num_labels):
        temp = labels == label
        temSum = temp.sum()
        if temSum < area:
            counter += 1
            labels[temp] = 0
        else:
            label -= counter
            labels[temp] = label
            d[label] = temSum
    # action = np.argmax([actionSum for actionSum in func(labels, d)]
    actions = np.array([actionSum for actionSum in func(labels, d)])
    if np.sum(actions > 0)==4:
        action = 4
    else:
        action = np.argmax(actions)
    print(actionSpaces[action])
    mask[180:-180, 180:-180] = 0
    plt.imshow(np.concatenate((mask / 255, labels), axis=1), cmap="gray")
    plt.show()
    input("bas")
