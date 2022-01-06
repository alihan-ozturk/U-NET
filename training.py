import gc
import os
import torch
import warnings
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import f1_score, jaccard_score
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name('cuda')
print(device_name)

warnings.filterwarnings('ignore')

TRAIN_IMAGE_DIR = r"C:\Users\RoboGor\Desktop\train_seg\img"
TRAIN_MASK_DIR = r"C:\Users\RoboGor\Desktop\train_seg\mask"
TEST_IMAGE_DIR = r"C:\Users\RoboGor\Desktop\test_seg\img"
TEST_MASK_DIR = r"C:\Users\RoboGor\Desktop\test_seg\mask"

LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 3
# A.RandomBrightnessContrast(p=0.2),
augmentations = A.Compose([
    A.RandomCrop(width=800, height=800),
    A.HorizontalFlip(p=0.5),
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
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            transform = self.transform(image=image, mask=mask)
            image = transform["image"]
            mask = transform["mask"]

        image = image / 255.0

        return image, mask


train_val = aerialDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, transform=augmentations)
test_set = aerialDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transform=augmentations)

train_val_size = len(train_val)
train_set_len = int(train_val_size * 0.8)
train_set, val_set = random_split(train_val, [train_set_len, train_val_size - train_set_len],
                                  generator=torch.Generator().manual_seed(1))

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


class ConvBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, bias=False, reluInplace=True):
        super(ConvBatchNormReLU, self).__init__()
        self.seqLayers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=reluInplace)
        )

    def forward(self, x):
        return self.seqLayers(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, poolKernel=2, poolStride=2, copySizes=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downConv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=poolKernel, stride=poolStride)
        self.upsampling = nn.Upsample(scale_factor=(poolKernel, poolKernel), mode='bilinear')
        self.conv = nn.ModuleList()
        self.upConv = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        # doubleConv-down
        for numberOfFeatures in copySizes:
            self.downConv.append(nn.Sequential(
                ConvBatchNormReLU(in_channels, numberOfFeatures),
                ConvBatchNormReLU(numberOfFeatures, numberOfFeatures)
            ))
            in_channels = numberOfFeatures

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBatchNormReLU(copySizes[-1], copySizes[-1] * 2),
            ConvBatchNormReLU(copySizes[-1] * 2, copySizes[-1] * 2)
        )

        # doubleConv-up
        for numberOfFeatures in reversed(copySizes):
            # TODO ALTERNATİF OLARAK UPSAMPLING'DE DE PARAMETRE ÖĞRENMESİ İSTENİRSE BİR ALT SATIRA EKLENECEK LAYER: CONVTRANSPOSE2D. BU KULLANIMDA UPSAMPLING FORWARD'DA KALDIRILACAK
            self.upConv.append(nn.Sequential(
                ConvBatchNormReLU(numberOfFeatures * poolKernel, numberOfFeatures),
                ConvBatchNormReLU(numberOfFeatures, numberOfFeatures)
            ))

            # conv after every upsampling
            self.conv.append(nn.Sequential(
                nn.Conv2d(numberOfFeatures * 2, numberOfFeatures, kernel_size=3, stride=1, padding=1, bias=False)
            ))

        # final convolution layer
        self.convFinal = nn.Conv2d(copySizes[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        copies = []  # copyNcrop'ta taşınacak matrisler. copyNcrop'un her bir elemanı bu matrislerin boyutunu verir.
        for doubleConv in self.downConv:
            x = doubleConv(x)
            copies.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        copies = copies[::-1]  # LİSTEDEKİ MATRİSLERİN SIRASI TERSİNE DÖNDÜ

        for index in range(len(self.upConv)):
            x = self.upsampling(x)
            x = self.conv[index](x)
            if x.shape != copies[index].shape:
                _, _, height, width = x.shape
                croppedCopy = F.center_crop(copies[index], output_size=(height, width))
                x = torch.cat((croppedCopy, x), dim=1)
            else:
                x = torch.cat((copies[index], x), dim=1)

            x = self.upConv[index](x)

        x = self.convFinal(x)
        x = self.sigmoid(x)

        return x


model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter()


def pixel_accuracy(output, mask):
    with torch.no_grad():
        # output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


train_val_time = 0
for epoch in range(EPOCHS):
    start = time.time() / 60
    tr_loss = 0
    tr_iou = 0
    tr_acc = 0
    tr_f1 = 0
    tr_dice = 0
    print(f"Epoch: {epoch}")
    if epoch % 5 == 0:
        print('Saving state dictionary...')
        torch.save(model.state_dict(), 'UNet-ForestDataset_06_01_Checkpoint' + str(epoch) + '.pth')
    model.train()
    for i, (images, masks) in enumerate(train_dataloader):
        print(f"batch: {i}")
        optimizer.zero_grad()
        outputs = model(images.to(device))
        masks = masks.float().unsqueeze(1).to(device)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        # tr_acc += pixel_accuracy(outputs, masks)
        # tr_f1 += f1_score(masks.detach().cpu().contiguous().view(-1), outputs.detach().cpu().contiguous().view(-1),
        #                   average='binary')
        # tr_dice += dice_score(outputs, masks)
        # optimizer.zero_grad()

        # tr_loss += loss.item()
        # # tr_iou += mIoU(outputs, masks)
        # tr_iou += jaccard_score(masks.detach().cpu().contiguous().view(-1),
        #                         outputs.detach().cpu().contiguous().view(-1))

    # print(f"Training Loss: {loss/len(train_dataloader)}    Training Accuracy: {acc}")
    # print(f"Training Loss: {tr_loss/len(train_dataloader)}  Training Accuracy: {tr_acc/len(train_dataloader)}
    # Training mIoU: {tr_iou/len(train_dataloader)}   Training F1: {tr_f1/len(train_dataloader)}
    # Training Dice: {tr_dice/len(train_dataloader)}")
    # print(
    #     f"Training Loss: {tr_loss / len(train_dataloader)}  Training Accuracy: {tr_acc / len(train_dataloader)}
    #     Training mIoU: {tr_iou / len(train_dataloader)}   Training F1: {tr_f1 / len(train_dataloader)}")

    model.eval()
    # val_loss = 0
    # val_acc = 0
    # val_iou = 0
    # val_f1 = 0
    # val_dice = 0
    for i, (images, masks) in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(images.to(device))

        masks = masks.float().unsqueeze(1).to(device)
        loss = criterion(outputs, masks)
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0


    train_val_time_epoch = time.time() / 60 - start
    print(f'Elapsed time: {train_val_time_epoch} minutes')
    train_val_time += train_val_time_epoch

print(f'Training completed. Elapsed time: {train_val_time} minutes')
print('Saving state dictionary...')
torch.save(model.state_dict(), 'UNet-ForestDataset_06_01_final.pth')
print('Save successful. Entering test phase...')

model.eval()
test_loss = 0
test_acc = 0
test_iou = 0
test_f1 = 0
test_dice = 0
x = 0
for i, (images, masks) in enumerate(test_dataloader):

    # optimizer.zero_grad()
    with torch.no_grad():
        outputs = model(images.to(device))

    masks = masks.float().unsqueeze(1).to(device)
    masks[masks >= 0.5] = 1.0
    masks[masks < 0.5] = 0.0
    loss = criterion(outputs, masks)
    outputs[outputs >= 0.5] = 1.0
    outputs[outputs < 0.5] = 0.0
    test_acc += pixel_accuracy(outputs, masks)
    print(f'acc: {test_acc}')

    image = images[x].cpu().detach().numpy() * 255
    output = outputs[x].cpu().detach().numpy() * 255
    mask = masks[x].cpu().detach().numpy() * 255

    image = np.transpose(image, (1, 2, 0))
    output = np.transpose(output, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imshow('original image', image.astype('uint8'))
    cv2.imshow('output', output.astype('uint8'))
    cv2.imshow('mask', mask.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

