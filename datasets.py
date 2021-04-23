import os
import cv2
import glob
import math
import random
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from augmentations import *

class ImageFolder(Dataset):
    def __init__(self, fpath, transform=None):
        self.files = [p for p in sorted(glob.glob("%s/*.*" % fpath)) if p[-4:] == '.jpg']
        self.transform = transform

    def __getitem__(self, index):
        imgpath = self.files[index % len(self.files)]
        img = cv2.imread(imgpath)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return imgpath, img

    def __len__(self):
        return len(self.files)

class ListDataset(Dataset):
    def __init__(self, directory, imgwh, multiscale, transform):
        self.files = [directory+p for p in sorted(os.listdir(directory)) if p[-4:] == '.jpg']
        self.imgwh = imgwh
        self.transform = transform
        self.batchcount = 0
        self.multiscale = multiscale
        self.maxsize = self.imgwh + 3 * 32
        self.minsize = self.imgwh - 3 * 32
    
    def __getitem__(self, idx):
        size = self.imgwh
        imgpath = self.files[idx % len(self.files)]
        txtpath = self.files[idx % len(self.files)][:-4]+'.txt'
    
        img = cv2.resize(cv2.imread(imgpath), (size, size))
        bboxes = np.loadtxt(txtpath).reshape(-1, 5)

        if self.transform:
            img, bbtargets = self.transform((img, bboxes))
        else:
            img = transforms.ToTensor()(img)
            bbtargets = torch.zeros(bboxes.shape[0], 6)
            bbtargets[:, 1:] = torch.FloatTensor(bboxes)

        return imgpath, img, bbtargets
    
    def __len__(self):
        return len(self.files)

    def collate_fn(self, batch):
        self.batchcount += 1

        batch = [data for data in batch if data is not None]
        paths, imgs, bbtargets = list(zip(*batch))

        imgs = torch.stack([img for img in imgs])

        if self.multiscale and self.batchcount % 10 == 0:
            self.imgwh = random.choice(range(self.minsize, self.maxsize + 1, 32))

        imgs = torch.stack([img_resize(img, self.imgwh) for img in imgs])

        for b, bbox in enumerate(bbtargets):
            bbox[:, 0] = b
        bbtargets = torch.cat(bbtargets, 0)

        return paths, imgs, bbtargets


