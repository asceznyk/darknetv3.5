import os
import cv2
import glob
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torchvision.transforms import ToTensor

from utils import *
from augmentations import *
from models import *
from losses import *
from datasets import *

'''
cfgpath = options['cfg']
weightspath = 'darknet53.conv.74'
customwpath = rootpath+'customwyolov3.pth'
traindir, validdir = 'train/', 'valid/'
pcolor = (255, 0, 0)

##hyperparameters
epochs = 100
batchsize = 4
accumgradient = 2
ncpu = 2
imgsize = 416
ckptinterval = 2

names = get_names('obj.names')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Darknet(cfgpath, imgwh=imgsize).to(device)
model.apply(init_normal)

print(model)

if weightspath.endswith('.pth'):
    model.load_state_dict(torch.load(weightspath))
else:
    model.load_weights(weightspath)

traindata = ListDataset(traindir, imgsize, multiscale=True, transform=AUGMENTATIONTRANSFORMS)
validdata = ListDataset(validdir, imgsize, multiscale=False, transform=DEFAULTTRANSFORMS)

trainloader = DataLoader(traindata, batchsize, shuffle=True,
                         num_workers=ncpu, pin_memory=True, 
                         collate_fn=traindata.collate_fn)

validloader = DataLoader(validdata, batchsize, shuffle=False,
                         num_workers=1, pin_memory=True, 
                         collate_fn=validdata.collate_fn)

optimizer = torch.optim.Adam(
            model.parameters(),
            )

criterion = ComputeLoss(model, IoULoss)

bestloss = 1e+5
patience = 1e+5
orgpatience = patience
for e in range(epochs):

    model.train()
    epochloss = 0
    for b, (_, imgs, targets) in enumerate(trainloader):
        batchesdone = len(trainloader) * e + b
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        outputs = model(imgs, 'train')
        loss = criterion(outputs, targets) 

        loss.backward()

        if  batchesdone % accumgradient == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        batchloss = to_cpu(loss).item()
        epochloss += batchloss

        print(f'training loss at batch {b}: {batchloss:.3f}')

    epochloss /= len(trainloader)

    print(f'training loss at epoch {e}: {epochloss:.3f}')

    model.eval()
    epochloss = 0
    for b, (_, imgs, targets) in enumerate(validloader):
        with torch.no_grad():
            imgs = Variable(imgs.to(device), requires_grad=False)
            targets = Variable(targets.to(device), requires_grad=False)

            outputs = model(imgs, 'train')
            loss = criterion(outputs, targets)
            
            batchloss = to_cpu(loss).item()
            epochloss += batchloss

        print(f'validation loss at batch {b}: {batchloss:.3f}')

    epochloss /= len(validloader)

    print(f'validation loss at epoch {e}: {epochloss:.3f}')

    patience -= 1
    if epochloss <= bestloss:
        torch.save(model.state_dict(), customwpath)
        bestloss = epochloss 
        patience = orgpatience

        print(f'saved best model weights at epoch {e} to {customwpath}')

    if not patience:
        print(f'early stopping.. validation loss did not improve from {bestloss:.3f}')
        print(f'you can change the patience value... current value {orgpatience} epochs')
        break

model.load_state_dict(torch.load(customwpath))
'''

def test_func(options):
    print(options.train_dir, options.valid_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, help='path to training set')
    parser.add_argument('--valid-dir', type=str, help='path to validation set')
    parser.add_argument('--names', type=str, help='path to file which contains all class names')
    parser.add_argument('--cfg', type=str, help='a .cfg file for model architecture')
    parser.add_argument('--ptweights-pth', type=str, help='path to pretrained weights')
    parser.add_argument('--ckpt-pth', type=str, help='path to save trained model')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)

    options = parser.parse_args()

    test_func(options)
