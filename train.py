import os
import cv2
import glob
import math
import random
import argparse
import yaml
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

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train_darknet(options):
    cfgpath = options.cfg
    ptweights = options.ptweights
    ckptpth = options.ckptpth
    traindir, validdir = options.traindir, options.validdir

    if options.lossfn == 'bboxloss':
        lossfn = BboxLoss
    else:
        lossfn = IoULoss

    scaler = amp.GradScaler(enabled=cuda)

    with open('hyperparams.yaml', 'r') as f:
        hyp = yaml.safe_load(f)

    imgsize = hyp['imgsize']

    epochs = options.epochs
    batchsize = options.batchsize
    accumgradient = 2
    ncpu = options.ncpu
    ckptinterval = 2

    model = Darknet(cfgpath, imgwh=imgsize).to(device)
    model.apply(init_normal)

    print(model)

    if ptweights.endswith('.pth'):
        model.load_state_dict(torch.load(ptweights, map_location=device))
    else:
        model.load_weights(ptweights)

    traindata = ListDataset(traindir, imgsize, multiscale=True, transform=AUGMENTATIONTRANSFORMS)
    validdata = ListDataset(validdir, imgsize, multiscale=False, transform=DEFAULTTRANSFORMS)

    trainloader = DataLoader(traindata, batchsize, shuffle=True,
                             num_workers=ncpu, pin_memory=True,
                             collate_fn=traindata.collate_fn)

    validloader = DataLoader(validdata, batchsize, shuffle=False,
                             num_workers=1, pin_memory=True,
                             collate_fn=validdata.collate_fn)

    optimizer = torch.optim.Adam(model.parameters())

    criterion = ComputeLoss(model, hyp, lossfn)

    bestloss = 1e+5
    patience = options.patience
    orgpatience = patience
    for e in range(epochs):

        model.train()
        epochloss = 0
        for b, (_, imgs, targets) in enumerate(trainloader):
            with amp.autocast(enabled=cuda):
                batchesdone = len(trainloader) * e + b

                outputs = model(imgs.to(device), 'train')
                loss = criterion(outputs, targets.to(device))

            scaler.scale(loss).backward()

            if batchesdone % accumgradient == 0:
                scaler.step(optimizer)
                scaler.update()
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
                #imgs = Variable(imgs.to(device), requires_grad=False)
                #targets = Variable(targets.to(device), requires_grad=False)

                outputs = model(Variable(imgs.to(device), requires_grad=False), 'train')
                loss = criterion(outputs, Variable(targets.to(device), requires_grad=False))

                batchloss = to_cpu(loss).item()
                epochloss += batchloss

            print(f'validation loss at batch {b}: {batchloss:.3f}')

        epochloss /= len(validloader)

        print(f'validation loss at epoch {e}: {epochloss:.3f}')

        patience -= 1
        if epochloss <= bestloss:
            torch.save(model.state_dict(), ckptpth)
            bestloss = epochloss
            patience = orgpatience

            print(f'saved best model weights at epoch {e} to {ckptpth}')

        if not patience:
            print(f'early stopping.. validation loss did not improve from {bestloss:.3f}')
            print(f'you can change the patience value... current value {orgpatience} epochs')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, help='path to training set')
    parser.add_argument('--validdir', type=str, help='path to validation set')
    parser.add_argument('--cfg', type=str, help='a .cfg file for model architecture')
    parser.add_argument('--ptweights', type=str, help='path to pretrained weights')
    parser.add_argument('--ckptpth', type=str, help='path to save trained model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ncpu', type=int, default=2)
    parser.add_argument('--lossfn', type=str, help='type bboxloss or iouloss', default='iouloss')

    options = parser.parse_args()

    train_darknet(options)
