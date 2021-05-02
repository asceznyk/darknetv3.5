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
import torch.optim.lr_scheduler as lr_scheduler

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

    print(model)

    ema = ModelEMA(model)

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

    #optimizer = torch.optim.Adam(model.parameters())

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    optimizer.add_param_group({'params': pg1, 'weight_decay':hyp['weightdecay']})
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)s
    del pg0, pg1, pg2

    if options.linearlr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion = lossfn(hyp)
    nw = max(round(hyp['warmupepochs'] * len(trainloader)), 1000)

    bestloss = 1e+5
    patience = options.patience
    orgpatience = patience
    for e in range(epochs):

        model.train()
        epochloss = 0
        for b, (_, imgs, targets) in enumerate(trainloader):
            batchesdone = ni = len(trainloader) * e + b

            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmupbiaslr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmupmomentum'], hyp['momentum']])

            with amp.autocast(enabled=cuda):
                outputs = model(imgs.to(device), 'train')
                loss = criterion(outputs, targets.to(device))

            scaler.scale(loss).backward()

            if batchesdone % accumgradient == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema:
                    print(f'updated ema weights! at batch {b} epoch {e}')

                    ema.update(model)

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

        scheduler.step()

        patience -= 1
        if epochloss <= bestloss:
            ckpt = msd = model.state_dict()
            if ema:
                ckpt = {
                    'model': msd,
                    'ema':ema.ema.state_dict()
                }
            torch.save(ckpt, ckptpth)
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
    parser.add_argument('--linearlr', action='store_true', help='linear LR')

    options = parser.parse_args()

    train_darknet(options)
