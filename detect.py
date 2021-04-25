import os
import cv2
import glob
import math
import yaml
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torchvision.transforms import ToTensor

from utils import *
from models import *
from datasets import *
from drawbox import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_darknet(options):
    with open('hyperparams.yaml', 'r') as f:
        hyp = yaml.safe_load(f)

    imgsize = hyp['imgsize']

    pcolor = (255, 0, 0)
    names = get_names(options.names)
    model = Darknet(options.cfg, imgwh=imgsize).to(device)
    model.load_state_dict(torch.load(options.weights, map_location=device))

    print('showing the actual boxes...')
    show_boxes(names, options.testdir, imgsize, options.boxdir)

    print('the predictions made by the model...')
    predict_boxes(model, names, options.testdir, imgsize, 1, options.savedir, pcolor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdir', type=str, help='path to image folder')
    parser.add_argument('--names', type=str, help='path to names file')
    parser.add_argument('--cfg', type=str, help='a .cfg file for model architecture')
    parser.add_argument('--weights', type=str, help='path to pre-trained weights')
    parser.add_argument('--savedir', type=str, help='folder to save detected images')
    parser.add_argument('--boxdir', type=str, help='images with ground-truth boxes drawn')

    options = parser.parse_args()

    detect_darknet(options)
