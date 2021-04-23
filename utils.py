import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_blocks(path):
    file = open(path)
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x.rstrip().lstrip() for x in lines if x[0] != '#']
 
    blocks = []
 
    for line in lines:
        if line[0] == '[':
            blocks.append({})
            blockh = line.replace('[', '').replace(']', '')
            blocks[-1]['type'] = blockh
            
            if blocks[-1]['type'] == 'convolutional':
                blocks[-1]['batch_normalize'] = 0
        else:
            prop, val = line.split('=')
            blocks[-1][prop.rstrip().lstrip()] = val.rstrip().lstrip()
    
    return blocks

def to_cpu(tensor):
    return tensor.detach().cpu()

def img_resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_names(path):
    names = []
    with open(path) as f:
        lines = f.read().split('\n')
        [names.append(line.lstrip().rstrip()) for line in lines]

    return list(filter(None, names))

def xywh_xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

