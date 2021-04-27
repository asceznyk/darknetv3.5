import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class Upsample(nn.Module):
    def __init__(self, scale, mode):
        super(Upsample, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLO(nn.Module):
    def __init__(self, anchors, nclasses, imgwh):
        super(YOLO, self).__init__()
        self.anchors = anchors
        self.nclasses = nclasses
        self.imgwh = imgwh
        self.gsize = 0

    def forward(self, x, mode):
        '''
        x.shape = (N, G, G, A, C+5) x = (tx, ty, tw, th, po, [p1, ... pc])
        targets.shape = (N * B, 6) targets = (batchidx, classidx, bx, by, bw, bh)
        '''

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        N, G = x.size(0), x.size(2)
        self.numanchors = len(self.anchors)

        preds = (
            x.view(N, self.numanchors, self.nclasses + 5, G, G)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        ) #preds.shape (N, A, G, G, C+5) #preds = (tx, ty, tw, th, po, [p1, ... pc])

        if G != self.gsize:
            self.gsize = g = G
            self.gridx, self.gridy, self.sclanchors, self.stride = \
                compute_grid(self.imgwh, self.gsize, self.anchors, x.is_cuda)

        ptx, pty = torch.sigmoid(preds[..., 0]), torch.sigmoid(preds[..., 1])
        ptw, pth = preds[..., 2], preds[..., 3]
        predconfs, predclasses = torch.sigmoid(preds[..., 4]), torch.sigmoid(preds[..., 5:])

        predboxes = FloatTensor(preds[..., :4].shape) 
        predboxes[..., 0] = (ptx.data + self.gridx)
        predboxes[..., 1] = (pty.data + self.gridy) 
        predboxes[..., 2] = torch.exp(ptw.data) * self.sclanchors[:, 0:1].view((1, self.numanchors, 1, 1))
        predboxes[..., 3] = torch.exp(pth.data) * self.sclanchors[:, 1:2].view((1, self.numanchors, 1, 1))

        if mode == 'train':
            #outputs = torch.cat((torch.stack([ptx, pty, ptw, pth], -1), predconfs.unsqueeze(-1), predclasses), -1)
            outputs = preds
        else:
            predboxes *= self.stride
            outputs = torch.cat((
                predboxes.view(N, -1, 4),
                predconfs.view(N, -1, 1), 
                predclasses.view(N, -1, self.nclasses)
            ), -1)

        return outputs, self.anchors

def create_module(blocks, imgwh):
    '''
    converts blocks to nn.ModuleList()
    where all the types of layers are present
    '''

    hyperparams = blocks.pop(0)
    outfilters = [int(hyperparams['channels'])]
    modulelist = nn.ModuleList()

    for l, block in enumerate(blocks):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            batchnorm = int(block['batch_normalize'])
            filters = int(block['filters'])
            ksize = int(block['size'])
            pad = (ksize - 1) // 2

            module.add_module(
                f'conv_{l}',
                nn.Conv2d(
                    outfilters[-1], filters, ksize, 
                    int(block['stride']), pad, bias = not batchnorm
                )
            )

            if batchnorm:
                module.add_module(f'bnorm_{l}', nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if block['activation'] == 'leaky':
                module.add_module(f'leaky_{l}', nn.LeakyReLU(0.1))

        elif block['type'] == 'upsample':
            module.add_module(f'upsample_{l}', Upsample(scale=int(block['stride']), mode='nearest'))

        elif block['type'] == 'shortcut':
            filters = outfilters[1:][int(block['from'])]
            module.add_module(f'shortcut_{l}', EmptyLayer())

        elif block['type'] == 'route':
            layers = [int(val.rstrip().lstrip()) for val in block['layers'].split(',')]
            filters = sum([outfilters[1:][l] for l in layers])
            module.add_module(f'route_{l}', EmptyLayer())

        elif block['type'] == 'yolo':
            mask = [int(x) for x in block['mask'].split(',')]
            anchors = [int(x) for x in block['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[m] for m in mask]
            module.add_module(f'yolo_{l}', YOLO(anchors, int(block['classes']), imgwh))

        modulelist.append(module)
        outfilters.append(filters)

    return hyperparams, modulelist

class Darknet(nn.Module):
    def __init__(self, cfgpath, imgwh=416):
        super(Darknet, self).__init__()
        self.imgwh = imgwh
        self.blocks = parse_blocks(cfgpath)
        self.hyperparams, self.modulelist = create_module(self.blocks, self.imgwh)
    
    def forward(self, x, mode='detect'):
        convs, detections, allanchors = [], [], []
        loss = 0
        for l, (block, module) in enumerate(zip(self.blocks, self.modulelist)):
            if block['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif block['type'] == 'route':
                x = torch.cat([convs[int(i)] for i in block['layers'].split(',')], 1)
            elif block['type'] == 'shortcut':   
                x = convs[-1] + convs[int(block['from'])]
            elif block['type'] == 'yolo':
                x, anchors = module[0](x, mode)  
                allanchors.append(anchors)
                detections.append(x)
            convs.append(x)
        
        if mode == 'detect':
            return to_cpu(torch.cat(detections, 1))

        return detections, allanchors

    def load_weights(self, path):
        with open(path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)

        cutoff=None
        if 'darknet53.conv.74' in path:
            cutoff=75

        ptr = 0
        for l, (block, module) in enumerate(zip(self.blocks, self.modulelist)):
            if l == cutoff:
                break

            if block['type'] == 'convolutional':
                conv = module[0]
                if block['batch_normalize']:
                    bnorm = module[1]
                    nbias = bnorm.bias.numel()

                    wbias = torch.from_numpy(weights[ptr : ptr+nbias]).view_as(bnorm.bias)
                    bnorm.bias.data.copy_(wbias)
                    ptr += nbias
                    wweight = torch.from_numpy(weights[ptr : ptr+nbias]).view_as(bnorm.weight)
                    bnorm.weight.data.copy_(wweight)
                    ptr += nbias
                    wrunmean = torch.from_numpy(weights[ptr : ptr+nbias]).view_as(bnorm.running_mean)
                    bnorm.running_mean.data.copy_(wrunmean)
                    ptr += nbias
                    wrunvar = torch.from_numpy(weights[ptr : ptr+nbias]).view_as(bnorm.running_var)
                    bnorm.running_var.data.copy_(wrunvar)
                    ptr += nbias
                else:
                    nbias = conv.bias.numel()
                    wbias = torch.from_numpy(weights[ptr : ptr+nbias]).view_as(conv.bias)
                    conv.bias.data.copy_(wbias)
                    ptr += nbias

                nweights = conv.weight.numel()
                wweight = torch.from_numpy(weights[ptr : ptr+nweights]).view_as(conv.weight)
                conv.weight.data.copy_(wweight)
                ptr += nweights

class ModelEMA:
    """
    implements tf.train.ExponentialMovingAverage: 
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x : decay * (1 - math.exp(-x / 2000))

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()

            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()



