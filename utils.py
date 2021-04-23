import os
import cv2
import glob
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from augmentations import *

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

def get_names(path):
    names = []
    with open(path) as f:
        lines = f.read().split('\n')
        [names.append(line.lstrip().rstrip()) for line in lines]

    return list(filter(None, names))

def to_cpu(tensor):
    return tensor.detach().cpu()

def img_resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

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

def bbox_whiou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    interarea = torch.min(w1, w2) * torch.min(h1, h2)
    unionarea = (w1 * h1 + 1e-16) + w2 * h2 - interarea

    return interarea / unionarea

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1x1, b1x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1y1, b1y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2x1, b2x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2y1, b2y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1x1, b1y1, b1x2, b1y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    interrectx1 = torch.max(b1x1, b2x1)
    interrecty1 = torch.max(b1y1, b2y1)
    interrectx2 = torch.min(b1x2, b2x2)
    interrecty2 = torch.min(b1y2, b2y2)

    # Intersection area
    interarea = torch.clamp(interrectx2 - interrectx1 + 1, min=0) * torch.clamp(interrecty2 - interrecty1 + 1, min=0)
    # Union Area
    b1area = (b1x2 - b1x1 + 1) * (b1y2 - b1y1 + 1)
    b2area = (b2x2 - b2x1 + 1) * (b2y2 - b2y1 + 1)
    
    unionarea = b1area + b2area - interarea + 1e-16
    iou = interarea / unionarea

    return iou

def calc_ious(box1, box2, x1y1x2y2=True, mode='iou', eps=1e-7):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1x1, b1x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1y1, b1y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2x1, b2x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2y1, b2y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1x1, b1y1, b1x2, b1y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    interrectx1 = torch.max(b1x1, b2x1)
    interrecty1 = torch.max(b1y1, b2y1)
    
    #print(interrectx1.grad_fn, interrecty1.grad_fn)

    interrectx2 = torch.min(b1x2, b2x2)
    interrecty2 = torch.min(b1y2, b2y2)

    #print(interrectx2.grad_fn, interrecty2.grad_fn)

    # Intersection area
    interw, interh = torch.clamp(interrectx2 - interrectx1, min=0), torch.clamp(interrecty2 - interrecty1, min=0)
    interarea = interw * interh

    # Union Area
    b1w, b1h = (b1x2 - b1x1), (b1y2 - b1y1)
    b2w, b2h = (b2x2 - b2x1), (b2y2 - b2y1)
    b1area = b1w * b1h
    b2area = b2w * b2h

    unionarea = b1area + b2area - interarea + eps

    iou = interarea / unionarea # IoU

    if mode == 'giou' or  mode == 'diou' or  mode == 'ciou':
        convexw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1) 
        convexh = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        carea = convexw * convexh
        if mode == 'giou':
            return iou - (carea - unionarea) / carea #GIoU
        elif mode == 'diou' or 'ciou':
            cdist = convexw ** 2 + convexh ** 2 + eps ##diagonal (pythagoras theorem hyp = adj ** 2 + opp ** 2)
            bdist = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
            if mode == 'diou':
                return iou - (bdist / cdist) #DIoU
            elif mode == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(b2w / b2h) - torch.atan(b1w / b1h), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - ((bdist / cdist) + alpha * v) ##CIoU

    return iou ##IoU
 

def nonmax_supression(prediction, confthresh=0.5, iouthresh=0.4):
    '''
    B = 10647 for a 416x416 image with scales (13, 26, 52)
    prediction shape:(N, B, C+5) eval:(bx, by, bw, bh, po, [p1, ... pc])
    returns outputs which is a list of tensors
    '''

    prediction[..., :4] = xywh_xyxy(prediction[..., :4])
    outputs = [None for _ in range(len(prediction))]

    for i, pi in enumerate(prediction):
        pi = pi[pi[:, 4] >= confthresh]

        if not pi.size(0):
            continue
        
        scores = pi[:, 4] * pi[:, 5:].max(1)[0]
        pi = pi[(-scores).argsort()]
        confs, idxs = pi[:, 5:].max(1, keepdim=True)
        detections = torch.cat((pi[:, :5], confs.float(), idxs.float()), 1)

        keepbox = []
        while detections.size(0):
            overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > iouthresh
            matchlabel = detections[0, -1] == detections[:, -1]

            invalid = overlap & matchlabel
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()

            keepbox += [detections[0]]
            detections = detections[~invalid]

        if keepbox:
            outputs[i] = torch.stack(keepbox)
  
    return outputs

def build_targets(targets, sclanchors, predclasses, predboxes, ignthresh, mode='logitbox'):
    BoolTensor = torch.cuda.BoolTensor if predboxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if predboxes.is_cuda else torch.FloatTensor

    N, A, G, C = predboxes.size(0), predboxes.size(1), predclasses.size(2), predclasses.size(-1)

    objmask, noobjmask = BoolTensor(N, A, G, G).fill_(0), BoolTensor(N, A, G, G).fill_(1)
    x, y = FloatTensor(N, A, G, G).fill_(0), FloatTensor(N, A, G, G).fill_(0)
    w, h = FloatTensor(N, A, G, G).fill_(0), FloatTensor(N, A, G, G).fill_(0)
    trueclasses = FloatTensor(N, A, G, G, C).fill_(0)

    batchidxs, labels = targets[:, :2].long().t()
    targetboxes = targets[:, 2:6] * G

    gx, gy = targetboxes[:, :2].t()
    gw, gh = targetboxes[:, 2:].t()
    gi, gj = gx.long(), gy.long()

    ious = torch.stack([bbox_whiou(anchor, targetboxes[:, 2:]) for anchor in sclanchors])
    bestious, bestn = ious.max(0)

    objmask[batchidxs, bestn, gj, gi] = 1
    noobjmask[batchidxs, bestn, gj, gi] = 0

    for i, anchoriou in enumerate(ious.t()):
        noobjmask[batchidxs[i], anchoriou > ignthresh, gj[i], gi[i]] = 0

    trueclasses[batchidxs, bestn, gj, gi, labels] = 1

    trueconfs = objmask.float()

    x[batchidxs, bestn, gj, gi] = gx - gx.floor()
    y[batchidxs, bestn, gj, gi] = gy - gy.floor()

    if mode == 'logitbox':
        w[batchidxs, bestn, gj, gi] = torch.log(gw / sclanchors[bestn][:, 0] + 1e-16)
        h[batchidxs, bestn, gj, gi] = torch.log(gh / sclanchors[bestn][:, 1] + 1e-16)

    elif mode == 'probbox':
        w[batchidxs, bestn, gj, gi] = targetboxes[:, 2]
        h[batchidxs, bestn, gj, gi] = targetboxes[:, 3]

    return objmask, noobjmask, x, y, w, h, trueclasses, trueconfs

def compute_grid(imgwh, gsize, anchors, cuda=True):
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    stride = imgwh / gsize 
    g = gsize
    gridx = torch.arange(g).repeat(g, 1).view([1,1,g,g]).type(FloatTensor)
    gridy = torch.arange(g).repeat(g, 1).t().view([1,1,g,g]).type(FloatTensor)
    sclanchors = FloatTensor([(aw / stride, ah / stride) for aw, ah in anchors])
    
    return gridx, gridy, sclanchors, stride

def draw_outputs(img, outputs, names, color=(0,255,0)):
    boxes, confs, classes = outputs[:, :4], outputs[:, -3], outputs[:, -1]
    classes = classes.astype(np.int32)
    nums = boxes.shape[0]

    print('number of boxes: {}'.format(nums))
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2])).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4])).astype(np.int32))

        img = cv2.rectangle(img, x1y1, x2y2, color)
        confidence = round(confs[i], 2)
        objstr = names[classes[i]]+' '+str(confidence)
        textcoords = (x1y1[0], x1y1[1] - 5)
        img = cv2.putText(img, objstr, textcoords, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, 1)

        print(classes[i], confidence, names[classes[i]])

    print(' ')

    return img

def load_img(path, size):
    orgimg = cv2.resize(cv2.imread(path), (size, size))
    normimg = transforms.ToTensor()(np.asarray(orgimg)).unsqueeze(0).float()
    return normimg, orgimg

def predict_boxes(model, names, directory, imgsize, batchsize, color=(0, 255, 0)):
    print('')
    print(f'showing from dir {directory}')

    testloader = DataLoader(
        ImageFolder(directory, transform= \
            transforms.Compose([DEFAULTTRANSFORMS, Resize(imgsize)])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    for i, (path, img) in enumerate(testloader):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        img = Variable(img.type(Tensor))

        print(path[0], imgsize)
        _, predimg = load_img(path[0], imgsize)

        model.eval()
        preds = model(img)
        bboxes = nonmax_supression(preds, confthresh=0.8)[0]

        if bboxes is not None:
            predimg = draw_outputs(predimg, bboxes.numpy(), names, color)
        
        cv2_imshow(predimg)

def show_boxes(names, directory, imgsize):
    print(f'showing from dir {directory}')

    paths = os.listdir(directory)
    paths = [directory+'/'+i[:-4] for i in paths if i[-4:] == '.jpg']
    for i, path in enumerate(paths):
        pimg, ptxt = path+'.jpg', path+'.txt'

        _, img = load_img(pimg, size=416)

        boxes = np.loadtxt(fname=ptxt).reshape(-1, 5)
        bboxes = np.zeros((boxes.shape[0], 7), dtype=np.float32)
        bboxes[:, :4] = boxes[:, 1:] * imgsize
        bboxes[:, :4] = xywh2xyxy_np(bboxes[:, :4])
        bboxes[:, -1] = boxes[:, 0]
        bboxes[:, 4:6] = [1,1]

        predimg = draw_outputs(img, bboxes, names)

        cv2.imwrite(predimg)


