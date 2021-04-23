import numpy as np

import torch

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
 
