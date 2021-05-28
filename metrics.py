import torch
from collections import Counter

from general import *

def compute_map(predictions, targets):
    '''
    detections each contents: (x1, y1, x2, y2, p1, p2, c)
    targets contents: (b, c, x, y, w, h)
    '''
    detections = nonmax_supression(predictions)

    preds = []
    for i, d in enumerate(detections):
        bbox = torch.zeros(7)
        bbox[0] = i
        bbox[1] = d[-1]
        bbox[2] = d[-3]
        bbox[3:7] = d[:4]
        preds.append(bbox)

    print(preds)



def mean_ap(predictions, targets, nclasses, iouthresh=0.5):
    '''
    Arguments:
    predictions: list of tensors: shape(B, 6) contents(b, c, p, x1, y1, x2, y2)
    targets: list of tensors: shape(B, 6) contents(b, c, p, x1, y1, x2, y2)
    nclasses: number of classes
    iouthresh: IoU threshold

    Returns:
    mAP value for all classes given IoU threshold
    '''

    aps = []
    epsilon = 1e-6

    for c in range(nclasses):
        detections = []
        groundtruths = []

        for detection in predictions:
            if detections[1] == c:
                detections.append(c)

