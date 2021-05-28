import torch
from collections import Counter

from general import *

def compute_map(predictions, targets):
    print(predictions)
    detections = nonmax_supression(predictions)
    print(predictions, targets.size())

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

