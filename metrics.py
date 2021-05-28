import torch
from collections import Counter

from general import *

def compute_map(predictions, targets, imgsize, nclasses):
    '''
    detections each contents: (x1, y1, x2, y2, p1, p2, c)
    targets contents: (b, c, x, y, w, h)
    '''
    detections = nonmax_supression(predictions)

    preds = []
    for i, d in enumerate(detections):
        if d != None:
            for j, pbox in enumerate(d):
                bbox = torch.zeros(7)
                bbox[0] = i
                bbox[1] = pbox[-1]
                bbox[2] = pbox[-3]
                bbox[3:7] = pbox[:4]
                preds.append(bbox)

    trgts = []
    for i, tbox in enumerate(targets):
        bbox = torch.zeros(7)
        bbox[0] = tbox[0]
        bbox[1] = tbox[1]
        bbox[2] = 1
        bbox[3:7] = xywh_xyxy(tbox[2:6] * imgsize)
        trgts.append(bbox)

    mean_ap(preds, trgts, nclasses)

def mean_ap(predictions, targets, nclasses, iouthresh=0.5):
    '''
    Arguments:
    predictions: list of tensors: shape: (B, 6) contents: (b, c, p, x1, y1, x2, y2)
    targets: list of tensors: shape: (B, 6) contents: (b, c, p, x1, y1, x2, y2)
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
            if detection[1] == c:
                detections.append(detection)

        for gtbox in targets:
            if gtbox[1] == c:
                groundtruths.append(gtbox)

        countgtboxs = Counter([gt[0].item() for gt in groundtruths])
        for k, v in countgtboxs.items():
            countgtboxs[k] = torch.zeros(v)

        detections.sort(key=lambda x: x[2], reverse=True)

        print(detections)
        print(countgtboxs)

        #for i, detection in enumerate(detections):



