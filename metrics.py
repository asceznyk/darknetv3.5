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

    print(mean_ap(preds, trgts, nclasses))

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
                detections.append(detection.tolist())

        for gtbox in targets:
            if gtbox[1] == c:
                groundtruths.append(gtbox.tolist())

        #countgtboxs = Counter([int(gt[0]) for gt in groundtruths])
        countgtboxs = {}
        for gtbox in groundtruths:
            idx = int(gtbox[0])
            if idx not in countgtboxs:
                countgtboxs[idx] = 1
            else:
                countgtboxs[idx] += 1

        #print(countgtboxs)
        for k, v in countgtboxs.items():
            countgtboxs[k] = torch.zeros(v)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        totalgts = len(groundtruths)

        if totalgts == 0:
            continue

        for i, detection in enumerate(detections):
            imggts = [gt for gt in groundtruths if gt[0] == detection[0]]
            bestiou = 0
            for j, gtbox in enumerate(imggts):
                iou = bbox_iou(torch.tensor(detection[3:]), torch.tensor(gtbox[3:]))

                if iou > bestiou:
                    bestiou = iou
                    bestgtidx = j

            if bestiou > iouthresh:
                if countgtboxs[detection[0]][bestgtidx] == 0:
                    TP[i] = 1
                    countgtboxs[detection[0]][bestgtidx] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TPcumsum = torch.cumsum(TP, dim=0)
        FPcumsum = torch.cumsum(FP, dim=0)
        recalls = TPcumsum / (totalgts + epsilon)
        precisions = TPcumsum / (TPcumsum + FPcumsum + epsilon)
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        aps.append(torch.trapz(precisions, recalls))

    return sum(aps) / len(aps)


