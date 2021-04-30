import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class FocalLoss(nn.Module):
    def __init__(self, lossfn, alpha=0.25, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.lossfn = lossfn
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = lossfn.reduction
        self.lossfn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.lossfn(pred, true)
        prob = torch.sigmoid(pred)
        pt = (1 - true) * (1 - prob) + true * prob
        alphat = (1 - self.alpha) * (1 - true) + self.alpha * true

        loss *= alphat * ((1 - pt) ** self.gamma)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.redcution == 'sum':
            return loss.sum()
        else:
            return loss

class BboxLoss:
    def __init__(self, hyp):
        super(BboxLoss, self).__init__()
        self.imgwh = hyp['imgsize']
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss()

        self.box, self.obj, self.noobj, self.cls = 1, 1, 100, 1

        self.ignthresh = 0.5

    def __call__(self, outputs, target):
        boxloss, confloss, classloss = 0, 0, 0
        preds, allanchors = outputs
        for i, pi in enumerate(preds):
            anchors = allanchors[i]
            nclasses = pi.size(-1) - 5
            predboxes, predconfs, predclasses = torch.split(pi, (4, 1, nclasses), -1)
            predconfs = predconfs.squeeze(-1)

            gridx, gridy, sclanchors, self.stride = compute_grid(
                self.imgwh, predboxes.size(2), anchors, pi.is_cuda
            )

            objmask, noobjmask, ttx, tty, ttw, tth, trueclasses, trueconfs =  \
                    build_targets(
                        target, sclanchors, predclasses,
                        predboxes, self.ignthresh, mode='logitbox'
                    )

            ptx, pty = torch.sigmoid(predboxes[..., 0]), torch.sigmoid(predboxes[..., 1])
            ptw, pth = predboxes[..., 2], predboxes[..., 3]

            xloss = self.mseloss(ptx[objmask], ttx[objmask])
            yloss = self.mseloss(pty[objmask], tty[objmask])
            wloss = self.mseloss(ptw[objmask], ttw[objmask])
            hloss = self.mseloss(pth[objmask], tth[objmask])

            boxloss += self.box * (xloss + yloss + wloss + hloss)

            objloss = self.bceloss(predconfs[objmask], trueconfs[objmask])
            noobjloss = self.bceloss(predconfs[noobjmask], trueconfs[noobjmask])
            confloss += self.obj * objloss + self.noobj * noobjloss

            classloss += self.cls * self.bceloss(predclasses[objmask], trueclasses[objmask])

        return boxloss + confloss + classloss

class IoULoss:
    def __init__(self, hyp):
        super(IoULoss, self).__init__()
        self.hyp = hyp
        self.imgwh = hyp['imgsize'] 

        bcecls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['clspw']))
        bceobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['objpw']))

        g = hyp['flgamma']
        self.bcecls, self.bceobj = FocalLoss(bcecls, gamma=g), FocalLoss(bceobj, gamma=g)
        self.ignthresh = 0.5

    def __call__(self, outputs, target):
        boxloss, objloss, clsloss = 0, 0, 0
        preds, allanchors = outputs
        for i, pi in enumerate(preds):
            anchors = allanchors[i]
            N = pi.size(0)
            C = pi.size(-1) - 5
            predboxes, predconfs, predclasses = torch.split(pi, (4, 1, C), -1)
            predconfs = predconfs.squeeze(-1)

            gridx, gridy, sclanchors, self.stride = compute_grid(
                self.imgwh, predboxes.size(2), anchors, pi.is_cuda
            )

            objmask, noobjmask, tbx, tby, tbw, tbh, trueclasses, trueconfs =  \
                    build_targets(
                        target, sclanchors, predclasses,
                        predboxes, self.ignthresh, mode='probbox'
                    )

            numanchors = len(anchors)

            pbx, pby = torch.sigmoid(predboxes[..., 0]), torch.sigmoid(predboxes[..., 1])
            pbw = torch.exp(predboxes[..., 2]) * sclanchors[:, 0:1].view((1, numanchors, 1, 1))
            pbh = torch.exp(predboxes[..., 3]) * sclanchors[:, 1:2].view((1, numanchors, 1, 1))
            predboxes = torch.stack([pbx, pby, pbw, pbh], -1)
            trueboxes = torch.stack([tbx, tby, tbw, tbh], -1)

            ious = calc_ious(predboxes[objmask], trueboxes[objmask], x1y1x2y2=False, mode='ciou')
            boxloss += (1.0 - ious).mean()

            trueconfs[objmask] = ious
            objloss += self.bceobj(predconfs, trueconfs)

            clsloss += self.bcecls(predclasses[objmask], trueclasses[objmask])

        boxloss *= self.hyp['box']
        objloss *= self.hyp['obj']
        clsloss *= self.hyp['cls']

        return N * (boxloss + objloss + clsloss)


