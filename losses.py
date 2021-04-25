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
    def __init__(self, imgwh):
        super(BboxLoss, self).__init__()
        self.imgwh = imgwh
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss()

        self.box = 1
        self.boxscale = self.box
        self.objscale = 1
        self.noobjscale = 100
        self.classscale = 1

        self.ignthresh = 0.5

    def __call__(self, output, target, anchors):
        ##assumes output is a rank-5 tensor (N, A, G, G, C+5)
        
        nclasses = output.size(-1) - 5 
        predboxes, predconfs, predclasses = torch.split(output, (4, 1, nclasses), -1)
        predconfs = predconfs.squeeze(-1)

        self.anchors = anchors

        self.gridx, self.gridy, self.sclanchors, self.stride = compute_grid(
            self.imgwh, predboxes.size(2), self.anchors, output.is_cuda
        )

        objmask, noobjmask, ttx, tty, ttw, tth, trueclasses, trueconfs =  \
                build_targets(
                    target, self.sclanchors, predclasses, 
                    predboxes, self.ignthresh, mode='logitbox'
                )

        ptx, pty = torch.sigmoid(predboxes[..., 0]), torch.sigmoid(predboxes[..., 1]) 
        ptw, pth = predboxes[..., 2], predboxes[..., 3]

        xloss = self.mseloss(ptx[objmask], ttx[objmask])
        yloss = self.mseloss(pty[objmask], tty[objmask])
        wloss = self.mseloss(ptw[objmask], ttw[objmask])
        hloss = self.mseloss(pth[objmask], tth[objmask])

        boxloss = xloss + yloss + wloss + hloss

        objloss = self.bceloss(predconfs[objmask], trueconfs[objmask]) 
        noobjloss = self.bceloss(predconfs[noobjmask], trueconfs[noobjmask])
        confloss = self.objscale * objloss + self.noobjscale * noobjloss

        classloss = self.classscale * self.bceloss(predclasses[objmask], trueclasses[objmask])

        return boxloss, confloss, classloss

class IoULoss:
    def __init__(self, hyp):
        super(IoULoss, self).__init__()
        self.imgwh = hyp['imgsize']

        self.hyp = hyp

        bcecls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['clspw']))
        bceobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['objpw']))

        self.bcecls, self.bceobj = FocalLoss(bcecls), FocalLoss(bceobj)

        self.obj, self.cls, self.box = hyp['obj'], hyp['cls'], hyp['box'] 

        '''self.boxscale = 0.05
        self.objscale = 50
        self.noobjscale = 100
        self.classscale = 1'''

        self.ignthresh = 0.5

    def __call__(self, output, target, anchors):
        ##assumes output is a rank-5 tensor (N, A, G, G, C+5) 

        nclasses = output.size(-1) - 5
        predboxes, predconfs, predclasses = torch.split(output, (4, 1, nclasses), -1)
        predconfs = predconfs.squeeze(-1)

        self.anchors = anchors

        self.gridx, self.gridy, self.sclanchors, self.stride = compute_grid(
            self.imgwh, predboxes.size(2), self.anchors, output.is_cuda
        )

        objmask, noobjmask, tbx, tby, tbw, tbh, trueclasses, trueconfs =  \
                build_targets(
                    target, self.sclanchors, predclasses, 
                    predboxes, self.ignthresh, mode='probbox'
                )
    
        numanchors = len(self.anchors)

        pbx, pby = torch.sigmoid(predboxes[..., 0]), torch.sigmoid(predboxes[..., 1])
        pbw = torch.exp(predboxes[..., 2]) * self.sclanchors[:, 0:1].view((1, numanchors, 1, 1))
        pbh = torch.exp(predboxes[..., 3]) * self.sclanchors[:, 1:2].view((1, numanchors, 1, 1))
        predboxes = torch.stack([pbx, pby, pbw, pbh], -1)
        trueboxes = torch.stack([tbx, tby, tbw, tbh], -1)

        iou = calc_ious(predboxes[objmask], trueboxes[objmask], x1y1x2y2=False, mode='ciou')

        boxloss = (1.0 - iou).mean()

        objloss = self.bceobj(predconfs, trueconfs) #self.flbce(predconfs, trueconfs)
        #noobjloss = self.flbce(predconfs[noobjmask], trueconfs[noobjmask])
        #confloss = self.objscale * objloss + self.noobjscale * noobjloss
        confloss = self.obj * objloss

        classloss = self.cls * self.bcecls(predclasses[objmask], trueclasses[objmask])

        return boxloss, confloss, classloss

class ComputeLoss:
    def __init__(self, model, hyp, loss):
        super(ComputeLoss, self).__init__()
        self.loss = loss(hyp)

        self.imgwh = model.imgwh
        self.hyperparams = model.hyperparams

    def __call__(self, outputs, targets):
        preds, allanchors = outputs

        boxloss, confloss, classloss = 0, 0, 0
        for i, pi in enumerate(preds):
            lbox, lconf, lclass = self.loss(pi, targets, allanchors[i])
            boxloss += lbox
            confloss += lconf
            classloss += lclass

        boxloss *= self.loss.box
        totalloss  = boxloss + confloss + classloss

        return totalloss


