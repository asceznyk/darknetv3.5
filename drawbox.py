import os
import cv2
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from torchvision import transforms

from general import *
from augmentations import AUGMENTATIONTRANSFORMS, DEFAULTTRANSFORMS, Resize
from datasets import ImageFolder

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

def predict_boxes(model, names, directory, imgsize, batchsize, savedir, color=(0, 255, 0)):
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

        _, predimg = load_img(path[0], imgsize)

        model.eval()
        preds = model(img)
        bboxes = nonmax_supression(preds, confthresh=0.8)[0]

        if bboxes is not None:
            predimg = draw_outputs(predimg, bboxes.numpy(), names, color)

        cv2.imwrite(savedir+'/'+path[0].split('/')[1], predimg)

def show_boxes(names, directory, imgsize, savedir):
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

        cv2.imwrite(savedir+'/'+pimg.split('/')[1], predimg)


