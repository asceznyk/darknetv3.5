import numpy as np

import torch

from torchvision import transforms

from torchvision.transforms import ToTensor

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from utils import *

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])
        
        # Convert bounding boxes to imgaug        
        bboxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes], 
            shape=img.shape)

        # Apply augmentations
        img, bboxes = self.augmentations(
            image=img, 
            bounding_boxes=bboxes)

        # Clip out of image boxes
        bboxes = bboxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bboxes), 5))
        for i, box in enumerate(bboxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[i, 0] = box.label
            boxes[i, 1] = ((x1 + x2) / 2)
            boxes[i, 2] = ((y1 + y2) / 2)
            boxes[i, 3] = (x2 - x1)
            boxes[i, 4] = (y2 - y1)

        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        w, h, _ = img.shape 
        boxes[:,[1,3]] /= h
        boxes[:,[2,4]] /= w
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        w, h, _ = img.shape 
        boxes[:,[1,3]] *= h
        boxes[:,[2,4]] *= w
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
            ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        img = transforms.ToTensor()(img)

        bbtargets = torch.zeros((len(boxes), 6))
        bbtargets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bbtargets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.2)),
            iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2,0.2)),
            iaa.AddToBrightness((-30, 30)), 
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ], random_order=True)

DEFAULTTRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

AUGMENTATIONTRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])


