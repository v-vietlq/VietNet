import imgaug.augmenters as iaa
import random

import numpy as np
import cv2
from PIL import Image


aug_transform = iaa.SomeOf((0, None), [
    iaa.OneOf([
        iaa.MultiplyAndAddToBrightness(mul=(0.3, 1.6), add=(-50, 50)),
        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
        iaa.ChannelShuffle(0.5),
        iaa.RemoveSaturation(),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.ChangeColorTemperature((1100, 35000)),
    ]),
    iaa.OneOf([
                iaa.MedianBlur(k=(3, 7)),
                iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                iaa.MotionBlur(k=(3, 9), angle=[-45, 45]),
                iaa.MeanShiftBlur(spatial_radius=(5.0, 10.0), color_radius=(5.0, 10.0)),
                iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                iaa.AllChannelsHistogramEqualization(),
                iaa.GammaContrast((0.5, 1.5), per_channel=True),
                iaa.GammaContrast((0.5, 1.5)),
                iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                iaa.HistogramEqualization(),
                iaa.Sharpen(alpha=0.5)
            ]),
    iaa.OneOf([
        iaa.AveragePooling([2, 3]),
        iaa.MaxPooling(([2,3],[2,3])),                
    ]),
    iaa.OneOf([
        iaa.Clouds(),
        iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        iaa.Rain(speed=(0.1, 0.3))
    ])
        ], random_order=True)
def get_color_augmentation(augment_prob):
    return iaa.Sometimes(augment_prob, aug_transform).augment_image



class SegCompose(object):
    def __init__(self, augmenters):
        super().__init__()
        self.augmenters = augmenters

    def __call__(self, image, label):
        for augmenter in self.augmenters:
            image, label = augmenter(image, label)
        return image, label

class Resize(object):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def __call__(self, image, label):
        width, height = self.size
        h, w = image.shape[0], image.shape[1]
        if width == -1:
            width = int(height/h * w)
        if height == -1:
            height = int(width/w * h)
        
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        label = label if label is None else cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
        return image, label

class RandomCrop(object):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def __call__(self, image, label):
        max_x = image.shape[1] - self.size[0]
        max_y = image.shape[0] - self.size[1]
        
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        image = image[y: y + self.size[1], x: x + self.size[0]]
        label = label if label is None else label[y: y + self.size[1], x: x + self.size[0]]
        
        return image, label

class RandomRotate(object):
    def __init__(self, max_angle):
        super().__init__()
        self.max_angle = max_angle
    
    def __call__(self, image, label):
        
        angle = random.randint(0, self.max_angle * 2) - self.max_angle
        
        image = Image.fromarray(image)
        image = image.rotate(angle, resample=Image.BILINEAR)
        image = np.array(image)
        
        if label is not None:
            label = Image.fromarray(label)
            label = label.rotate(angle, resample=Image.NEAREST)
            label = np.array(label)
           
        return image, label

class RandomFlip(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image, label):
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label