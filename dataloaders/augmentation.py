#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np


# Generic data augmentation
class Augmenter:
    """ Generic data augmentation class with chained operations
    """

    def __init__(self, ops=[]):
        if not isinstance(ops, list):
            print("Error: ops must be a list of functions")
            quit()
        self.ops = ops
    
    def add(self, op):
        self.ops.append(op)

    def augment(self, img):
        aug = img.copy()
        for op in self.ops:
            aug = op(aug)
        return aug
    
    def __call__(self, img):
        return self.augment(img)

##########
# Images #
##########
def horizontal_flip(p=0.5):
    def fc(img):
        if random.random() < p:
            return img[..., ::-1]
        else:
            return img
    return fc

def vertical_flip(p=0.5):
    def fc(img):
        if random.random() < p:
            return img[..., ::-1, :]
        else:
            return img
    return fc

def gaussian_noise(p=0.5, mean=0, sigma=0.02):
    def fc(img):
        if random.random() < p:
            gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
            return img + gauss
        else:
            return img
    return fc

def black_vstripe(p=0.5, size=10):
    def fc(img):
        if random.random() < p:
            j = int(random.random() * (img.shape[1]-size))
            img[..., j:j+size] = 0
            return img
        else:
            return img
    return fc

def black_hstripe(p=0.5, size=10):
    def fc(img):
        if random.random() < p:
            j = int(random.random() * (img.shape[0]-size))
            img[..., j:j+size, :] = 0
            return img
        else:
            return img
    return fc


def default_augmenter(p=0.5, strip_size=3, mean=0, sigma=0.02):
    """Default data augmentation with horizontal flip, vertical flip, gaussian noise, black hstripe, and black vstripe.
    
    Returns:
        Augmenter object. Use as: aug.augment(img)
    """
    print("Using default image augmenter")
    return Augmenter([ horizontal_flip(p), gaussian_noise(p, mean, sigma), black_hstripe(p, size=strip_size), black_vstripe(p, size=strip_size) ])


##########
# Videos #
##########

def horizontal_flip_vid(p=0.5):
    def fc(vid):
        if random.random() < p:
            return vid[..., ::-1]
        else:
            return vid
    return fc

def black_vstripe_vid(p=0.5, size=10):
    def fc(batch):
        if random.random() < p:
            j = int(random.random() * (batch.shape[-1]-size))
            batch[..., j:j+size] = 0
            return batch
        else:
            return batch
    return fc

def black_hstripe_vid(p=0.5, size=10):
    def fc(batch):
        if random.random() < p:
            j = int(random.random() * (batch.shape[-2]-size))
            batch[..., j:j+size, :] = 0
            return batch
        else:
            return batch
    return fc

def default_augmenter_vid(p=0.5, strip_size=3, mean=0, sigma=0.02):
    """Default data augmentation with horizontal flip, gaussian noise, black hstripe, and black vstripe.
    
    Returns:
        Augmenter object. Use as: aug.augment(img)
    """

    return Augmenter([ horizontal_flip_vid(p), gaussian_noise(p, mean, sigma), black_hstripe_vid(p, size=strip_size), black_vstripe_vid(p, size=strip_size) ])



