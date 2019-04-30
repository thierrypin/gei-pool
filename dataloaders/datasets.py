#!/home/thierry/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
from random import randint

from sklearn.preprocessing import LabelEncoder

# from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import Dataset
from mxnet import nd
# output[i] = (input[i] - mi) / si


def default_preprocess(arr):
    arr = arr.astype(np.float32) / 128.
    arr -= 1.
    return arr

def preprocess(samples):
    if isinstance(samples, np.ndarray):
        samples = [samples]
    for im in samples:
        assert isinstance(im, np.ndarray), "Expect np.ndarray, got {}".format(type(im))

    for j in range(len(samples)):
        samples[j] = samples[j].astype(np.float32) / 255.


# Generic Numpy .npy loader
class ArrayData(Dataset):
    def __init__(self, list_path, nb_frames=4, augmenter=None, eager=False, encoder=None, testing=False):
        self.eager = eager
        self.nb_frames = nb_frames

        ###
        # Specific part
        self.augmenter = augmenter
        self.testing = testing
        self.files = np.loadtxt(list_path, delimiter=';', dtype=str)
        
        ###
        # Encoding
        labels = np.array([ path.split('-')[0] for path in self.files ])

        if encoder is None:
            self.encoder = LabelEncoder()
            self.ground_truth = self.encoder.fit_transform(labels)
        else:
            self.encoder = encoder
            self.ground_truth = encoder.transform(labels)
        
        self.nb_classes = len(set(self.ground_truth))

        # Image data
        data_path = os.path.dirname(list_path)
        if self.eager:
            # Array processing is easier with np.ndarray
            self.data = [ np.load(os.path.join(data_path, video)) for video in self.files ]
            self.getter = self.eager_getter
            d = self.data[0]
        else:
            self.data = [ os.path.join(data_path, video) for video in self.files ]
            self.getter = self.lazy_getter
            d = np.load(self.data[0])

        self.shape = (1,) + tuple(d.shape[1:])

    def gei(self, video):
        seq_size = video.shape[0]

        if seq_size > self.nb_frames:
            start = randint(0, seq_size - self.nb_frames)
            end = start + self.nb_frames
            tmp = video[start:end].astype(np.float32) / 255.
            if self.augmenter:
                tmp = self.augmenter(tmp)
            return tmp.mean(axis=0)[np.newaxis , ...]
        else:
            tmp = video.astype(np.float32) / 255.
            if self.augmenter:
                tmp = self.augmenter(tmp)
            return tmp.mean(axis=0)[np.newaxis , ...]

    def get_nb_samples(self):
        return len(self.data)

    def lazy_getter(self, idx):
        if self.testing:
            return np.load(self.data[idx]), self.ground_truth[idx]

        gei_img = self.gei(np.load(self.data[idx]))
        ground_truth = self.ground_truth[idx]

        return nd.array(gei_img), ground_truth

    def eager_getter(self, idx):
        if self.testing:
            return self.data[idx], self.ground_truth[idx]

        gei_img = self.gei(self.data[idx])
        ground_truth = self.ground_truth[idx]
        
        return nd.array(gei_img), ground_truth

    def get_encoder(self):
        return self.encoder

    def __getitem__(self, key):
        return self.getter(key)
    
    def __len__(self):
        return self.files.shape[0]

