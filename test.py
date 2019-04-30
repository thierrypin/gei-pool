#!/home/thierry/anaconda3/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import os
import re
import sys
import math
from tqdm import tqdm
from time import time, gmtime, strftime

from dataloaders.datasets import ArrayData
from utils.params import read_params, write_params
from utils.fs import mkdir_p

import mxnet as mx
from mxnet import nd

import numpy as np

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

from pyopf import OPFClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

class CasiabEncoder:
    def transform(self, labels):
        transformed = np.empty_like(labels)
        for i in range(labels.shape[0]):
            transformed[i] = int(labels[i])-1
        
        return nd.array(transformed)

def accuracy(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    print(output.argmax(axis=1), label, output.argmax(axis=1) == label.astype('float32'))
    # return (output.argmax(axis=1) == label).mean().asscalar()

def gei(video):
    tmp = video.astype(np.float32) / 255.
    return tmp.mean(axis=0)[np.newaxis , ...]

def batchify(video, gei_size=4):
    nb_frames = video.shape[0]
    
    batch = []
    for i in range(0, nb_frames, gei_size):
        batch.append(gei(video[i:i+gei_size]))
    
    return np.array(batch) # expand_dims axis=1


def main(gallery_list, probe_list, exp, ckpt_folder, batch_size, nb_frames, eager, params=None, **kwargs):

    print("Unused arguments:", kwargs)

    setname = gallery_list.split(os.sep)[0]
    # Timestamp to name experiment folder
    xptime = strftime("%Y-%m-%d_%Hh%Mm%Ss", gmtime())
    xp_folder = "tests/%s-%s_%s" % (setname, exp, xptime)
    # Make folder
    mkdir_p(xp_folder)
    print("\nSaving experiment data to:", xp_folder)

    # Save command (as well as possible)
    with open(os.path.join(xp_folder, 'command.sh'), "w") as f:
        command = " ".join(sys.argv[:]) + "\n"
        f.write(command)

    # Save employed parameters for future reference
    if params is not None:
        write_params(os.path.join(xp_folder, 'params.json'), params)

    #############################
    #          Loading          #
    #############################
    
    # Dataset classes
    gallery_data = ArrayData(gallery_list, nb_frames=nb_frames, augmenter=None, eager=eager, testing=True)
    print("Gallery size", gallery_data[0][0].shape)
    nb_gallery = len(gallery_data) # loader should provide the number of sampĺes

    probe_data = ArrayData(probe_list, nb_frames=nb_frames, augmenter=None, eager=eager, encoder=gallery_data.get_encoder(), testing=True)
    print("Probe size", probe_data[0][0].shape)
    nb_probe = len(probe_data) # loader should provide the number of sampĺes

    # Find the best model
    regex = re.compile(r'hdf5-(?P<epoch>\d{4}).params')
    epoch = 0
    model_path = None
    for p in os.listdir(ckpt_folder):
        m = regex.search(p)
        if m:
            e = int(m.group('epoch'))
            if epoch < e:
                epoch = e
                model_path = p[:-12]
    
    print("Model path", model_path)

    # Loading complete model and its weights
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
    all_layers = sym.get_internals() # feature layer

    # Get the feature layer output
    out_list = all_layers.list_outputs()[-10:]
    layername = None
    for out in out_list:
        if 'feature_relu_fwd_output' in out:
            layername = out
            break

    # Feature extraction network
    sym3 = all_layers[layername]
    model = mx.mod.Module(symbol=sym3, label_names=None, context=mx.gpu())
    model.bind(for_training=False, data_shapes=[('data', (1,1,64,44))])
    model.set_params(arg_params, aux_params)


    # A little more verbosity
    print("************************************")
    print(nb_gallery, "gallery samples,")
    print(nb_probe, "probe samples,")
    print("************************************")


    ###########
    # Testing #
    ###########
    start_time = time()

    train_data = []
    train_labels = []
    # calculate testing accuracy
    for i in tqdm(range(nb_gallery), desc='Computing gallery features'):
        video, label = gallery_data[i]

        # Split video in clips and build their GEIs
        data = batchify(video)
        data = nd.array(data, ctx=mx.gpu(0))

        # They all have the same label
        nb_gei = data.shape[0]
        label = np.repeat(label, nb_gei)

        # Compute outputs and accuracy
        model.forward(Batch([data]))
        preds = model.get_outputs()[0].asnumpy()

        train_data.append(preds)
        train_labels.append(label)

    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)

    print("train_size", train_data.shape, train_labels.shape)

    test_data = []
    test_labels = []
    # calculate testing accuracy
    for i in tqdm(range(nb_probe), desc='Computing probe features'):
        video, label = probe_data[i]

        # Split video in clips and build their GEIs
        data = batchify(video)
        data = nd.array(data, ctx=mx.gpu(0))

        # They all have the same label
        nb_gei = data.shape[0]
        label = np.repeat(label, nb_gei)

        # Compute outputs and accuracy
        model.forward(Batch([data]))
        preds = model.get_outputs()[0].asnumpy()

        test_data.append(preds)
        test_labels.append(label)

    test_data = np.vstack(test_data)
    test_labels = np.hstack(test_labels)

    print("test_size", test_data.shape, test_labels.shape)

    print()
    nn_start = time()
    nn = KNeighborsClassifier(1)
    nn.fit(train_data, train_labels)
    nn_fit = time()
    preds = nn.predict(test_data)
    nn_end = time()

    nn_acc = accuracy_score(preds, test_labels)
    print("NN accuracy:", nn_acc)
    print("nn train", nn_fit - nn_start, ", nn_test", nn_end - nn_fit)

    opf_start = time()
    opf = OPFClassifier()
    opf.fit(train_data, train_labels)
    opf_fit = time()
    preds = opf.predict(test_data)
    opf_end = time()

    opf_acc = accuracy_score(preds, test_labels)
    print("OPF accuracy:", opf_acc)
    print("opf train", opf_fit - opf_start, ", opf_test", opf_end - opf_fit)

    hours, rem = divmod(time()-start_time, 3600)
    days, hours = divmod(hours, 24)
    minutes, seconds = divmod(rem, 60)

    print("Testing finished in %dd, %dh%dm%.2fs." % (int(days), int(hours), int(minutes), seconds))


if __name__ == "__main__":
    args = read_params(json_path='params.json', make_json=True)
    main(**args)


