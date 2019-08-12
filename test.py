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

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

import numpy as np

from pyopf import OPFClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

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


def main(dest_fold, dest_angle, ckpt_folder, nb_frames, eager, params=None, **kwargs):

    print("Unused arguments:", kwargs)

    # Find information about the model
    expr = r"(?P<set>\w+)-(?P<model>[\w_]+)-(?P<fold>\d{2})-(?P<angle>\d{2})"
    m = re.search(expr, ckpt_folder)
    if m:
        setname = m.group('set')
        modelname = m.group('model')
        fold = m.group('fold')
        angle = m.group('angle')

    # setname = gallery_list.split(os.sep)[0]

    # Output folder
    xp_folder = "tests/%s-%s_%s>%s_%s>%s" % (setname, modelname, fold, dest_fold, angle, dest_angle)
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
    gallery_list = "OULP/CV%s.txt_gallery_%s" % (dest_fold, dest_angle)
    gallery_data = ArrayData(gallery_list, nb_frames=nb_frames, augmenter=None, eager=eager, testing=True)
    print("Gallery size", gallery_data[0][0].shape)
    nb_gallery = len(gallery_data) # loader should provide the number of sampĺes

    probe_list = "OULP/CV%s.txt_probe_%s" % (dest_fold, dest_angle)
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
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(ckpt_folder, model_path), epoch)
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
    model.bind(for_training=False, data_shapes=[('data', (1, 1, 64, 44))])
    model.set_params(arg_params, aux_params)


    # A little more verbosity
    print("************************************")
    print(nb_gallery, "gallery samples,")
    print(nb_probe, "probe samples,")
    print("************************************")


    ###########
    # Testing #
    ###########
    train_data = []
    train_labels = []

    # Time measurement
    start_time = time()

    # Compute "train" features
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
    
    gallery_time = time() - start_time

    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)

    print("train_size", train_data.shape, train_labels.shape)

    test_data = []
    test_labels = []

    start_probe = time()
    # Compute "test" features
    for i in tqdm(range(nb_probe), desc='Computing probe features'):
        video, label = probe_data[i]

        # Split video in clips and build their GEIs
        data = batchify(video)
        data = nd.array(data, ctx=mx.gpu(0))

        # They all have the same label
        nb_gei = data.shape[0]
        # label = np.repeat(label, nb_gei)

        # Compute outputs and accuracy
        model.forward(Batch([data]))
        preds = model.get_outputs()[0].asnumpy()

        test_data.append(preds)
        test_labels.append(label)
    
    probe_time = time() - start_probe

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    print("test_size", test_data.shape, test_labels.shape)
    print()

    # Nearest Neighbors
    nn_start = time()
    nn = KNeighborsClassifier(1, n_jobs=-1)
    nn.fit(train_data, train_labels)
    nn_fit = time()
    
    nn_preds = []
    for sample in tqdm(test_data, desc="Nearest neighbors predictions"):
        out = nn.predict(sample)
        nn_preds.append(mode(out).mode[0])
    nn_preds = np.array(nn_preds)

    nn_end = time()

    nn_fit_time = nn_fit - nn_start
    nn_predict_time = nn_end - nn_fit

    nn_acc = accuracy_score(nn_preds, test_labels)
    print("NN accuracy:", nn_acc)
    print("nn train", nn_fit_time, ", nn_test", nn_predict_time)
    print()

    # OPF
    opf_start = time()
    opf = OPFClassifier()
    opf.fit(train_data, train_labels)
    opf_fit = time()

    opf_preds = []
    for sample in tqdm(test_data, desc="OPF predictions"):
        out = opf.predict(sample)
        opf_preds.append(mode(out).mode[0])
    opf_preds = np.array(opf_preds)

    opf_end = time()

    opf_fit_time = opf_fit - opf_start
    opf_predict_time = opf_end - opf_fit

    opf_acc = accuracy_score(opf_preds, test_labels)
    print("OPF accuracy:", opf_acc)
    print("opf train", opf_fit_time, ", opf_test", opf_predict_time)
    print()


    print("Saving experiment data")
    np.save(os.path.join(xp_folder, 'gt.npy'), test_labels)
    np.save(os.path.join(xp_folder, 'nn_preds.npy'), nn_preds)
    np.save(os.path.join(xp_folder, 'opf_preds.npy'), opf_preds)

    headers = ['gallery_time', 'probe_time', 'nn_fit_time', 'nn_predict_time', 'nn_acc', 'opf_fit_time', 'opf_predict_time', 'opf_acc']
    info = [gallery_time, probe_time, nn_fit_time, nn_predict_time, nn_acc, opf_fit_time, opf_predict_time, opf_acc]
    info = [str(i) for i in info]
    csv_contents = ";".join(headers) + "\n" + ";".join(info)

    with open(os.path.join(xp_folder, 'nn_acc'), 'w') as f:
        f.write(str(nn_acc))
    with open(os.path.join(xp_folder, 'opf_acc'), 'w') as f:
        f.write(str(opf_acc))
    with open(os.path.join(xp_folder, 'info.csv'), 'w') as f:
        f.write(csv_contents)

    hours, rem = divmod(time()-start_time, 3600)
    days, hours = divmod(hours, 24)
    minutes, seconds = divmod(rem, 60)

    print("Testing finished in %dd, %dh%dm%.2fs." % (int(days), int(hours), int(minutes), seconds))


if __name__ == "__main__":
    args = read_params(json_path='params.json', make_json=True)
    main(**args)


