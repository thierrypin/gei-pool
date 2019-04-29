#!/home/thierry/anaconda3/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import math
from tqdm import tqdm
from time import time, gmtime, strftime

from dataloaders.datasets import ArrayData
from utils.params import read_params, write_params
from utils.fs import mkdir_p

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader

import numpy as np
import pickle

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

def main(test_list, exp, saved_model, batch_size, encoder, nb_frames, eager, video, params=None, **kwargs):

    print("Unused arguments:", kwargs)

    setname = test_list.split(os.sep)[0]
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

    # Label encoder
    split = saved_model.split(os.sep)
    enc_path = os.sep.join(split[:split.index('checkpoints')])
    try:
        with open(os.path.join(enc_path, 'encoder.pkl'), 'rb') as f:
            encoder = pickle.load(train_data.get_encoder(), f)
    except:
        encoder = CasiabEncoder()
    print(encoder)
    
    # Dataset classes
    test_data = ArrayData(test_list, nb_frames=nb_frames, augmenter=None, eager=eager, video=video, encoder=encoder)

    # Train loader
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=6)
    nb_samples = len(test_data) # loader should provide the number of sampĺes

    # Compute number of steps
    steps = math.ceil(nb_samples / batch_size)

    # The model
    symbols = saved_model[:-11] + "symbol.json"
    net = nn.SymbolBlock.imports(symbols, ['data'], saved_model, ctx=mx.gpu())

    # A little more verbosity
    print("************************************")
    print("Batch size:", batch_size)
    print(nb_samples, "testing samples,", steps, "steps")
    print("************************************")


    ###########
    # Testing #
    ###########
    progress_desc = "Acc %.3f        "
    start_time = time()

    gts = []
    outs = []
    # calculate testing accuracy
    prog = tqdm(test_loader, desc='Running test', unit='batch')
    for data, label in prog:
        data = data.copyto(mx.gpu(0))
        label = label.copyto(mx.gpu(0))

        # Compute outputs and accuracy
        output = net(data)
        preds = output.softmax().argmax(axis=1)

        gts.append(label.copyto(mx.cpu()).asnumpy()[..., 0])
        outs.append(preds.copyto(mx.cpu()).asnumpy())

    gts = np.hstack(gts)
    outs = np.hstack(outs)
    print(gts.shape, outs.shape)

    acc = (gts == outs).mean()
    print("Accuracy:", acc)

    hours, rem = divmod(time()-start_time, 3600)
    days, hours = divmod(hours, 24)
    minutes, seconds = divmod(rem, 60)

    print("Testing finished in %dd, %dh%dm%.2fs." % (int(days), int(hours), int(minutes), seconds))


if __name__ == "__main__":
    args = read_params(json_path='params.json', make_json=True)
    main(**args)


