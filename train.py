#!/home/thierry/anaconda3/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import math
from tqdm import tqdm
from time import time, gmtime, strftime

from utils.callbacks import ModelCheckpoint, HistoryKeeper, EarlyStopper, TerminateOnNaN

from dataloaders.datasets import ArrayData
from models.models import ResearchModels
from utils.params import read_params, write_params
from utils.fs import mkdir_p
from dataloaders.augmentation import default_augmenter_vid

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.optimizer import SGD, Adam, Nadam
from mxnet.gluon.data import DataLoader
from mxnet.metric import Accuracy

import pickle

def accuracy(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    print(output.argmax(axis=1), label, output.argmax(axis=1) == label.astype('float32'))
    # return (output.argmax(axis=1) == label).mean().asscalar()

def main(train_list, val_list, model, exp, saved_model, batch_size, optimizer, nb_epochs, augment, max_lr, min_lr, loss_function, train_all, nb_frames, eager, params=None, **kwargs):

    print("Unused arguments:", kwargs)

    setname = train_list.split(os.sep)[0]
    # Timestamp to name experiment folder
    xptime = strftime("%Y-%m-%d_%Hh%Mm%Ss", gmtime())
    xp_folder = "experiments/%s-%s-%s_%s" % (setname, model, exp, xptime)
    # Make folder
    mkdir_p(xp_folder)
    mkdir_p(os.path.join(xp_folder, 'checkpoints'))
    mkdir_p(os.path.join(xp_folder, 'tb'))
    print("\nSaving experiment data to:", xp_folder)

    # Save command (as well as possible)
    with open(os.path.join(xp_folder, 'command.sh'), "w") as f:
        command = " ".join(sys.argv[:]) + "\n"
        f.write(command)

    # Save employed parameters for future reference
    if params is not None:
        write_params(os.path.join(xp_folder, 'params.json'), params)

    #############
    # Callbacks #
    #############

    # Helper: Save the model.
    ckpt_fmt = os.path.join(xp_folder, 'checkpoints', model + '-' + exp + '.{epoch:03d}-loss{val_loss:.3f}-acc{val_acc:.3f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=ckpt_fmt, verbose=1, save_best_only=True, monitor='val_acc')

    # Helper: TensorBoard
    tb = HistoryKeeper(logdir=os.path.join(xp_folder), keys=['val_acc', 'val_loss', 'train_time', 'val_time'])

    # Helper: Stop when we stop learning.
    # early_stopper = EarlyStopper(patience=15)

    # Helper: Terminate when finding a NaN loss
    nan_term = TerminateOnNaN()

    callbacks = [tb, checkpointer, nan_term]
    #############

    #############
    #  Loading  #
    #############
    if augment:
        augmenter = default_augmenter_vid(strip_size=4)
    else:
        augment = False
        augmenter = None


    # Dataset classes
    train_data = ArrayData(train_list, nb_frames=nb_frames, augmenter=augmenter, eager=eager)
    val_data = ArrayData(val_list, nb_frames=nb_frames, augmenter=None, eager=eager, encoder=train_data.get_encoder())

    # Saving encoder
    with open(os.path.join(xp_folder, 'encoder.pkl'), 'wb') as f:
        pickle.dump(train_data.get_encoder(), f)

    # Train loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, last_batch='keep', num_workers=10)
    nb_samples = len(train_data) # loader should provide the number of sampĺes

    # Validation loader
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=10)
    nb_validation = len(val_data) # loader should provide the number of sampĺes

    # Compute number of steps
    steps_per_epoch = math.ceil(nb_samples / batch_size)
    validation_steps = math.ceil(nb_validation / batch_size)

    # The model
    net = ResearchModels(train_data.nb_classes, model, saved_model, input_shape=train_data.shape, train_all=train_all).model

    # A little more verbosity
    print("************************************")
    if train_all:
        print("Train all layers.")
    print("Max lr:", max_lr, " Min lr:", min_lr)
    print("Batch size:", batch_size)
    print(nb_samples, "training samples,", steps_per_epoch, "steps per epoch")
    print(nb_validation, "validation samples,", validation_steps, "validation steps")
    print("Optimizer:", optimizer)
    if augment:
        print("Using data augmentation")
    else:
        print("WARNING: Not using data augmentation")
    print("************************************")
    

    ############################
    #   Loss and Optimization  #
    ############################

    trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': max_lr})

    if loss_function == 'categorical_crossentropy':
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        loss_fn.hybridize()


    ############
    # Training #
    ############
    progress_desc = "Super epoch %03d - acc %.3f - loss %.3f  "
    acc = Accuracy()
    start_time = time()

    super_epoch_size = 250
    # Learning rate decay
    iteration = 1
    decay_alpha = 0.01 ** 0.25
    lr = max_lr

    for epoch in range(1, nb_epochs+1):
        train_loss, val_loss = 0., 0.
        nb_batches = 0
        tic = time()
        acc.reset()

        start_training = time()
        t = tqdm(range(super_epoch_size), unit='epochs')
        for _ in t:
            for data, label in train_loader:
                # Learning rate decay
                if iteration % 10000 == 0:
                    lr *= decay_alpha
                    trainer.set_learning_rate(lr)
                    print("Learning rate updated to", lr)
                iteration += 1

                current_batch_size = data.shape[0]
                data = data.copyto(mx.gpu(0))
                label = label.copyto(mx.gpu(0))

                with autograd.record():
                    output = net(data)
                    loss = loss_fn(output, label)
                loss.backward()
                # print(mx.nd.log_softmax(output[0], axis=-1), label[0])

                # update parameters
                trainer.step(current_batch_size)

                # calculate training metrics
                train_loss += loss.mean().asscalar()
                # accuracy(output, label)
                acc.update(preds=output, labels=label)

                nb_batches += 1
            
            t.set_description(progress_desc % (epoch, acc.get()[1], train_loss/nb_batches))
        
        train_time = time() - start_training
        
        train_loss /= steps_per_epoch * super_epoch_size
        train_acc = acc.get()[1]
        
        acc.reset()
        start_val = time()
        # calculate validation accuracy
        tval = tqdm(val_loader, leave=False, desc='Running validation', unit='batch')
        for data, label in tval:
            data = data.copyto(mx.gpu(0))
            label = label.copyto(mx.gpu(0))

            # Compute outputs
            output = net(data)
            loss = loss_fn(output, label)
            
            # Compute metrics
            val_loss += loss.mean().asscalar()
            # val_acc += accuracy(output, label)
            acc.update(preds=output, labels=label)
        
        val_time = time() - start_val
        
        val_loss /= validation_steps
        val_acc = acc.get()[1]

        print("Epoch %d: loss %.3f, acc %.3f, val_loss %.3f, val_acc %.3f, in %.1f sec" % (
            epoch, train_loss, train_acc,
            val_loss, val_acc, time()-tic))
        print("--------------------------------------------------------------------------------")

        stop = False
        train_info = {'epoch':epoch, 'loss': train_loss, 'acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'train_time': train_time, 'val_time': val_time}
        for cb in callbacks:
            if cb(net, train_info):
                stop = True
        
        if stop:
            break
        print()
    
    hours, rem = divmod(time()-start_time, 3600)
    days, hours = divmod(hours, 24)
    minutes, seconds = divmod(rem, 60)
    
    print("%d training epochs in %dd, %dh%dm%.2fs." % (nb_epochs, int(days), int(hours), int(minutes), seconds))


if __name__ == "__main__":
    args = read_params(json_path='params.json', make_json=True)
    main(**args)


