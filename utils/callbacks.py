import os
from math import isnan

import operator

from mxboard import SummaryWriter

def better(name):
    if 'acc' in name:
        return operator.gt, float('-inf')
    elif 'loss' in name:
        return operator.lt, float('inf') 


class ModelCheckpoint():
    def __init__(self, filepath, verbose=1, save_best_only=True, monitor='val_acc'):
        self.filepath = filepath
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.better, self.best_val = better(monitor)

    # Return True to interrupt training
    def __call__(self, model, params):
        save = False
        if self.save_best_only:
            if self.better(params[self.monitor], self.best_val):
                save = True
                self.best_val = params[self.monitor]
            else:
                print("{0} did not improve from {1} ({2})".format(self.monitor, self.best_val, params[self.monitor]))
        else:
            save = True
        
        if save:
            name = self.filepath.format(**params)
            model.export(name, epoch=params['epoch'])
            print("Model saved to", name + "-%04d.params"%params['epoch'])
    # nn.SymbolBlock.imports('gru-symbol.json', ['data'], 'gru-0000.params')


class HistoryKeeper():
    def __init__(self, logdir, keys=['val_acc', 'val_loss']):
        if not isinstance(keys, (list, tuple)):
            raise ValueError("Keys should be a list or a tuple.")
        self.keys = keys
        
        self.sw = SummaryWriter(logdir=os.path.join(logdir, 'tb'))
        self.csv_path = os.path.join(logdir, 'history.csv')

        with open(self.csv_path, 'w') as f:
            f.write(";".join(keys))
    
    # Return True to interrupt training
    def __call__(self, model, params):
        epoch = params['epoch']
        pars_ = []
        for key in self.keys:
            if key in params:
                self.sw.add_scalar(key, params[key], epoch)
                pars_.append(str(params[key]))

        with open(self.csv_path, 'a') as f:
            f.write(";".join(pars_))


class EarlyStopper():
    def __init__(self, patience=25, monitor='val_acc'):
        self.patience = patience
        self.monitor = monitor
        self.better, self.best_val = better(monitor)
        self.count = 0
    
    # Return True to interrupt training
    def __call__(self, model, params):
        if self.better(params[self.monitor], self.best_val):
            self.count = 0
            self.best_val = params[self.monitor]
        else:
            self.count += 1
        
        if self.count > self.patience:
            return True
    

class TerminateOnNaN():
    # def __init__(self):
    
    # Return True to interrupt training
    def __call__(self, model, params):
        if isnan(params['val_acc']) or isnan(params['acc']):
            return True



