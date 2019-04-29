#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np

import mxnet as mx
from mxnet.gluon import nn


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--lists_folder", action='store', help="Folder to find lists.", type=str)
    parser.add_argument("--saved_model", action='store', help="Folder to find lists.", type=str)
    parser.add_argument("--output_folder", action='store', help="Folder to output features.", type=str)

    return parser.parse_args()


def get_features_per_list(model, list_path):
    lst = np.loadtxt(list_path, dtype=str)





def main():
    args = parse_args()

    print("Loading model %s" % args.saved_model)
    symbols = args.saved_model[:-11] + "symbol.json"
    # self.model.load_parameters(saved_model, ctx=mx.gpu())
    model = nn.SymbolBlock.imports(symbols, ['data'], args.saved_model, ctx=mx.gpu())



if __name__ == "__main__":
    main()


