#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

# from pydoc import locate

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_params(json_path='params.json', make_json=False):
    """Read the parameters from json file and command line arguments.
    
    Keyword Arguments:
        json_path {str} -- Path to json file (default: {'params.json'})
    
    Returns:
        args -- Dictionary of arguments
    """
    
    args = {} # output

    # Getting default parameters
    with open(json_path, 'r') as fp: 
        json_args = json.load(fp)
    
    # Setting up command line argument parser
    parser = argparse.ArgumentParser(description=json_args['description'])
    parser.add_argument("--exp", action='store', type=str, help='Experiment description', required=False)

    # Populate parameters and parser
    for key, content in json_args.items():
        if key != 'description':
            # dtype = locate(content['type'])
            dtype = eval(content['type'])
            def_val = content['default']
            if def_val is not None:
                def_val = dtype(def_val)

            args[key] = def_val

            parser.add_argument("--"+key, action='store', type=dtype, help=content['help'], required=False)
    
    # Parse command line attributes and set non-default values
    cl_args = vars(parser.parse_args())
    
    for key, value in cl_args.items():
        if value is not None:
            args[key] = value
            if key != 'exp':
                json_args[key]['default'] = value # *anchor
    

    # *anchor
    # Keep the parameters in json format in order to persist them in the experiments folder
    if make_json:
        args['params'] = json_args
    
    # If no experiment description is given, make it empty
    if 'exp' not in args:
        args['exp'] = ''

    return args


def write_params(path, args):
    with open(path, 'w') as fp:
        json.dump(args, fp, indent=True)
