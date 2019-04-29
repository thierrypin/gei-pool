#!/usr/bin/python3
# -*- coding: utf-8 -*-


import argparse
import os
from glob import glob
import shutil



def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--folder", action='store', help="Folder where the models are saved.", type=str, default='experiments')

    return parser.parse_args()




def delete_empty(folder):
    lst = glob(os.path.join(folder, "*/"))
    for d in lst:
        print(d)
        delete = False
        ckpt = os.path.join(d, 'checkpoints')
        if os.path.isdir(ckpt):
            if not os.listdir(ckpt):
                delete = True
        
        if delete:
            shutil.rmtree(d)
            print("\tapagado")
        


def main():
    args = parse_args()
    delete_empty(args.folder)
    #



if __name__ == "__main__":
    main()




