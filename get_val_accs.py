#!/usr/bin/python3
# -*- coding: utf-8 -*-


import argparse
import os
import re
from glob import glob
import shutil



def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--folder", action='store', help="Folder where the models are saved.", type=str, default='experiments')

    return parser.parse_args()


def val_accs(folder):
    regex = re.compile(r'(?:acc)(?P<acc>\d\.\d{3})(?:\.hdf5-symbol)')
    lst = glob(os.path.join(folder, "*/"))
    accs = {}
    for d in lst:
        ckpt = os.path.join(d, 'checkpoints')
        if os.path.isdir(ckpt):
            if not os.listdir(ckpt):
                shutil.rmtree(d)
                continue

            best_acc = 0.
            results = os.listdir(ckpt)
            for res in results:
                m = regex.search(res)
                if m:
                    acc = float(m.group('acc'))
                    if acc > best_acc:
                        best_acc = acc
            
            accs[d] = best_acc
    
    accs_view = sorted( [(v,k) for k,v in accs.items()], reverse=True)
    for acc in accs_view:
        print(acc)



def main():
    args = parse_args()
    val_accs(args.folder)
    #



if __name__ == "__main__":
    main()




