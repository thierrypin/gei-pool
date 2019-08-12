#!/home/thierry/anaconda3/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import re
import os
from glob import glob

import numpy as np


def main():

    models = ['geinet', 'no_pool_geinet', 'no_pool']
    for model in models:
        exp_folder = 'experiments'
        stats = [('acc', 'sum_train', 'mean_train', 'sum_val', 'mean_val')]

        for fold in range(1, 11):

            for angle in [55, 65, 75, 85]:
                prefix = "%s/OULP-%s-%02d-%d_*" % (exp_folder, model, fold, angle)
                exp = glob(prefix)[0]
                print(exp)
            
                # Find the best checkpoint
                regex = re.compile(r'acc(?P<acc>\d\.\d{3})\.')
                ckpt_folder = os.path.join(exp, 'checkpoints')
                acc = 0
                for p in os.listdir(ckpt_folder):
                    m = regex.search(p)
                    if m:
                        e = float(m.group('acc'))
                        if acc < e:
                            acc = e

                info = np.loadtxt(os.path.join(exp, 'history.csv'), delimiter=';', skiprows=1)
                sum_train = info[:, 2].sum()
                mean_train = info[:, 2].mean()

                sum_val = info[:, 3].sum()
                mean_val = info[:, 3].mean()

                stats.append((acc, sum_train, mean_train, sum_val, mean_val))
        
        np.savetxt('%s_stats.csv' % model, np.array(stats), delimiter=';', fmt='%s')




if __name__ == "__main__":
    main()


