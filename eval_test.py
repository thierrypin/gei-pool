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
        exp_folder = 'tests'
        wilcoxon_stats = [('gallery_time', 'probe_time', 'nn_fit_time', 'nn_predict_time', 'nn_acc', 'opf_fit_time', 'opf_predict_time', 'opf_acc')]
        stats = [('gallery_time', 'probe_time', 'nn_fit_time', 'nn_predict_time', 'nn_acc', 'opf_fit_time', 'opf_predict_time', 'opf_acc')]

        for angle_from in [55, 65, 75, 85]:

            for angle_to in [55, 65, 75, 85]:
                
                infos = []
                for fold in range(1, 6):
                    f1 = 2*fold - 1
                    f2 = 2*fold
                    exp = "%s/OULP-%s_%02d>%02d_%02d>%02d" % (exp_folder, model, f1, f2, angle_from, angle_to)
                    print(exp)
                    info1 = np.loadtxt(os.path.join(exp, 'info.csv'), delimiter=';', skiprows=1)

                    exp = "%s/OULP-%s_%02d>%02d_%02d>%02d" % (exp_folder, model, f1, f2, angle_from, angle_to)
                    info2 = np.loadtxt(os.path.join(exp, 'info.csv'), delimiter=';', skiprows=1)

                    wilcoxon_stats.append(info1)
                    wilcoxon_stats.append(info2)
                    infos.append(np.vstack((info1, info2)).mean(axis=0))
                
                stats.append(np.vstack(infos).mean(axis=0))
        
        # wilcoxon_stats = np.array(wilcoxon_stats)
        
        np.savetxt('%s_test_wilcoxon_stats.csv' % model, np.array(wilcoxon_stats), delimiter=';', fmt='%s')
        np.savetxt('%s_test_stats.csv' % model, np.array(stats), delimiter=';', fmt='%s')




if __name__ == "__main__":
    main()


