# -*- coding: utf-8 -*-

import numpy as np
import os
import re
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit


path = 'volumes'

for i in range(10):
    list_path = 'volumes/CV%02d.txt' % (i + 1)
    persons = np.loadtxt(list_path, delimiter=';', dtype=str)

    expr = r"(\d{7}).*\d{2}-\d-Gallery"
    gallery = []

    for f in tqdm(os.listdir(path)):
        # print(f)
        m = re.match(expr, f)

        # group(1) refers to '(\d{7})'
        if m and m.group(1) in persons:
            gallery.append(f)
    
    gallery = np.array(gallery)

    expr = r"(\d{7}).*\d{2}-\d-Probe"
    probe = []
    labels = []

    for f in tqdm(os.listdir(path)):
        # print(f)
        m = re.match(expr, f)

        # group(1) refers to '(\d{7})'
        if m and m.group(1) in persons:
            probe.append(f)
            labels.append(m.group(1))

    probe = np.array(probe)
    labels = np.array(labels)

    # Select half the probe videos to validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    probe_train, probe_val = next(sss.split(probe, labels))
    
    train = np.hstack((gallery, probe[probe_train]))
    val = probe[probe_val]
    
    np.savetxt(list_path+'_train', train, fmt='%s')
    np.savetxt(list_path+'_val', val, fmt='%s')
    














