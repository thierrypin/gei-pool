# -*- coding: utf-8 -*-

import numpy as np
import os
import re
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit


path = 'volumes'
lst = os.listdir(path)

for i in tqdm(range(10)):
    list_path = 'volumes/CV%02d.txt' % (i + 1)
    persons = np.loadtxt(list_path, delimiter=';', dtype=str)

    for angle in ['55', '65', '75', '85']:
        expr = r"(\d{7}).*" + angle + r"-\d-Gallery"
        gallery = []
        # labels = []

        for f in lst:
            # print(f)
            m = re.match(expr, f)

            # group(1) refers to '(\d{7})'
            if m and m.group(1) in persons:
                gallery.append(f)
                # labels.append(m.group(1))
        
        gallery = np.array(gallery)

        expr = r"(\d{7}).*" + angle + r"-\d-Probe"
        probe = []

        for f in lst:
            # print(f)
            m = re.match(expr, f)

            # group(1) refers to '(\d{7})'
            if m and m.group(1) in persons:
                probe.append(f)
                # labels.append(m.group(1))

        probe = np.array(probe)
        # labels = np.array(labels)
        full = np.hstack((gallery, probe))
        

        np.savetxt(list_path+'_gallery_'+angle, gallery, fmt='%s')
        np.savetxt(list_path+'_probe_'+angle, probe, fmt='%s')
        np.savetxt(list_path+'_full_'+angle, full, fmt='%s')
    














