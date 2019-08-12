# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'


import glob
import os
import numpy as np
import errno
import cv2
from tqdm import tqdm

# Creates a new directory, ignoring if it already exists
# Similar to `mkdir -p` shell command
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

        

# Read lists
lists = glob.glob('OULP-lists/*C1V2-A*')
print(lists)


# Check contents
frames_fmt = 'silhouettes/Seq0%d/%07d/%08d.png'
for l in lists:
    data = np.loadtxt(l, delimiter=',', dtype=int)
    for i, seq, start, end in data:
        s = frames_fmt%(seq, i, start)
        e = frames_fmt%(seq, i, end)
        if not os.path.isfile(s):
            print('Does not exist:', s)
        if not os.path.isfile(e):
            print('Does not exist:', e)



mkdir_p('volumes')

# Find list's properties, load data, and save numpy arrays

import re

ex = re.compile('(?<=C1V2-[AB]-)(?P<angle>\d\d|All)_(?P<set>Probe|Gallery)')

# Iterate over lists, load images and save npy
l_frames = []
for l in tqdm(lists):
    n_frames = []
    g = ex.search(l)
    angle = g.group('angle')
    set_ = g.group('set')
    data = np.loadtxt(l, delimiter=',', dtype=int)
    for i, seq, start, end in tqdm(data):
        vid = []
        for framenum in range(start, end+1):
            frame = cv2.imread(frames_fmt%(seq, i, framenum), cv2.IMREAD_GRAYSCALE)
            vid.append(cv2.resize(frame, (44, 64)))
        vid = np.array(vid)
        out_path = 'volumes/%07d-A-%s-%d-%s.npy'%(i, angle, seq, set_)
        np.save(out_path, vid)
        n_frames.append(vid.shape[0])
    l_frames.append(n_frames)

    
