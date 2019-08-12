
import os
import glob
from subprocess import call


for i in range(1, 5):
    split_1 = 2 * i - 1
    split_2 = 2 * i

    for src_angle in [55, 65, 75, 85]:
        for dest_angle in [55, 65, 75, 85]:
            for model in ['geinet', 'no_pool_geinet', 'no_pool']:
                print("%d-%d - %d > %d: %s" % (split_1, split_2, src_angle, dest_angle, model))

                # split 1 > 2
                # print('experiments/OULP-%s-%d-%d*' % (model, split_1, src_angle))
                lst = glob.glob('experiments/OULP-%s-%02d-%d*' % (model, split_1, src_angle))
                print(lst)
                ckpt_folder = os.path.join(lst[0], 'checkpoints')
                call(['python', 'test.py', '--ckpt_folder', ckpt_folder, '--dest_fold', "%02d"%(split_2), '--dest_angle', str(dest_angle)])

                # split 2 > 1
                lst = glob.glob('experiments/OULP-%s-%02d-%d*' % (model, split_2, src_angle))
                ckpt_folder = os.path.join(lst[0], 'checkpoints')
                call(['python', 'test.py', '--ckpt_folder', ckpt_folder, '--dest_fold', "%02d"%(split_1), '--dest_angle', str(dest_angle)])






