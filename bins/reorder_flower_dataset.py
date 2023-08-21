"""

Re-order the data following class labels for analysis.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""

import os
import shutil


def my_mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    root_dir = r'E:\Machine_Learning\Dataset\Processing\102flowers'
    path_mat = r'E:\Machine_Learning\Dataset\Processing\102flowers\imagelabels.mat'
    reorder_dir = os.path.join(root_dir, 'Reorder')
    img_dir = os.path.join(root_dir, 'jpg')

    from scipy.io import loadmat
    label_array = loadmat(path_mat)['labels'].squeeze()

    names = os.listdir(img_dir)
    names = [p for p in names if p.endswith('.jpg')]

    for name in names:
        idx = int(name[6:11])
        label = label_array[idx-1]-1
        out_dir = os.path.join(reorder_dir, str(label))
        path_def = os.path.join(img_dir, name)
        my_mkdir(out_dir)
        shutil.copy(path_def, out_dir)
