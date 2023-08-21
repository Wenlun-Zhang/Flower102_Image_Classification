"""

Divide original dataset into <train>, <valid>, and <test>.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""

import os
import shutil
import random


def make_dir(dir):
    """
    Create directory if target root does not exist.
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)


def copy_img(imgs, root_dir, setname):
    """
    Copy image files from original directory to target directory.
    :param imgs: Original images abs directory.
    :param root_dir: Target root directory.
    :param setname: Target set directory name.
    """
    data_dir = os.path.join(root_dir, setname)
    make_dir(data_dir)
    for path_img in imgs:
        print(path_img)
        shutil.copy(path_img, data_dir)
    print('{} dataset, copy {} imgs to {}'.format(setname, len(imgs), data_dir))


if __name__ == "__main__":      # Testbench
    # 0. Config
    # Set random seed, divide dataset with ratio [8:1:1] for [train:valid:test].
    # random_seed = 6666
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    # 1. Read list and shuffle.
    # Set root directory and data directory first.
    # random.seed(random_seed)
    root_dir = r'E:\Machine_Learning\Dataset\Processing\102flowers'
    data_dir = os.path.join(root_dir, 'jpg')
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith('.jpg')]         # List up image names.
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]            # Create abs directory for all images.
    random.shuffle(path_imgs)                                                   # Shuffle images.

    # 2. Divide list into 3 by pre-defined ratio.
    train_breakpoint = int(len(path_imgs) * train_ratio)
    valid_breakpoint = int(len(path_imgs) * (train_ratio + valid_ratio))
    train_imgs = path_imgs[:train_breakpoint]
    valid_imgs = path_imgs[train_breakpoint:valid_breakpoint]
    test_imgs = path_imgs[valid_breakpoint:]

    # 3. Copy files.
    copy_img(train_imgs, root_dir, 'Train')
    copy_img(valid_imgs, root_dir, 'Valid')
    copy_img(test_imgs, root_dir, 'Test')
