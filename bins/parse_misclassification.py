"""

Parsing misclassified images for root cause investigation.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""
import os
import pickle
import shutil


def load_pickle(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    return data


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':
    path_pkl = r'E:\Machine_Learning\Project\Flower102_Image_Classification\run\23-08-19_23-46\misc_img_50.pkl'
    data_root_dir = r'E:\Machine_Learning\Dataset\Processing\102flowers\jpg'
    out_dir = path_pkl[:-4]     # Output files directory
    error_info = load_pickle(path_pkl)

    for setname, info in error_info.items():
        for imgs_data in info:
            label, pred, path_img_rel = imgs_data
            path_img = os.path.join(data_root_dir, os.path.basename(path_img_rel))
            img_dir = os.path.join(out_dir, setname, str(label), str(pred))     # Image folder
            my_mkdir(img_dir)
            shutil.copy(path_img, img_dir)      # Copy files
