"""

Dataset file of image classification task for <102 Category Flower Dataset>.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""

import os
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):

    class_num = 102                                     # Number of flower category
    names = tuple([i for i in range(class_num)])

    def __init__(self, root_dir, transform=None):
        """
        Get directory of dataset and pre-processing methods.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []                              # [(dir, label), ... , ]
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        """
        Input an index (scalar), read data from disk, transform and convert to tensor.
        """
        img_dir, label = self.img_info[index]
        img = Image.open(img_dir).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_dir

    def __len__(self):
        """
        Assert for debug.
        """
        if len(self.img_info) == 0:
            raise Exception('\ndata_dir: {} is an empty directory! Please check your directory settings!'.format(
                self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        """
        Read dataset, and summarize the directory & label in a list (img_info).
        [(dir, label), ... ,]
        """
        # Obtain all image file name into a list.
        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith('.jpg')]

        # Read label from .mat file.
        label_file = 'imagelabels.mat'
        label_file_dir = os.path.join(self.root_dir, '..', label_file)      # Copy label file here!
        from scipy.io import loadmat
        # Indexing original dict with key == 'label' and squeeze down to 1-D Array.
        label_array = loadmat(label_file_dir)['labels'].squeeze()
        self.label_array = label_array

        # Label pairing
        idx_img = [int(idx[6:11]) for idx in names_imgs]
        img_dir = [os.path.join(self.root_dir, n) for n in names_imgs]      # List up image abs directory.
        # Get image dir with selected index in target folder. Label start with 0.
        self.img_info = [(p, int(label_array[idx-1]-1)) for p, idx in zip(img_dir, idx_img)]


if __name__ == "__main__":      # Testbench

    root_dir = r'E:\Machine_Learning\Dataset\Processing\102flowers\Train'
    test_dataset = FlowerDataset(root_dir)

    print(len(test_dataset))
    print(next(iter(test_dataset)))
