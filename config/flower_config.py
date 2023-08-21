"""

Parameter management config for <102 Category Flower Dataset> classification.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""

import os

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from easydict import EasyDict


cfg = EasyDict()
# Obtain value via .key like key-value pair

# cfg.model_name = 'resnet18'
cfg.model_name = 'vgg16_bn'

cfg.data_dir = r'E:\Machine_Learning\Dataset\Processing\102flowers'
cfg.path_resnet18 = r'E:\Machine_Learning\Model\resnet18\resnet18-f37072fd.pth'
cfg.path_vgg16_bn = r'E:\Machine_Learning\Model\vgg16_bn\vgg16_bn-6c64b313.pth'

cfg.train_bs = 16
cfg.valid_bs = 16
cfg.workers = 8

cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1    # gamma (lr decay param) when using MultiStepLR
cfg.milestones = [30, 45]
cfg.max_epoch = 50

cfg.log_interval = 10

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(norm_mean, norm_std)

cfg.transforms_train = transforms.Compose([
    transforms.Resize((256)),       # Shorter edge = 256
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normTransform,
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform,
])

if __name__ == '__main__':      # Testbench

    from dataset.flower_102 import FlowerDataset
    from torch.utils.data import DataLoader
    from tools.common_tools import inverse_transform
    train_dir = os.path.join(cfg.data_dir, 'Train')
    train_data = FlowerDataset(root_dir=train_dir, transform=cfg.transforms_train)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True)

    for epoch in range(cfg.max_epoch):
        for i, data in enumerate(train_loader):

            inputs, labels, dir = data       # B C H W

            img_tensor = inputs[0, ...]     # C H W
            img = inverse_transform(img_tensor, cfg.transforms_train)
            plt.imshow(img)
            plt.show()
            plt.pause(0.5)
            plt.close()
