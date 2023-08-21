"""

Load model function.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import vgg16_bn


def load_model(cfg, class_num, logger):
    """
    Create model
    """
    if cfg.model_name == 'resnet18':
        model = resnet18()
        if os.path.exists(cfg.path_resnet18):
            pretrained_state_dict = torch.load(cfg.path_resnet18, map_location='cpu')
            model.load_state_dict(pretrained_state_dict)
            logger.info('Load pretrained model.')
        # Modify last classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, class_num)   # 102
    elif cfg.model_name == 'vgg16_bn':
        model = vgg16_bn()
        if os.path.exists(cfg.path_vgg16_bn):
            pretrained_state_dict = torch.load(cfg.path_vgg16_bn, map_location='cpu')
            model.load_state_dict(pretrained_state_dict)
            logger.info('Load pretrained model.')
        # Modify classifier
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, class_num)    # 102
    else:
        raise Exception('Invalid model name. Trying to load {}.'.format(cfg.model_name))
    return model
