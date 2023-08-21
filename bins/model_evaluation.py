"""

Evaluate model on test dataset.

Source: DeepShare Community

Edit by Wenlun-Zhang

"""
import torch
import numpy as np
import torch.nn as nn
from dataset.flower_102 import FlowerDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Step 0: config
    data_dir = r'E:\Machine_Learning\Dataset\Processing\102flowers\Test'      # Test data directory
    path_state_dict = r'E:\Machine_Learning\Project\Flower102_Image_Classification\run\23-08-19_22-13\checkpoint_best.pkl'       # Pickle file directory

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform,
    ])

    bs = 64
    workers = 4

    # Step 1: Dataset
    test_data = FlowerDataset(root_dir=data_dir, transform=transforms_test)
    test_loader = DataLoader(dataset=test_data, batch_size=bs, num_workers=workers)

    # Step 2: Model
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, test_data.class_num)
    # Load pretrained model
    checkpoint = torch.load(path_state_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Step 3: Inference
    class_num = test_loader.dataset.class_num
    conf_mat = np.zeros((class_num, class_num))

    for i, data in enumerate(test_loader):
        # inputs, labels = data
        inputs, labels, path_img = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        # Generate confusion matrix
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.

    acc_avg = conf_mat.trace() / conf_mat.sum()
    print('Test Acc: {:.2%}'.format(acc_avg))
