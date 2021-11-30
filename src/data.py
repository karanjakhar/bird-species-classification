from torch.nn.modules.conv import Conv2d
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from config import CONFIG


def imshow(img):
    img = (img * 0.5) + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class_dict = pd.read_csv('../data/class_dict.csv')
classes = class_dict['class']

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                     0.229, 0.224, 0.225]),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomAdjustSharpness(2),
                                transforms.RandomAutocontrast(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(
                                    brightness=3, contrast=4, saturation=2, hue=0.5)
                                ])
valid_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                          0.229, 0.224, 0.225])
                                      ])

bird_dataset_train = torchvision.datasets.ImageFolder(
    '../data/train', transform=transform)
bird_dataset_valid = torchvision.datasets.ImageFolder(
    '../data/valid/', transform=valid_transform)
bird_test_dataset = torchvision.datasets.ImageFolder(
    '../data/test/', transform=valid_transform)

bird_train_dataloader = torch.utils.data.DataLoader(
    bird_dataset_train, batch_size=CONFIG['batch_size'])
bird_valid_dataloader = torch.utils.data.DataLoader(
    bird_dataset_valid, batch_size=CONFIG['batch_size'])
bird_test_dataloader = torch.utils.data.DataLoader(
    bird_test_dataset, batch_size=CONFIG['batch_size'])


# MNIST Dataset
mnist_train_dataset = torchvision.datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), transforms.Resize((64, 64))]),
)

mnist_valid_dataset = torchvision.datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), transforms.Resize((64, 64))]),
)

mnist_train_dataloader = torch.utils.data.DataLoader(
    mnist_train_dataset, batch_size=CONFIG['batch_size'])
mnist_valid_dataloader = torch.utils.data.DataLoader(
    mnist_valid_dataset, batch_size=CONFIG['batch_size'])


if __name__ == '__main__':
    images, labels = iter(mnist_train_dataloader).next()
    imshow(torchvision.utils.make_grid(images))
    print(' || '.join(f'{classes[labels.tolist()[j]]}' for j in range(4)))
