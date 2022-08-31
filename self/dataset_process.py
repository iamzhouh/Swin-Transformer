import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transformers
from torch.utils.data import DataLoader

IMAGE_H = 224
IMAGE_W = 224

class FlowerDataset(data.Dataset):
    def __init__(self, mode, dir, transform = None):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = transform

        if self.mode == "train":
            dir = dir + '/train/'
        elif self.mode == "test":
            dir = dir + '/test/'
        else:
            return print('Undefined Dataset!')

        for name in os.listdir(dir):
            for file in os.listdir(dir + name):
                self.list_img.append(dir + name + '/' + file)
                self.data_size += 1

                if name == 'daisy':
                    self.list_label.append(0)
                elif name =='dandelion':
                    self.list_label.append(1)
                elif name =='roses':
                    self.list_label.append(2)
                elif name =='sunflowers':
                    self.list_label.append(3)
                elif name =='tulips':
                    self.list_label.append(4)

    def __getitem__(self, item):
        if self.mode == 'train' or 'test':
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H,IMAGE_W))
            img = np.array(img)[:, :, :3]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        else:
            print('None')

    def __len__(self):
        return self.data_size